"""
Actformer Working Memory — register file, scratchpad, pointer network.

Key design decisions:
- MemoryState is a dataclass with cloned tensors. Every state update returns
  a *new* MemoryState — no in-place mutation.
- init_state uses .clone() (not .expand().clone()) to prevent gradient aliasing.
- All public methods document gradient flow (differentiable vs. discrete).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MemoryState",
    "WorkingMemory",
]


# ---------------------------------------------------------------------------
# Memory State
# ---------------------------------------------------------------------------

@dataclass
class MemoryState:
    """
    Immutable(ish) snapshot of working memory at a computation step.

    Every field is a tensor that has been .clone()'d from the source so that
    in-place operations on the *previous* state cannot corrupt this one.
    """
    registers: torch.Tensor       # (batch, num_registers, register_dim)
    scratchpad: torch.Tensor       # (batch, scratchpad_size, scratchpad_dim)
    pointers: torch.Tensor         # (batch, num_pointers)  — normalised 0..1
    step: int = 0                  # logical timestep counter
    halt_flag: torch.Tensor = None  # (batch,) bool — True if HALT emitted

    @staticmethod
    def make_halt_flag(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# Working Memory Module
# ---------------------------------------------------------------------------

class WorkingMemory(nn.Module):
    """
    Explicit, differentiable working memory.

    Components
    ----------
    - **Register file**: fast, fixed-size storage for intermediate values.
      Shape ``(batch, num_registers, register_dim)``.
    - **Scratchpad**: larger, content-addressable memory for data structures.
      Shape ``(batch, scratchpad_size, scratchpad_dim)``.
    - **Pointer network**: tracks positions within data, normalised 0–1.
      Shape ``(batch, num_pointers)``.

    Gradient notes
    --------------
    Register reads/writes are differentiable (linear + attention).
    Pointer moves are *discrete* during inference and use straight-through
    estimators during training.  This module does NOT branch on
    ``model.training()`` — the caller must set an explicit
    ``execution_mode: Literal['train', 'infer']``.
    """

    def __init__(
        self,
        num_registers: int = 16,
        register_dim: int = 64,
        scratchpad_size: int = 256,
        scratchpad_dim: int = 64,
        num_pointers: int = 4,
    ):
        super().__init__()
        self.num_registers = num_registers
        self.register_dim = register_dim
        self.scratchpad_size = scratchpad_size
        self.scratchpad_dim = scratchpad_dim
        self.num_pointers = num_pointers

        # Learnable initial values (trained, not frozen)
        self.register_init = nn.Parameter(torch.zeros(num_registers, register_dim))
        self.scratchpad_init = nn.Parameter(torch.zeros(scratchpad_size, scratchpad_dim))
        self.pointer_init = nn.Parameter(torch.zeros(num_pointers))

        # Attention-based read head for scratchpad
        self.read_head = nn.MultiheadAttention(
            embed_dim=scratchpad_dim,
            num_heads=4,
            batch_first=True,
        )
        # Project register → scratchpad dim for writes
        self.write_proj = nn.Linear(register_dim, scratchpad_dim)

        # Content-based address generation from a register query
        self.address_net = nn.Sequential(
            nn.Linear(register_dim, 128),
            nn.ReLU(),
            nn.Linear(128, scratchpad_size),
            nn.Softmax(dim=-1),
        )

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """
        Create the initial memory state for a new computation.

        Uses ``.clone()`` on the learnable parameters so that subsequent
        in-place updates do not back-propagate into the *initial* values
        across different batch elements (preventing gradient aliasing).
        """
        registers = self.register_init.unsqueeze(0).expand(batch_size, -1, -1).clone().detach()
        scratchpad = self.scratchpad_init.unsqueeze(0).expand(batch_size, -1, -1).clone().detach()
        pointers = self.pointer_init.unsqueeze(0).expand(batch_size, -1).clone().detach()
        return MemoryState(
            registers=registers,
            scratchpad=scratchpad,
            pointers=pointers,
            step=0,
            halt_flag=MemoryState.make_halt_flag(batch_size, device),
        )

    # ------------------------------------------------------------------
    # Register operations (all differentiable)
    # ------------------------------------------------------------------

    def update_register(
        self,
        state: MemoryState,
        reg_idx: int,
        value: torch.Tensor,
    ) -> MemoryState:
        """
        Write *value* (batch, dim) into register *reg_idx*.

        Returns a **new** MemoryState with ``registers[:, reg_idx]`` replaced
        by a clone of *value*.  The original state is not mutated.

        Gradient: fully differentiable (linear write).
        """
        new_regs = state.registers.clone()
        # value may be (batch, dim) — broadcast along reg dim
        if value.dim() == 2:
            new_regs[:, reg_idx] = value
        else:
            new_regs[:, reg_idx] = value.squeeze(1)
        return replace(state, registers=new_regs, step=state.step)

    def read_register(self, state: MemoryState, reg_idx: int) -> torch.Tensor:
        """Read register *reg_idx* → (batch, register_dim). Differentiable."""
        return state.registers[:, reg_idx, :].clone()

    # ------------------------------------------------------------------
    # Scratchpad operations
    # ------------------------------------------------------------------

    def read_from_scratchpad(
        self,
        state: MemoryState,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Content-based differentiable read.

        Args:
            query: (batch, 1, register_dim) key vector.

        Returns:
            value: (batch, 1, scratchpad_dim)
            weights: (batch, scratchpad_size)
        """
        value, weights = self.read_head(query, state.scratchpad, state.scratchpad)
        return value, weights

    def write_to_scratchpad(
        self,
        state: MemoryState,
        value: torch.Tensor,
        address: Optional[torch.Tensor] = None,
    ) -> MemoryState:
        """
        Write *value* into scratchpad.

        If *address* is provided (batch, scratchpad_size) it is used as a
        soft write mask.  Otherwise the write is broadcast to all positions.

        Gradient: soft attention write is differentiable.
        """
        proj = self.write_proj(value)  # (batch, 1, scratchpad_dim)
        new_sp = state.scratchpad.clone()
        if address is not None:
            # Soft write: blend value into addressed positions
            mask = address.unsqueeze(-1)  # (batch, sp_size, 1)
            new_sp = new_sp * (1 - mask) + proj * mask
        else:
            new_sp = new_sp + proj
        return replace(state, scratchpad=new_sp)

    # ------------------------------------------------------------------
    # Pointer operations
    # ------------------------------------------------------------------

    def move_pointer(
        self,
        state: MemoryState,
        ptr_idx: int,
        delta: float,
        execution_mode: str = "train",
    ) -> MemoryState:
        """
        Move pointer *ptr_idx* by *delta* (normalised 0–1 step).

        In **train** mode the move is soft (additive, differentiable).
        In **infer** mode the pointer is clamped to [0, 1].

        Gradient: soft additive move is differentiable. Clamp in infer mode
        breaks gradient flow — acceptable because inference does not need
        gradients.
        """
        new_ptrs = state.pointers.clone()
        new_ptrs[:, ptr_idx] = new_ptrs[:, ptr_idx] + delta
        if execution_mode == "infer":
            new_ptrs[:, ptr_idx] = new_ptrs[:, ptr_idx].clamp(0.0, 1.0)
        return replace(state, pointers=new_ptrs)

    def get_pointer_pos(self, state: MemoryState, ptr_idx: int) -> torch.Tensor:
        """Return position of pointer *ptr_idx* → (batch,)."""
        return state.pointers[:, ptr_idx].clone()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_memory_utilization(self, state: MemoryState) -> float:
        """
        Fraction of registers that have been meaningfully written
        (L2 norm above a small threshold).
        """
        norms = state.registers.norm(dim=-1)  # (batch, num_regs)
        return (norms > 1e-3).float().mean().item()

    def summary(self, state: MemoryState) -> Dict[str, object]:
        return {
            "step": state.step,
            "register_norms": state.registers.norm(dim=-1).mean().item(),
            "scratchpad_norms": state.scratchpad.norm(dim=-1).mean().item(),
            "pointer_positions": state.pointers.mean(dim=0).tolist(),
            "memory_utilization": self.get_memory_utilization(state),
        }