"""
Actformer Action Execution Engine — executes ActionTokens against WorkingMemory.

Design principles:
  1. Every public method returns a *new* MemoryState — no in-place mutation.
  2. ``execution_mode`` is an explicit parameter ('train' or 'infer'), NOT
     derived from ``model.training()``.  This prevents subtle bugs where
     ``model.eval()`` silently changes execution semantics.
  3. Control-flow actions (IF, LOOP, BREAK) use soft branching during
     training (so gradients can flow) and hard branching during inference.
  4. Every non-trivial method has a docstring documenting gradient flow.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_space import (
    ActionToken,
    ActionType,
    ActionSpace,
)
from .primitives import DifferentiablePrimitives
from .working_memory import MemoryState, WorkingMemory

__all__ = [
    "ActionExecutionEngine",
]


class ActionExecutionEngine(nn.Module):
    """
    Dispatches an ActionToken to the appropriate primitive operation,
    updates MemoryState, and returns the new state.

    All 15+ ActionTypes are handled.  Unrecognised op_ids are treated as NOPs
    (no state change) rather than raising — this keeps rollouts stable during
    exploration when the action predictor emits novel tokens.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        working_memory: WorkingMemory,
        primitive_dim: int = 64,
    ):
        super().__init__()
        self.action_space = action_space
        self.wm = working_memory
        self.primitives = DifferentiablePrimitives(dim=primitive_dim)

        # Soft gate parameters for control flow (trained)
        self.if_gate = nn.Parameter(torch.tensor(0.0))
        self.loop_gate = nn.Parameter(torch.tensor(0.0))

        # Scratchpad address cache for sequential writes
        self._scratchpad_write_pos: int = 0

        # Learned projection: scratchpad_dim → register_dim (replaces random hack)
        self.scratchpad_proj = nn.Linear(primitive_dim, primitive_dim)

    def reset_scratchpad_pos(self) -> None:
        self._scratchpad_write_pos = 0

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def execute(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str = "train",
    ) -> MemoryState:
        """
        Execute *action* on *state* and return a new MemoryState.

        Args:
            action: Structured action token.
            state: Current working memory (not mutated).
            execution_mode: ``'train'`` (soft branches, STE) or ``'infer'``
                (hard branches, argmax).

        Returns:
            New MemoryState with the action applied.
        """
        op = ActionType(action.op_id) if action.op_id < len(ActionType) else None

        if op is None or op == ActionType.NOP:
            return state

        # Clamp args to valid register range (model output head can produce OOB indices)
        nregs = self.wm.num_registers
        safe_action = ActionToken(
            op_id=action.op_id,
            arg0=action.arg0 % nregs,
            arg1=action.arg1 % nregs,
            arg2=action.arg2 % nregs,
            modifier=action.modifier,
        )

        dispatch = {
            ActionType.READ:         self._exec_read,
            ActionType.WRITE:        self._exec_write,
            ActionType.ADD:          self._exec_add,
            ActionType.SUBTRACT:     self._exec_subtract,
            ActionType.MULTIPLY:     self._exec_multiply,
            ActionType.COMPARE:      self._exec_compare,
            ActionType.LOAD:         self._exec_load,
            ActionType.STORE:        self._exec_store,
            ActionType.POINTER_MOVE: self._exec_pointer_move,
            ActionType.IF:           self._exec_if,
            ActionType.LOOP:         self._exec_loop,
            ActionType.BREAK:        self._exec_break,
            ActionType.OUTPUT:       self._exec_output,
            ActionType.HALT:         self._exec_halt,
            ActionType.DIGIT_EXTRACT: self._exec_digit_extract,
            ActionType.DIGIT_PACK:  self._exec_digit_pack,
            ActionType.SHIFT_LEFT:  self._exec_shift_left,
            ActionType.SHIFT_RIGHT: self._exec_shift_right,
            ActionType.MAX_OP:       self._exec_max,
            ActionType.MIN_OP:       self._exec_min,
            ActionType.MOD_OP:       self._exec_mod,
            ActionType.DIVIDE_SAFE:  self._exec_divide_safe,
            ActionType.CALL_TOOL:    self._exec_nop,
        }

        handler = dispatch.get(op, self._exec_nop)
        return handler(safe_action, state, execution_mode)

    # ------------------------------------------------------------------
    # Primitive operations (all fully differentiable)
    # ------------------------------------------------------------------

    def _exec_read(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        READ dst_reg, src_reg, ptr_idx

        Read from scratchpad (content-addressed by src_reg query) into dst_reg.
        Gradient: attention read is differentiable.
        """
        dst_reg = action.arg0
        src_reg = action.arg1
        _ptr_idx = action.arg2

        query = state.registers[:, src_reg, :].unsqueeze(1)  # (batch, 1, dim)
        value, _weights = self.wm.read_from_scratchpad(state, query)
        # Project scratchpad_dim → register_dim if needed
        value = value.squeeze(1)  # (batch, scratchpad_dim)
        if value.shape[-1] != self.wm.register_dim:
            value = self.scratchpad_proj(value)  # learned projection
        return self.wm.update_register(state, dst_reg, value)

    def _exec_write(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        WRITE src_reg, dst_reg

        Write from register to scratchpad at next available position.
        Gradient: soft write is differentiable.
        """
        src_reg = action.arg0
        value = state.registers[:, src_reg, :].unsqueeze(1)
        return self.wm.write_to_scratchpad(state, value)

    def _exec_add(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        ADD src_a, src_b, dst, modifier

        Element-wise add reg[src_a] + reg[src_b] → reg[dst].
        Modifier=1 means "add with carry" (uses reg[dst+1 mod N] as carry).
        Gradient: fully differentiable (dL/dsrc = dL/dout, additive).
        """
        src_a = action.arg0
        src_b = action.arg1
        dst = action.arg2
        carry_mod = action.modifier

        val_a = self.wm.read_register(state, src_a)
        val_b = self.wm.read_register(state, src_b)

        result = self.primitives.add(val_a, val_b)

        if carry_mod == 1:
            # Add carry from the next register
            carry_reg = (dst + 1) % self.wm.num_registers
            carry_val = self.wm.read_register(state, carry_reg)
            result = self.primitives.add(result, carry_val)

        return self.wm.update_register(state, dst, result)

    def _exec_subtract(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """SUBTRACT src_a, src_b, dst → reg[dst] = reg[src_a] - reg[src_b]."""
        val_a = self.wm.read_register(state, action.arg0)
        val_b = self.wm.read_register(state, action.arg1)
        result = self.primitives.subtract(val_a, val_b)
        return self.wm.update_register(state, action.arg2, result)

    def _exec_multiply(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """MULTIPLY src_a, src_b, dst → reg[dst] = reg[src_a] * reg[src_b]."""
        val_a = self.wm.read_register(state, action.arg0)
        val_b = self.wm.read_register(state, action.arg1)
        result = self.primitives.multiply(val_a, val_b)
        return self.wm.update_register(state, action.arg2, result)

    def _exec_compare(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        COMPARE src_a, src_b, dst → reg[dst] ≈ 1 if reg[src_a] > reg[src_b] else ≈ 0.
        Soft comparison (sigmoid). Differentiable.
        """
        val_a = self.wm.read_register(state, action.arg0)
        val_b = self.wm.read_register(state, action.arg1)
        result = self.primitives.compare(val_a, val_b)
        return self.wm.update_register(state, action.arg2, result)

    def _exec_max(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        val_a = self.wm.read_register(state, action.arg0)
        val_b = self.wm.read_register(state, action.arg1)
        result = self.primitives.max_soft(val_a, val_b)
        return self.wm.update_register(state, action.arg2, result)

    def _exec_min(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        val_a = self.wm.read_register(state, action.arg0)
        val_b = self.wm.read_register(state, action.arg1)
        result = self.primitives.min_soft(val_a, val_b)
        return self.wm.update_register(state, action.arg2, result)

    def _exec_divide_safe(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        val_a = self.wm.read_register(state, action.arg0)
        val_b = self.wm.read_register(state, action.arg1)
        result = self.primitives.divide_safe(val_a, val_b)
        return self.wm.update_register(state, action.arg2, result)

    def _exec_mod(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        val_a = self.wm.read_register(state, action.arg0)
        val_b = self.wm.read_register(state, action.arg1)
        result = self.primitives.mod_op(val_a, val_b)
        return self.wm.update_register(state, action.arg2, result)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def _exec_load(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        LOAD value_reg, target_reg

        Copy the scalar encoded in value_reg[0] into target_reg as an
        embedding (first component = scalar, rest = 0).
        Gradient: copy is differentiable.
        """
        value_reg = action.arg0
        target_reg = action.arg1
        val = self.wm.read_register(state, value_reg)
        return self.wm.update_register(state, target_reg, val)

    def _exec_store(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        STORE src_reg, scratchpad_pos

        Write register to scratchpad at position derived from modifier.
        Gradient: soft write is differentiable.
        """
        src_reg = action.arg0
        value = state.registers[:, src_reg, :].unsqueeze(1)
        return self.wm.write_to_scratchpad(state, value)

    def _exec_pointer_move(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        POINTER_MOVE ptr_idx, direction, amount

        Move pointer by direction * amount * step_size.
        Gradient: soft additive move in train mode.
        """
        ptr_idx = action.arg0 % self.wm.num_pointers
        direction = action.arg1   # +1 or -1
        amount = action.arg2 + 1  # 1..32 steps
        step_size = 0.1 / max(self.wm.scratchpad_size, 1)
        delta = direction * amount * step_size
        return self.wm.move_pointer(state, ptr_idx, delta, execution_mode)

    # ------------------------------------------------------------------
    # Control flow (soft in train, hard in infer)
    # ------------------------------------------------------------------

    def _exec_if(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        IF cond_reg, modifier

        In train mode: soft gate — multiplies next-state by σ(condition).
        In infer mode: no state change (actual branching is handled by the
        action loop's trace logic, not by memory mutation).
        Gradient: soft gate is differentiable.
        """
        if execution_mode == "train":
            cond_reg = action.arg0
            cond_val = self.wm.read_register(state, cond_reg)
            gate = torch.sigmoid(cond_val.mean(dim=-1, keepdim=True)).unsqueeze(1)  # (batch,1,1)
            # Apply gate to all registers (soft conditional)
            gated_regs = state.registers * gate
            return MemoryState(
                registers=gated_regs,
                scratchpad=state.scratchpad.clone(),
                pointers=state.pointers.clone(),
                step=state.step,
                halt_flag=state.halt_flag,
            )
        return state  # inference: branching is external

    def _exec_loop(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        LOOP counter_reg, limit_reg

        In train mode: increment counter with soft clamp.
        In infer mode: no state change (loop control is external).
        """
        if execution_mode == "train":
            counter_reg = action.arg0
            val = self.wm.read_register(state, counter_reg)
            new_val = val + self.primitives.scalar_to_embedding(
                torch.ones(val.shape[0], 1, device=val.device), self.wm.register_dim
            ) * 0.01
            return self.wm.update_register(state, counter_reg, new_val)
        return state

    def _exec_break(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """BREAK — sets halt flag in inference mode. No-op in train mode."""
        if execution_mode == "infer":
            new_halt = state.halt_flag.clone()
            new_halt[:] = True
            return MemoryState(
                registers=state.registers.clone(),
                scratchpad=state.scratchpad.clone(),
                pointers=state.pointers.clone(),
                step=state.step,
                halt_flag=new_halt,
            )
        return state

    # ------------------------------------------------------------------
    # Digit operations
    # ------------------------------------------------------------------

    def _exec_digit_extract(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        DIGIT_EXTRACT src_reg, dst_reg, digit_pos

        Extract the digit_pos-th digit from src_reg into dst_reg.
        Straight-through estimator: forward is discrete, backward is identity.
        """
        src_val = self.wm.read_register(state, action.arg0)
        digit_pos = action.arg2
        result = self.primitives.digit_extract(src_val, digit_pos)
        return self.wm.update_register(state, action.arg1, result)

    def _exec_digit_pack(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        DIGIT_PACK src_reg, dst_reg, digit_pos

        Pack digit from src_reg into digit_pos of accumulator in dst_reg.
        """
        src_val = self.wm.read_register(state, action.arg0)
        acc_val = self.wm.read_register(state, action.arg1)
        result = self.primitives.digit_pack(src_val, action.arg2, acc_val)
        return self.wm.update_register(state, action.arg1, result)

    def _exec_shift_left(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        val = self.wm.read_register(state, action.arg0)
        result = self.primitives.shift_left(val, shift=action.arg2 + 1)
        return self.wm.update_register(state, action.arg1, result)

    def _exec_shift_right(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        val = self.wm.read_register(state, action.arg0)
        result = self.primitives.shift_right(val, shift=action.arg2 + 1)
        return self.wm.update_register(state, action.arg1, result)

    # ------------------------------------------------------------------
    # Meta operations
    # ------------------------------------------------------------------

    def _exec_output(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """
        OUTPUT reg — no state change; the output decoder reads from the
        specified register at decode time.  This action is informational
        (the action predictor learns when to emit it).
        """
        return state

    def _exec_halt(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """HALT — sets halt flag to stop the action loop."""
        new_halt = state.halt_flag.clone()
        new_halt[:] = True
        return MemoryState(
            registers=state.registers.clone(),
            scratchpad=state.scratchpad.clone(),
            pointers=state.pointers.clone(),
            step=state.step,
            halt_flag=new_halt,
        )

    def _exec_nop(
        self,
        action: ActionToken,
        state: MemoryState,
        execution_mode: str,
    ) -> MemoryState:
        """NOP — no state change."""
        return state

    # ------------------------------------------------------------------
    # Macro expansion
    # ------------------------------------------------------------------

    def execute_macro(
        self,
        sub_actions: List[ActionToken],
        state: MemoryState,
        execution_mode: str = "train",
    ) -> MemoryState:
        """
        Expand and execute a macro action sequence in-line.

        Returns the final MemoryState after all sub-actions.
        """
        for sub_action in sub_actions:
            state = self.execute(sub_action, state, execution_mode)
        return state