"""
Actformer Action Space — structured action tokens, factorized embeddings, and macro action support.

An action token is a fixed-length structured vector:
    [OP_ID (int), ARG_0 (int), ARG_1 (int), ARG_2 (int), MODIFIER (int)]

Encoding: e = E_op(op_id) + E_arg0(arg0) + E_arg1(arg1) + E_arg2(arg2) + E_mod(mod)
Vocab sizes: op_id=64, each arg=32 (register/pointer indices), modifier=16
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ActionType",
    "ActionToken",
    "FactorizedActionEmbedding",
    "ActionSpace",
    "MacroAction",
]


# ---------------------------------------------------------------------------
# Action Types
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    """Categories of computational actions the model can perform."""
    # Primitive Operations
    READ = 0
    WRITE = 1
    ADD = 2
    SUBTRACT = 3
    MULTIPLY = 4
    COMPARE = 5
    # Memory Operations
    LOAD = 6
    STORE = 7
    POINTER_MOVE = 8
    # Control Flow
    IF = 9
    LOOP = 10
    BREAK = 11
    # Meta Operations
    OUTPUT = 12
    HALT = 13
    CALL_TOOL = 14
    # Extended
    DIGIT_EXTRACT = 15
    DIGIT_PACK = 16
    SHIFT_LEFT = 17
    SHIFT_RIGHT = 18
    MAX_OP = 19
    MIN_OP = 20
    MOD_OP = 21
    DIVIDE_SAFE = 22
    NOP = 23  # no-op for padding


NUM_ACTION_TYPES = len(ActionType)  # 24


# ---------------------------------------------------------------------------
# Structured Action Token
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionToken:
    """
    Structured, deterministic action representation.

    Fields map directly to embedding table indices — no hashing, no collisions.
    """
    op_id: int       # index into OP vocabulary (0..NUM_ACTION_TYPES-1)
    arg0: int = 0    # register / pointer / value index (0..31)
    arg1: int = 0
    arg2: int = 0
    modifier: int = 0  # flags: e.g. carry-bit, branch condition (0..15)

    def __post_init__(self):
        # Clamp to vocab ranges to prevent OOB on embedding lookup
        object.__setattr__(self, "op_id", min(max(self.op_id, 0), NUM_ACTION_TYPES - 1))
        object.__setattr__(self, "arg0", min(max(self.arg0, 0), 31))
        object.__setattr__(self, "arg1", min(max(self.arg1, 0), 31))
        object.__setattr__(self, "arg2", min(max(self.arg2, 0), 31))
        object.__setattr__(self, "modifier", min(max(self.modifier, 0), 15))

    def components(self) -> Tuple[int, int, int, int, int]:
        return (self.op_id, self.arg0, self.arg1, self.arg2, self.modifier)

    def to_flat_index(self, op_slots: int = 64, arg_slots: int = 32, mod_slots: int = 16) -> int:
        """Convert to a single integer for flat softmax prediction."""
        return (
            self.op_id * (arg_slots ** 3) * mod_slots
            + self.arg0 * (arg_slots ** 2) * mod_slots
            + self.arg1 * arg_slots * mod_slots
            + self.arg2 * mod_slots
            + self.modifier
        )

    @staticmethod
    def from_flat_index(idx: int, arg_slots: int = 32, mod_slots: int = 16) -> "ActionToken":
        """Inverse of to_flat_index."""
        modifier = idx % mod_slots
        idx //= mod_slots
        arg2 = idx % arg_slots
        idx //= arg_slots
        arg1 = idx % arg_slots
        idx //= arg_slots
        arg0 = idx % arg_slots
        op_id = idx // arg_slots
        return ActionToken(op_id=op_id, arg0=arg0, arg1=arg1, arg2=arg2, modifier=modifier)

    def is_halt(self) -> bool:
        return self.op_id == ActionType.HALT

    def __repr__(self) -> str:
        try:
            op_name = ActionType(self.op_id).name
        except ValueError:
            op_name = f"UNK({self.op_id})"
        return f"ActionToken({op_name}, a0={self.arg0}, a1={self.arg1}, a2={self.arg2}, mod={self.modifier})"


# Convenience constructors
def make_add(src_a: int, src_b: int, dst: int, carry_mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.ADD), arg0=src_a, arg1=src_b, arg2=dst, modifier=carry_mod)

def make_load(value_reg: int, target_reg: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.LOAD), arg0=value_reg, arg1=target_reg, modifier=mod)

def make_read(src_reg: int, dst_reg: int, ptr: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.READ), arg0=src_reg, arg1=dst_reg, arg2=ptr)

def make_write(src_reg: int, dst_reg: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.WRITE), arg0=src_reg, arg1=dst_reg, modifier=mod)

def make_store(src_reg: int, dst_reg: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.STORE), arg0=src_reg, arg1=dst_reg, modifier=mod)

def make_output(reg: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.OUTPUT), arg0=reg, modifier=mod)

def make_halt() -> ActionToken:
    return ActionToken(op_id=int(ActionType.HALT))

def make_pointer_move(ptr_idx: int, direction: int, amount: int = 1) -> ActionToken:
    return ActionToken(op_id=int(ActionType.POINTER_MOVE), arg0=ptr_idx, arg1=direction, arg2=amount)

def make_digit_extract(src_reg: int, dst_reg: int, digit_pos: int) -> ActionToken:
    return ActionToken(op_id=int(ActionType.DIGIT_EXTRACT), arg0=src_reg, arg1=dst_reg, arg2=digit_pos)

def make_digit_pack(src_reg: int, dst_reg: int, digit_pos: int) -> ActionToken:
    return ActionToken(op_id=int(ActionType.DIGIT_PACK), arg0=src_reg, arg1=dst_reg, arg2=digit_pos)

def make_subtract(src_a: int, src_b: int, dst: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.SUBTRACT), arg0=src_a, arg1=src_b, arg2=dst, modifier=mod)

def make_multiply(src_a: int, src_b: int, dst: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.MULTIPLY), arg0=src_a, arg1=src_b, arg2=dst, modifier=mod)

def make_compare(src_a: int, src_b: int, dst: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.COMPARE), arg0=src_a, arg1=src_b, arg2=dst, modifier=mod)

def make_loop(counter_reg: int, limit_reg: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.LOOP), arg0=counter_reg, arg1=limit_reg, modifier=mod)

def make_if(cond_reg: int, mod: int = 0) -> ActionToken:
    return ActionToken(op_id=int(ActionType.IF), arg0=cond_reg, modifier=mod)

def make_break() -> ActionToken:
    return ActionToken(op_id=int(ActionType.BREAK))

def make_nop() -> ActionToken:
    return ActionToken(op_id=int(ActionType.NOP))


# ---------------------------------------------------------------------------
# Factorized Action Embedding
# ---------------------------------------------------------------------------

class FactorizedActionEmbedding(nn.Module):
    """
    Factorized embedding: e = E_op(op) + E_arg0(a0) + E_arg1(a1) + E_arg2(a2) + E_mod(m).

    This is composable, deterministic, and has zero hash collisions.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        op_vocab: int = NUM_ACTION_TYPES,
        arg_vocab: int = 32,
        mod_vocab: int = 16,
        include_macro_vocab: bool = False,
        macro_vocab: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.op_embed = nn.Embedding(op_vocab, embed_dim)
        self.arg0_embed = nn.Embedding(arg_vocab, embed_dim)
        self.arg1_embed = nn.Embedding(arg_vocab, embed_dim)
        self.arg2_embed = nn.Embedding(arg_vocab, embed_dim)
        self.mod_embed = nn.Embedding(mod_vocab, embed_dim)

        self.include_macro_vocab = include_macro_vocab
        if include_macro_vocab:
            self.macro_embed = nn.Embedding(macro_vocab, embed_dim)

    def forward(
        self,
        op_ids: torch.LongTensor,
        arg0s: torch.LongTensor,
        arg1s: torch.LongTensor,
        arg2s: torch.LongTensor,
        mods: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            All inputs are (batch,) or (batch, seq) integer tensors.

        Returns:
            (batch, embed_dim) or (batch, seq, embed_dim) embeddings.
        """
        e = (
            self.op_embed(op_ids)
            + self.arg0_embed(arg0s)
            + self.arg1_embed(arg1s)
            + self.arg2_embed(arg2s)
            + self.mod_embed(mods)
        )
        return e

    def embed_token(self, token: ActionToken) -> torch.Tensor:
        """Embed a single ActionToken → (1, embed_dim)."""
        ops = torch.tensor([token.op_id])
        a0s = torch.tensor([token.arg0])
        a1s = torch.tensor([token.arg1])
        a2s = torch.tensor([token.arg2])
        ms = torch.tensor([token.modifier])
        return self.forward(ops, a0s, a1s, a2s, ms)

    def embed_tokens(self, tokens: List[ActionToken]) -> torch.Tensor:
        """Embed a list of ActionTokens → (len, embed_dim)."""
        ops = torch.tensor([t.op_id for t in tokens])
        a0s = torch.tensor([t.arg0 for t in tokens])
        a1s = torch.tensor([t.arg1 for t in tokens])
        a2s = torch.tensor([t.arg2 for t in tokens])
        ms = torch.tensor([t.modifier for t in tokens])
        return self.forward(ops, a0s, a1s, a2s, ms)


# ---------------------------------------------------------------------------
# Macro Action
# ---------------------------------------------------------------------------

@dataclass
class MacroAction:
    """
    A composed action built from a sequence of primitive ActionTokens.

    The embedding is a learned nn.Parameter, initialised as the mean of
    sub-action embeddings.  This lets the macro develop its own compressed
    representation that may diverge from the sum of its parts.
    """
    name: str
    sub_actions: List[ActionToken] = field(default_factory=list)
    embedding: Optional[nn.Parameter] = None
    usage_count: int = 0
    composition_score: float = 0.0
    macro_id: int = 0  # index into macro embedding table

    def initialize_embedding(self, embed_module: FactorizedActionEmbedding) -> None:
        """Set embedding as mean of sub-action embeddings."""
        if not self.sub_actions:
            self.embedding = nn.Parameter(torch.zeros(embed_module.embed_dim))
            return
        sub_embs = embed_module.embed_tokens(self.sub_actions)  # (len, dim)
        mean_emb = sub_embs.mean(dim=0)
        self.embedding = nn.Parameter(mean_emb.detach().clone())

    def expand(self) -> List[ActionToken]:
        """Return the flat primitive sequence."""
        return list(self.sub_actions)

    def trace_length(self) -> int:
        return len(self.sub_actions)


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class ActionSpace:
    """
    Manages the vocabulary of actions — both primitive and macro.

    Flat vocab size for the action predictor output head:
        flat_vocab = op_vocab * arg_vocab^3 * mod_vocab
    """

    def __init__(
        self,
        num_registers: int = 16,
        num_pointers: int = 4,
        op_vocab: int = NUM_ACTION_TYPES,
        arg_vocab: int = 8,  # small default to keep flat vocab manageable
        mod_vocab: int = 4,
    ):
        self.num_registers = num_registers
        self.num_pointers = num_pointers
        self.op_vocab = op_vocab
        self.arg_vocab = arg_vocab
        self.mod_vocab = mod_vocab

        self.action_types: List[ActionType] = list(ActionType)

        # Flat vocab size for single-step prediction
        self.flat_vocab_size = op_vocab * (arg_vocab ** 3) * mod_vocab

        # Macro library
        self.macro_library: Dict[str, MacroAction] = {}
        self._next_macro_id: int = 0

    def register_macro(self, macro: MacroAction) -> None:
        macro.macro_id = self._next_macro_id
        self._next_macro_id += 1
        self.macro_library[macro.name] = macro

    def remove_macro(self, name: str) -> None:
        self.macro_library.pop(name, None)

    def total_macro_count(self) -> int:
        return len(self.macro_library)

    def decode_flat_token(self, idx: int) -> ActionToken:
        return ActionToken.from_flat_index(idx, arg_slots=self.arg_vocab, mod_slots=self.mod_vocab)

    def encode_token_flat(self, token: ActionToken) -> int:
        return token.to_flat_index(op_slots=self.op_vocab, arg_slots=self.arg_vocab, mod_slots=self.mod_vocab)