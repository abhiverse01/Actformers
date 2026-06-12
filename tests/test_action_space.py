"""Tests for action space: ActionToken encoding, factorized embedding, ActionSpace."""

import pytest
import torch

from actformers.core.action_space import (
    ActionType,
    ActionToken,
    ActionSpace,
    FactorizedActionEmbedding,
    make_add,
    make_load,
    make_halt,
    make_output,
)


class TestActionToken:
    def test_basic_creation(self):
        t = ActionToken(op_id=0, arg0=1, arg1=2, arg2=3, modifier=0)
        assert t.op_id == 0
        assert t.arg0 == 1

    def test_clamping(self):
        t = ActionToken(op_id=999, arg0=999, modifier=999)
        assert t.op_id < len(ActionType)
        assert t.arg0 <= 31
        assert t.modifier <= 15

    def test_flat_index_roundtrip(self):
        t = ActionToken(op_id=2, arg0=1, arg1=2, arg2=3, modifier=5)
        flat = t.to_flat_index()
        recovered = ActionToken.from_flat_index(flat)
        assert recovered.op_id == t.op_id
        assert recovered.arg0 == t.arg0
        assert recovered.arg1 == t.arg1
        assert recovered.arg2 == t.arg2
        assert recovered.modifier == t.modifier

    def test_halt_detection(self):
        assert make_halt().is_halt()
        assert not make_add(0, 1, 2).is_halt()

    def test_repr(self):
        t = make_add(0, 1, 2)
        assert "ADD" in repr(t)

    def test_frozen(self):
        t = make_add(0, 1, 2)
        with pytest.raises(AttributeError):
            t.op_id = 5


class TestActionSpace:
    def test_flat_vocab_size(self):
        aspace = ActionSpace(num_registers=16)
        assert aspace.flat_vocab_size > 0

    def test_decode_encode_roundtrip(self):
        aspace = ActionSpace(num_registers=16, arg_vocab=32, mod_vocab=16)
        token = make_add(3, 7, 5, carry_mod=1)
        flat = aspace.encode_token_flat(token)
        recovered = aspace.decode_flat_token(flat)
        assert recovered.op_id == token.op_id
        assert recovered.arg0 == token.arg0
        assert recovered.arg1 == token.arg1
        assert recovered.arg2 == token.arg2
        assert recovered.modifier == token.modifier


class TestFactorizedEmbedding:
    def test_embed_shape(self):
        emb = FactorizedActionEmbedding(embed_dim=64)
        ops = torch.tensor([0, 1, 2])
        a0s = torch.tensor([0, 1, 2])
        a1s = torch.tensor([0, 0, 0])
        a2s = torch.tensor([0, 0, 0])
        mods = torch.tensor([0, 0, 0])
        result = emb(ops, a0s, a1s, a2s, mods)
        assert result.shape == (3, 64)

    def test_composability(self):
        """Different arg combos with same op should give different embeddings."""
        emb = FactorizedActionEmbedding(embed_dim=64)
        t1 = emb.embed_token(make_add(0, 1, 2))
        t2 = emb.embed_token(make_add(3, 4, 5))
        diff = (t1 - t2).abs().sum().item()
        assert diff > 0, "Same op with different args should differ"