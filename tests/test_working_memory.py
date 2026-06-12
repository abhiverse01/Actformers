"""Tests for WorkingMemory — state initialization, register ops, cloning."""

import pytest
import torch

from actformers.core.working_memory import MemoryState, WorkingMemory


@pytest.fixture
def wm():
    return WorkingMemory(num_registers=8, register_dim=32, scratchpad_size=16, scratchpad_dim=32)


class TestMemoryState:
    def test_make_halt_flag(self):
        flag = MemoryState.make_halt_flag(4, torch.device("cpu"))
        assert flag.shape == (4,)
        assert not flag.any()

    def test_init_state_shape(self, wm):
        state = wm.init_state(batch_size=2, device=torch.device("cpu"))
        assert state.registers.shape == (2, 8, 32)
        assert state.scratchpad.shape == (2, 16, 32)
        assert state.pointers.shape == (2, 4)
        assert state.step == 0


class TestRegisterOps:
    def test_update_and_read(self, wm):
        state = wm.init_state(1, torch.device("cpu"))
        value = torch.zeros(1, 32)
        value[0, 0] = 42.0
        new_state = wm.update_register(state, 0, value)
        # Original state unchanged
        assert state.registers[0, 0, 0].item() != 42.0
        # New state has the value
        read = wm.read_register(new_state, 0)
        assert read[0, 0].item() == 42.0

    def test_no_mutation(self, wm):
        state = wm.init_state(1, torch.device("cpu"))
        original_regs = state.registers.clone()
        value = torch.randn(1, 32)
        _ = wm.update_register(state, 0, value)
        # state should be unchanged (immutable semantics)
        assert torch.equal(state.registers, original_regs), "Original state was mutated"


class TestCloning:
    def test_gradient_isolation(self, wm):
        state = wm.init_state(1, torch.device("cpu"))
        value = torch.randn(1, 32, requires_grad=True)
        new_state = wm.update_register(state, 0, value)
        # Gradients on new_state should not affect original
        loss = new_state.registers.sum()
        loss.backward()
        assert value.grad is not None