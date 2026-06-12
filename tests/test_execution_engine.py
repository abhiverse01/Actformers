"""Tests for ActionExecutionEngine — all action types."""

import pytest
import torch

from actformers.core.action_space import (
    ActionSpace, ActionToken, ActionType,
    make_add, make_load, make_halt, make_output, make_nop,
    make_subtract, make_multiply, make_compare,
)
from actformers.core.working_memory import WorkingMemory
from actformers.core.execution_engine import ActionExecutionEngine


@pytest.fixture
def engine():
    wm = WorkingMemory(num_registers=8, register_dim=32, scratchpad_size=16, scratchpad_dim=32)
    aspace = ActionSpace(num_registers=8)
    return ActionExecutionEngine(action_space=aspace, working_memory=wm, primitive_dim=32)


@pytest.fixture
def state(engine):
    return engine.wm.init_state(batch_size=1, device=torch.device("cpu"))


class TestPrimitiveActions:
    def test_add(self, engine, state):
        val_a = torch.zeros(1, 32); val_a[0, 0] = 5.0
        val_b = torch.zeros(1, 32); val_b[0, 0] = 3.0
        state = engine.wm.update_register(state, 0, val_a)
        state = engine.wm.update_register(state, 1, val_b)
        action = make_add(0, 1, 2)
        new_state = engine.execute(action, state, "infer")
        result = engine.wm.read_register(new_state, 2)
        assert result[0, 0].item() == pytest.approx(8.0)

    def test_subtract(self, engine, state):
        val_a = torch.zeros(1, 32); val_a[0, 0] = 10.0
        val_b = torch.zeros(1, 32); val_b[0, 0] = 4.0
        state = engine.wm.update_register(state, 0, val_a)
        state = engine.wm.update_register(state, 1, val_b)
        action = make_subtract(0, 1, 2)
        new_state = engine.execute(action, state, "infer")
        result = engine.wm.read_register(new_state, 2)
        assert result[0, 0].item() == pytest.approx(6.0)

    def test_multiply(self, engine, state):
        val_a = torch.zeros(1, 32); val_a[0, 0] = 5.0
        val_b = torch.zeros(1, 32); val_b[0, 0] = 3.0
        state = engine.wm.update_register(state, 0, val_a)
        state = engine.wm.update_register(state, 1, val_b)
        action = make_multiply(0, 1, 2)
        new_state = engine.execute(action, state, "infer")
        result = engine.wm.read_register(new_state, 2)
        assert result[0, 0].item() == pytest.approx(15.0)

    def test_nop(self, engine, state):
        new_state = engine.execute(make_nop(), state, "infer")
        assert new_state.registers.shape == state.registers.shape

    def test_halt_sets_flag(self, engine, state):
        new_state = engine.execute(make_halt(), state, "infer")
        assert new_state.halt_flag.all()


class TestMemoryActions:
    def test_load(self, engine, state):
        val = torch.zeros(1, 32); val[0, 0] = 42.0
        state = engine.wm.update_register(state, 0, val)
        action = make_load(value_reg=0, target_reg=5)
        new_state = engine.execute(action, state, "infer")
        result = engine.wm.read_register(new_state, 5)
        assert result[0, 0].item() == pytest.approx(42.0)