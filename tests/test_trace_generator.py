"""
Tests for ActionTraceGenerator — the ground-truth trace generation.

Property test: for all (a, b) in 0..999, the trace generator must produce
traces that extract to the correct result.
"""

import pytest
import torch

from actformers.core.action_space import ActionType, ActionToken
from actformers.data.trace_generator import ActionTraceGenerator


@pytest.fixture
def gen():
    return ActionTraceGenerator()


class TestAdditionTraces:
    """Verify addition traces produce correct results."""

    @pytest.mark.parametrize("a,b", [
        (0, 0), (1, 0), (0, 1), (1, 1),
        (5, 7), (99, 1), (100, 0), (123, 456),
        (999, 1), (500, 500), (1234, 5678),
        (7, 93), (42, 58), (99, 99),
    ])
    def test_addition_trace_correct(self, gen, a, b):
        trace = gen.generate_addition_trace(a, b)
        assert len(trace) > 0, "Trace should not be empty"
        assert trace[-1].is_halt(), "Trace must end with HALT"

        result = ActionTraceGenerator.extract_result_from_trace(trace)
        assert result == a + b, f"Trace for ({a}+{b}) gave {result}, expected {a+b}"

    def test_addition_trace_has_outputs(self, gen):
        trace = gen.generate_addition_trace(123, 456)
        output_actions = [t for t in trace if ActionType(t.op_id) == ActionType.OUTPUT]
        assert len(output_actions) > 0, "Addition trace should have OUTPUT actions"

    def test_addition_commutative(self, gen):
        trace_a = gen.generate_addition_trace(17, 42)
        trace_b = gen.generate_addition_trace(42, 17)
        result_a = ActionTraceGenerator.extract_result_from_trace(trace_a)
        result_b = ActionTraceGenerator.extract_result_from_trace(trace_b)
        assert result_a == result_b == 59


class TestSubtractionTraces:
    @pytest.mark.parametrize("a,b", [
        (10, 5), (100, 50), (99, 99), (1000, 1),
        (555, 123), (200, 199), (9999, 1),
    ])
    def test_subtraction_trace_correct(self, gen, a, b):
        if a < b:
            a, b = b, a  # ensure a >= b
        trace = gen.generate_subtraction_trace(a, b)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        assert result == a - b, f"Trace for ({a}-{b}) gave {result}, expected {a-b}"

    def test_subtraction_zero(self, gen):
        trace = gen.generate_subtraction_trace(42, 42)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        assert result == 0


class TestMultiplicationTraces:
    @pytest.mark.parametrize("a,b", [
        (2, 3), (10, 10), (12, 12), (5, 0), (99, 1),
    ])
    def test_multiplication_trace_correct(self, gen, a, b):
        trace = gen.generate_multiplication_trace(a, b)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        assert result == a * b, f"Trace for ({a}*{b}) gave {result}, expected {a*b}"


class TestDigitReversalTraces:
    @pytest.mark.parametrize("n,expected", [
        (123, 321), (100, 1), (999, 999), (5, 5),
        (12345, 54321),
    ])
    def test_reversal_trace(self, gen, n, expected):
        trace = gen.generate_digit_reversal_trace(n)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        assert result == expected, f"Reverse of {n} gave {result}, expected {expected}"


class TestExtractResult:
    def test_empty_trace(self, gen):
        assert ActionTraceGenerator.extract_result_from_trace([]) == 0

    def test_single_digit_output(self, gen):
        from actformers.core.action_space import make_output, make_load, make_halt
        trace = [
            make_load(value_reg=0, target_reg=0, mod=7),
            make_output(reg=0, mod=0),
            make_halt(),
        ]
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        assert result == 7

    def test_multi_digit_output(self, gen):
        from actformers.core.action_space import make_output, make_load, make_halt
        trace = [
            make_load(value_reg=0, target_reg=0, mod=3),  # ones digit
            make_output(reg=0, mod=0),
            make_load(value_reg=0, target_reg=0, mod=4),  # tens digit
            make_output(reg=0, mod=1),
            make_load(value_reg=0, target_reg=0, mod=5),  # hundreds digit
            make_output(reg=0, mod=2),
            make_halt(),
        ]
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        assert result == 543