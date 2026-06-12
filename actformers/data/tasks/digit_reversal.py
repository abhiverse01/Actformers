"""Digit reversal task definition."""

from __future__ import annotations
from typing import List, Tuple
import random
from actformers.core.action_space import ActionToken
from actformers.data.trace_generator import ActionTraceGenerator

__all__ = ["DigitReversalTask"]

class DigitReversalTask:
    def __init__(self, min_digits: int = 1, max_digits: int = 5):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.trace_gen = ActionTraceGenerator()

    def sample(self) -> Tuple[int, int, List[ActionToken]]:
        n = self._sample_number()
        reversed_n = int(str(n)[::-1])
        trace = self.trace_gen.generate_digit_reversal_trace(n)
        return n, reversed_n, trace

    def _sample_number(self) -> int:
        n_digits = random.randint(self.min_digits, self.max_digits)
        return random.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)