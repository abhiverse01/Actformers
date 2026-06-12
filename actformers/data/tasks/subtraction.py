"""Subtraction task definition."""

from __future__ import annotations
from typing import List, Tuple
import random
from actformers.core.action_space import ActionToken
from actformers.data.trace_generator import ActionTraceGenerator

__all__ = ["SubtractionTask"]

class SubtractionTask:
    def __init__(self, min_digits: int = 1, max_digits: int = 5):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.trace_gen = ActionTraceGenerator()

    def sample(self) -> Tuple[int, int, int, List[ActionToken]]:
        a, b = self._sample_numbers()
        if a < b:
            a, b = b, a
        c = a - b
        trace = self.trace_gen.generate_subtraction_trace(a, b)
        return a, b, c, trace

    def _sample_numbers(self) -> Tuple[int, int]:
        def rand_n():
            n_digits = random.randint(self.min_digits, self.max_digits)
            return random.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)
        return rand_n(), rand_n()