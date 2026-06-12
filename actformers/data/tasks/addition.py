"""Addition task definition — generates (a, b, a+b) with ground-truth action traces."""

from __future__ import annotations
from typing import List, Tuple
from actformers.core.action_space import ActionToken
from actformers.data.trace_generator import ActionTraceGenerator

__all__ = ["AdditionTask"]

class AdditionTask:
    """Generates addition problem instances with action traces."""

    def __init__(self, min_digits: int = 1, max_digits: int = 5):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.trace_gen = ActionTraceGenerator()

    def sample(self) -> Tuple[int, int, int, List[ActionToken]]:
        a, b = self._sample_numbers()
        c = a + b
        trace = self.trace_gen.generate_addition_trace(a, b)
        return a, b, c, trace

    def _sample_numbers(self) -> Tuple[int, int]:
        import random
        def rand_n():
            n_digits = random.randint(self.min_digits, self.max_digits)
            low = 10 ** (n_digits - 1)
            high = 10 ** n_digits - 1
            return random.randint(low, high)
        return rand_n(), rand_n()

    def generate_trace(self, a: int, b: int) -> List[ActionToken]:
        return self.trace_gen.generate_addition_trace(a, b)