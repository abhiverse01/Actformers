"""
Curriculum Trainer — phased training with automatic advancement.

Phases:
  1. PRIMITIVE_EXECUTION — Learn individual action semantics.
  2. SHORT_SEQUENCES — 1-digit arithmetic, < 10 steps.
  3. FULL_ALGORITHM — 2-3 digit, full traces supervised.
  4. GENERALIZATION — 4-10 digit, mix supervised + RL.
  5. MULTI_TASK — Multiple task types, transfer learning.
  6. COMPOSITION — Macro discovery enabled.

Phase advancement is triggered by hitting a success threshold on the
current phase's evaluation suite.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

import torch

__all__ = ["CurriculumPhase", "CurriculumTrainer"]


class CurriculumPhase(Enum):
    PRIMITIVE_EXECUTION = 1
    SHORT_SEQUENCES = 2
    FULL_ALGORITHM = 3
    GENERALIZATION = 4
    MULTI_TASK = 5
    COMPOSITION = 6


PHASE_CONFIGS = {
    CurriculumPhase.PRIMITIVE_EXECUTION: {
        "min_digits": 1,
        "max_digits": 1,
        "max_steps": 10,
        "success_threshold": 0.90,
        "description": "Learn individual action semantics",
    },
    CurriculumPhase.SHORT_SEQUENCES: {
        "min_digits": 1,
        "max_digits": 2,
        "max_steps": 20,
        "success_threshold": 0.90,
        "description": "1-digit arithmetic, short sequences",
    },
    CurriculumPhase.FULL_ALGORITHM: {
        "min_digits": 2,
        "max_digits": 4,
        "max_steps": 50,
        "success_threshold": 0.85,
        "description": "Full algorithm with carry/borrow",
    },
    CurriculumPhase.GENERALIZATION: {
        "min_digits": 3,
        "max_digits": 10,
        "max_steps": 100,
        "success_threshold": 0.80,
        "description": "Generalize to longer numbers",
    },
    CurriculumPhase.MULTI_TASK: {
        "min_digits": 2,
        "max_digits": 5,
        "max_steps": 100,
        "success_threshold": 0.75,
        "description": "Multi-task transfer learning",
    },
    CurriculumPhase.COMPOSITION: {
        "min_digits": 3,
        "max_digits": 10,
        "max_steps": 100,
        "success_threshold": 0.70,
        "description": "Macro action discovery enabled",
    },
}


class CurriculumTrainer:
    """
    Manages phased curriculum training.

    Tracks performance metrics per phase and advances when the success
    threshold is met.  Also supports phase regression if performance drops.
    """

    def __init__(
        self,
        model,
        start_phase: CurriculumPhase = CurriculumPhase.SHORT_SEQUENCES,
        eval_every: int = 100,
    ):
        self.model = model
        self.current_phase = start_phase
        self.eval_every = eval_every
        self.phase_history: List[Dict] = []
        self.total_steps = 0

    @property
    def config(self) -> Dict:
        return PHASE_CONFIGS[self.current_phase]

    def should_evaluate(self) -> bool:
        return self.total_steps % self.eval_every == 0 and self.total_steps > 0

    def check_advancement(self, metrics: Dict[str, float]) -> Optional[CurriculumPhase]:
        """
        Check if the current phase's success threshold has been met.

        Args:
            metrics: Dict with at least 'exact_match' accuracy.

        Returns:
            Next phase if advancement criteria met, else None.
        """
        accuracy = metrics.get('exact_match', 0.0)
        threshold = self.config['success_threshold']

        if accuracy >= threshold:
            next_phase_idx = self.current_phase.value + 1
            if next_phase_idx <= len(CurriculumPhase):
                old_phase = self.current_phase
                self.current_phase = CurriculumPhase(next_phase_idx)
                self.phase_history.append({
                    'from': old_phase.name,
                    'to': self.current_phase.name,
                    'step': self.total_steps,
                    'accuracy': accuracy,
                })
                return self.current_phase
        return None

    def get_phase_summary(self) -> Dict:
        return {
            'current_phase': self.current_phase.name,
            'phase_number': self.current_phase.value,
            'total_steps': self.total_steps,
            'config': self.config,
            'phase_history': self.phase_history,
        }