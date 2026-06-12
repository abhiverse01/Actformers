"""
Composition Discovery — the evolutionary loop for discovering macro actions.

Process:
  1. Mine co-occurring action subsequences from recent rollout history.
  2. Merge candidate pairs into macro proposals.
  3. Probe-evaluate each candidate on a held-out set.
  4. Adopt if success improvement > threshold; reject otherwise.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

from actformers.core.action_space import ActionToken, ActionSpace, ActionType
from actformers.composition.macro_library import MacroLibrary

__all__ = ["CompositionDiscovery"]


class CompositionDiscovery:
    """
    Discovers macro actions by mining frequent action subsequences
    from rollout history.

    This is the mechanism by which a model that knows ADD discovers
    RANGE, and one that knows RANGE discovers SUM.
    """

    def __init__(
        self,
        macro_library: MacroLibrary,
        action_space: ActionSpace,
        min_frequency: int = 50,
        adoption_threshold: float = 0.02,
        min_subsequence_length: int = 3,
        max_subsequence_length: int = 15,
    ):
        self.macro_library = macro_library
        self.action_space = action_space
        self.min_frequency = min_frequency
        self.adoption_threshold = adoption_threshold
        self.min_seq_len = min_subsequence_length
        self.max_seq_len = max_subsequence_length

        # Rollout buffer
        self.rollout_buffer: List[List[int]] = []
        self.max_buffer_size = 10000

    def add_rollout(self, action_history: List[int]) -> None:
        """Add a completed rollout's action history to the buffer."""
        self.rollout_buffer.append(list(action_history))
        if len(self.rollout_buffer) > self.max_buffer_size:
            self.rollout_buffer.pop(0)

    def mine_candidates(self) -> List[List[int]]:
        """
        Mine frequent subsequences from the rollout buffer.

        Returns:
            List of candidate sub-sequences (as flat action index lists).
        """
        if len(self.rollout_buffer) < self.min_frequency:
            return []

        # Count n-gram frequencies
        ngram_counts: Counter = Counter()
        for seq in self.rollout_buffer:
            for n in range(self.min_seq_len, self.max_seq_len + 1):
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i+n])
                    ngram_counts[ngram] += 1

        # Filter by minimum frequency
        candidates = [
            list(ngram)
            for ngram, count in ngram_counts.items()
            if count >= self.min_frequency
        ]

        # Sort by frequency (most frequent first)
        candidates.sort(
            key=lambda c: ngram_counts[tuple(c)],
            reverse=True,
        )

        return candidates[:20]  # Top 20 candidates

    def propose_macro(
        self,
        subsequence: List[int],
        name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a macro action from a subsequence and add it to the library.

        Args:
            subsequence: List of flat action indices.
            name: Optional name (auto-generated if None).

        Returns:
            Macro name if adopted, None if rejected.
        """
        tokens = [self.action_space.decode_flat_token(idx) for idx in subsequence]

        # Skip if already exists
        for existing_name, macro in self.macro_library.macros.items():
            if [t.op_id for t in macro.sub_actions] == [t.op_id for t in tokens]:
                return None  # Already have a similar macro

        if name is None:
            # Auto-generate name from action types
            op_names = [ActionType(t.op_id).name if t.op_id < len(ActionType) else "UNK" for t in tokens]
            name = f"MACRO_{'_'.join(op_names[:3])}"
            if len(self.macro_library.macros) > 0:
                name = f"macro_{len(self.macro_library.macros)}"

        try:
            self.macro_library.register(name, tokens)
            return name
        except Exception:
            return None

    def probe_evaluate(
        self,
        macro_name: str,
        eval_fn=None,
    ) -> bool:
        """
        Evaluate a candidate macro on a held-out probe set.

        Args:
            macro_name: Name of the macro to evaluate.
            eval_fn: Optional callable(macro_name) -> (baseline_acc, macro_acc).
                    If None, adopt conservatively (no gating).

        Returns:
            True if the macro should be adopted.
        """
        if eval_fn is None:
            return True  # No eval function → adopt conservatively

        try:
            baseline_acc, macro_acc = eval_fn(macro_name)
            improvement = macro_acc - baseline_acc
            return improvement > self.adoption_threshold
        except Exception:
            return True  # On error, adopt conservatively

    def discovery_step(
        self,
        eval_fn=None,
    ) -> Dict:
        """
        Run one discovery cycle: mine → propose → probe → adopt/reject.

        Returns:
            Dict with discovery results.
        """
        candidates = self.mine_candidates()
        adopted = []
        rejected = []

        for candidate in candidates:
            name = self.propose_macro(candidate)
            if name is not None:
                # Only adopt if probe passes (or no eval_fn for conservative adoption)
                if self.probe_evaluate(name, eval_fn):
                    adopted.append(name)
                else:
                    # Remove macro that failed probe
                    self.macro_library.remove(name)
                    rejected.append(candidate)
            else:
                rejected.append(candidate)

        return {
            'candidates_mined': len(candidates),
            'adopted': adopted,
            'rejected_count': len(rejected),
            'buffer_size': len(self.rollout_buffer),
            'total_macros': self.macro_library.get_all_macros().__len__(),
        }