"""
Macro Library — registry of composed macro actions with scoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from actformers.core.action_space import (
    ActionToken,
    ActionSpace,
    MacroAction,
    FactorizedActionEmbedding,
)

__all__ = ["MacroLibrary"]


class MacroLibrary(nn.Module):
    """
    Manages the library of macro actions.

    Each macro is a named, scored composition of primitive actions.
    The library supports CRUD operations, scoring updates, and pruning.

    The macro embeddings are registered as parameters so they can be
    trained alongside the rest of the model.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        action_embed: FactorizedActionEmbedding,
        max_macros: int = 128,
    ):
        super().__init__()
        self.action_space = action_space
        self.action_embed = action_embed
        self.max_macros = max_macros

        self.macros: Dict[str, MacroAction] = {}

    def register(
        self,
        name: str,
        sub_actions: List[ActionToken],
    ) -> MacroAction:
        """
        Register a new macro action.

        Args:
            name: Unique name for the macro.
            sub_actions: List of primitive ActionTokens.

        Returns:
            The registered MacroAction.
        """
        if len(self.macros) >= self.max_macros:
            # Prune lowest-scoring macro
            self._prune_worst()

        macro = MacroAction(
            name=name,
            sub_actions=list(sub_actions),
            macro_id=len(self.macros),
        )
        macro.initialize_embedding(self.action_embed)
        self.macros[name] = macro
        self.action_space.register_macro(macro)
        return macro

    def remove(self, name: str) -> None:
        """Remove a macro by name."""
        if name in self.macros:
            del self.macros[name]

    def get(self, name: str) -> Optional[MacroAction]:
        """Get a macro by name."""
        return self.macros.get(name)

    def update_score(self, name: str, success: bool) -> None:
        """Update composition score for a macro based on task outcome."""
        macro = self.macros.get(name)
        if macro is None:
            return
        macro.usage_count += 1
        if success:
            # Exponential moving average
            alpha = 1.0 / macro.usage_count
            macro.composition_score = (
                (1 - alpha) * macro.composition_score + alpha * 1.0
            )

    def _prune_worst(self) -> None:
        """Remove the lowest-scoring, least-used macro."""
        if not self.macros:
            return
        worst = min(
            self.macros.values(),
            key=lambda m: (m.composition_score, m.usage_count),
        )
        self.remove(worst.name)

    def get_all_macros(self) -> List[MacroAction]:
        """Return all macros sorted by composition score."""
        return sorted(
            self.macros.values(),
            key=lambda m: m.composition_score,
            reverse=True,
        )

    def summary(self) -> Dict[str, object]:
        return {
            'num_macros': len(self.macros),
            'macros': {
                name: {
                    'usage': m.usage_count,
                    'score': f"{m.composition_score:.3f}",
                    'length': m.trace_length(),
                }
                for name, m in self.macros.items()
            },
        }