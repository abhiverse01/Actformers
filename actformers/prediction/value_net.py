"""
Value Network — critic / baseline for RL policy gradient.

Maps the current MemoryState to a scalar value estimate used as a
baseline for variance reduction in REINFORCE.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from actformers.core.working_memory import MemoryState

__all__ = ["ValueNet"]


class ValueNet(nn.Module):
    """
    Critic network that estimates the expected return from the current state.

    Input: flattened working memory state (registers + pointers).
    Output: scalar value estimate V(s).

    Architecture: MLP with residual connections.
    """

    def __init__(
        self,
        num_registers: int = 16,
        register_dim: int = 64,
        num_pointers: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = num_registers * register_dim + num_pointers

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: MemoryState) -> torch.Tensor:
        """
        Args:
            state: Current MemoryState.

        Returns:
            (batch, 1) scalar value estimate.
        """
        batch_size = state.registers.shape[0]
        flat_regs = state.registers.reshape(batch_size, -1)
        features = torch.cat([flat_regs, state.pointers], dim=-1)
        return self.net(features)