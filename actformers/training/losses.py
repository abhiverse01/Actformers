"""
Loss functions for Actformer training.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ActionCrossEntropyLoss",
    "SupervisedOutputLoss",
    "RLPolicyLoss",
]


class ActionCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss on predicted action logits vs ground-truth flat action indices.

    This is the primary supervised loss for Phase 1–2 training.
    """

    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -100):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, flat_vocab) or (batch, seq, flat_vocab)
            targets: (batch,) or (batch, seq) flat action indices.
        """
        return F.cross_entropy(
            logits, targets,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )


class SupervisedOutputLoss(nn.Module):
    """
    MSE loss on the numeric output vs target result.

    Used as an auxiliary loss alongside action prediction loss.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_output: torch.Tensor,
        target_output: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(predicted_output, target_output)


class RLPolicyLoss(nn.Module):
    """
    REINFORCE policy gradient loss with value baseline.

    loss = -E[advantage * log_prob]
    advantage = reward - V(s)

    Also returns value loss for joint training.
    """

    def __init__(self, entropy_coeff: float = 0.01):
        super().__init__()
        self.entropy_coeff = entropy_coeff

    def forward(
        self,
        log_probs: List[torch.Tensor],
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple:
        """
        Args:
            log_probs: List of (batch,) log-prob tensors, one per step.
            rewards: (batch,) scalar rewards.
            values: (batch,) value baseline estimates.

        Returns:
            policy_loss: scalar REINFORCE loss.
            value_loss: scalar MSE value loss.
            entropy: scalar entropy bonus.
        """
        # Stack log_probs: (num_steps, batch)
        log_probs_stack = torch.stack(log_probs, dim=0)
        advantages = (rewards - values).detach()

        # Weight each step's log_prob by the advantage
        policy_loss = -(advantages.unsqueeze(0) * log_probs_stack).mean()

        # Value loss
        value_loss = F.mse_loss(values, rewards)

        # Entropy bonus (estimated from log_probs)
        entropy = -log_probs_stack.mean()

        total_loss = policy_loss + self.entropy_coeff * entropy
        return total_loss, value_loss, entropy