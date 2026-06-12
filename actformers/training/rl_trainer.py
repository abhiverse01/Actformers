"""
RL Trainer — REINFORCE with baseline for Phase 4+ training.

Reward shaping: +1 for each correct output digit, +2 bonus for exact match.
This makes RL stable on longer sequences.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from actformers.core.model import Actformer
from actformers.prediction.value_net import ValueNet
from actformers.training.losses import RLPolicyLoss

__all__ = ["RLTrainer"]


def shape_reward(
    predicted_output: int,
    target_output: int,
) -> float:
    """
    Reward shaping: partial credit for correct digits + exact match bonus.

    +1 per correct digit, +2 bonus for exact match.
    """
    pred_str = str(abs(predicted_output))
    target_str = str(abs(target_output))
    max_len = max(len(pred_str), len(target_str))
    pred_str = pred_str.zfill(max_len)
    target_str = target_str.zfill(max_len)

    digit_reward = sum(1.0 for pc, tc in zip(pred_str, target_str) if pc == tc)
    exact_bonus = 2.0 if predicted_output == target_output else 0.0
    return digit_reward + exact_bonus


class RLTrainer:
    """
    REINFORCE with value baseline.

    Collect rollouts, compute advantages, update policy and value network.
    Never train from scratch — always fine-tune from a supervised checkpoint.
    """

    def __init__(
        self,
        model: Actformer,
        value_net: ValueNet,
        policy_lr: float = 1e-5,
        value_lr: float = 1e-4,
        gamma: float = 0.99,
    ):
        self.model = model
        self.value_net = value_net
        self.gamma = gamma

        self.policy_optimizer = torch.optim.Adam(
            model.parameters(), lr=policy_lr
        )
        self.value_optimizer = torch.optim.Adam(
            value_net.parameters(), lr=value_lr
        )

        self.rl_loss_fn = RLPolicyLoss(entropy_coeff=0.01)

    def collect_rollout(
        self,
        inputs: torch.Tensor,
        target: int,
        execution_mode: str = "train",
    ) -> Dict:
        """
        Collect a single rollout: run model autoregressively, compute reward.

        We collect actions without grad, then re-run the forward pass WITH grad
        to obtain differentiable log_probs for policy gradient update.

        Returns:
            Dict with actions taken, reward, and info for re-computation.
        """
        self.model.eval()
        with torch.no_grad():
            output, info = self.model(
                inputs, execution_mode=execution_mode
            )

        predicted = round(output.item() * 100)  # denormalize
        reward = shape_reward(predicted, target)

        # Re-run with grad to get differentiable log_probs
        self.model.train()
        output_grad, info_grad = self.model(
            inputs, execution_mode=execution_mode
        )

        return {
            'log_probs': info_grad['log_probs'],
            'reward': reward,
            'predicted': predicted,
            'target': target,
            'actions_taken': info['actions_taken'],
            'initial_state': info_grad.get('final_state', None),
        }

    def update(self, rollouts: List[Dict]) -> Dict[str, float]:
        """
        Update policy and value network from collected rollouts.

        Args:
            rollouts: List of rollout dicts from collect_rollout.

        Returns:
            Dict of loss metrics.
        """
        self.model.train()
        self.value_net.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for rollout in rollouts:
            reward = torch.tensor(
                [rollout['reward']], dtype=torch.float
            )
            log_probs = rollout['log_probs']

            if not log_probs:
                continue

            # Get value estimate from the final state as baseline
            # Detach state to create independent graph for value_net
            state = rollout.get('initial_state', None)
            if state is not None:
                from actformers.core.working_memory import MemoryState
                detached_state = MemoryState(
                    registers=state.registers.detach(),
                    scratchpad=state.scratchpad.detach(),
                    pointers=state.pointers.detach(),
                    step=state.step,
                    halt_flag=state.halt_flag.detach() if state.halt_flag is not None else None,
                )
                value = self.value_net(detached_state).reshape(-1)  # ensure 1D
            else:
                value = torch.zeros(reward.shape[0])
            # Ensure value and reward shapes match
            if value.shape[0] > reward.shape[0]:
                value = value[:reward.shape[0]]
            elif value.shape[0] < reward.shape[0]:
                reward = reward[:value.shape[0]]

            # Compute losses separately to avoid graph conflicts
            # Policy loss (uses log_probs from model)
            log_probs_stack = torch.stack(log_probs, dim=0)
            advantages = (reward - value.detach()).unsqueeze(0)  # detach baseline
            policy_loss = -(advantages * log_probs_stack).mean()
            entropy = -log_probs_stack.mean()
            policy_loss = policy_loss + self.rl_loss_fn.entropy_coeff * entropy

            # Value loss (uses value_net only)
            value_loss = F.mse_loss(value, reward)

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.policy_optimizer.step()

            # Update value net (separate backward)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'n_rollouts': len(rollouts),
        }