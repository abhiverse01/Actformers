"""
Supervised Trainer — standard teacher-forcing training on ground-truth action traces.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from actformers.core.model import Actformer
from actformers.training.losses import ActionCrossEntropyLoss, SupervisedOutputLoss

__all__ = ["SupervisedTrainer"]


class SupervisedTrainer:
    """
    Trains Actformer with teacher forcing on ground-truth action traces.

    At each step:
      1. Encode input → initial state.
      2. For each action in the ground-trace:
         a. Predict action logits from (state, history).
         b. Compute CE loss against ground-truth action.
         c. Execute ground-truth action (teacher forcing).
      3. Decode output from final state.
      4. Compute MSE loss on output vs target.
    """

    def __init__(
        self,
        model: Actformer,
        action_loss: ActionCrossEntropyLoss,
        output_loss: Optional[SupervisedOutputLoss] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        output_loss_weight: float = 0.1,
        max_grad_norm: float = 1.0,
        execution_mode: str = "train",
    ):
        self.model = model
        self.action_loss = action_loss
        self.output_loss = output_loss or SupervisedOutputLoss()
        self.output_loss_weight = output_loss_weight
        self.max_grad_norm = max_grad_norm
        self.execution_mode = execution_mode

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.step_count = 0

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dict with 'input', 'output', 'trace', 'flat_trace', 'trace_length'.

        Returns:
            Dict of loss values for logging.
        """
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch['input'].unsqueeze(0)  # (1, input_len)
        targets = batch['output'].unsqueeze(0)  # (1, 1)
        flat_trace = batch['flat_trace'].unsqueeze(0)  # (1, max_trace_len)
        trace_length = batch['trace_length']

        # Forward pass with teacher forcing
        output, info = self.model(
            inputs,
            target_trace=flat_trace[0, :trace_length].tolist(),
            execution_mode=self.execution_mode,
        )

        # Action prediction loss
        total_action_loss = torch.tensor(0.0, device=inputs.device)
        n_action_steps = 0
        if 'step_losses' in info and info['step_losses']:
            for loss_val in info['step_losses']:
                if loss_val is not None:
                    total_action_loss = total_action_loss + loss_val
                    n_action_steps += 1

        if n_action_steps > 0:
            total_action_loss = total_action_loss / n_action_steps

        # Output loss
        out_loss = self.output_loss(output, targets)

        # Combined loss
        loss = total_action_loss + self.output_loss_weight * out_loss

        # Gradient clipping
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        self.optimizer.step()
        self.step_count += 1

        return {
            'total_loss': loss.item(),
            'action_loss': total_action_loss.item() if n_action_steps > 0 else 0.0,
            'output_loss': out_loss.item(),
            'grad_norm': grad_norm.item(),
            'actions_taken': info.get('actions_taken', 0),
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_every: int = 100,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {
            'total_loss': 0.0,
            'action_loss': 0.0,
            'output_loss': 0.0,
            'num_batches': 0,
        }

        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)

            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            epoch_metrics['num_batches'] += 1

            if batch_idx % log_every == 0:
                avg = {k: v / max(epoch_metrics['num_batches'], 1)
                       for k, v in epoch_metrics.items()
                       if k != 'num_batches'}
                print(
                    f"Step {self.step_count} | "
                    f"loss={avg.get('total_loss', 0):.4f} | "
                    f"action_loss={avg.get('action_loss', 0):.4f} | "
                    f"output_loss={avg.get('output_loss', 0):.4f}"
                )

        for k in epoch_metrics:
            if k != 'num_batches':
                epoch_metrics[k] /= max(epoch_metrics['num_batches'], 1)

        return epoch_metrics