"""
Actformer — Top-level model combining all components.

Forward pass: encode_input → action loop (predict → execute → update history) → decode_output.

Supports two modes:
  1. Teacher forcing: ground-trace provided, actions executed from trace.
  2. Autoregressive: model samples actions from its own policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_space import (
    ActionSpace,
    ActionToken,
    FactorizedActionEmbedding,
    MacroAction,
    ActionType,
)
from .working_memory import MemoryState, WorkingMemory
from .primitives import DifferentiablePrimitives
from .execution_engine import ActionExecutionEngine
from actformers.prediction.action_predictor import ActionPredictor

__all__ = ["Actformer"]


class Actformer(nn.Module):
    """
    The Actformer model.

    Architecture overview:
      1. **NumericInputEncoder** (external): maps input numbers → register state.
      2. **WorkingMemory**: register file + scratchpad + pointers.
      3. **ActionPredictor**: dual-attention next-action predictor.
      4. **ActionExecutionEngine**: dispatches actions against memory.
      5. **OutputDecoder** (external): maps final register state → scalar output.

    The action loop runs for up to max_steps or until HALT is emitted.
    In teacher-forcing mode, the ground-truth action is used at each step.
    In autoregressive mode, the model samples from its policy.

    Args:
        num_registers: Number of working memory registers.
        register_dim: Dimension of each register.
        scratchpad_size: Size of scratchpad memory.
        scratchpad_dim: Dimension of scratchpad cells.
        hidden_dim: Hidden dimension for action predictor.
        num_heads: Number of attention heads in transformer decoder.
        num_layers: Number of transformer decoder layers.
        max_steps: Maximum computation steps per forward pass.
        input_encoder: Optional pre-built input encoder module.
        output_decoder: Optional pre-built output decoder module.
    """

    def __init__(
        self,
        num_registers: int = 16,
        register_dim: int = 64,
        scratchpad_size: int = 256,
        scratchpad_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_steps: int = 100,
        input_encoder: Optional[nn.Module] = None,
        output_decoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.num_registers = num_registers
        self.register_dim = register_dim
        self.scratchpad_size = scratchpad_size
        self.scratchpad_dim = scratchpad_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        # Action space
        self.action_space = ActionSpace(
            num_registers=num_registers,
        )

        # Working memory
        self.working_memory = WorkingMemory(
            num_registers=num_registers,
            register_dim=register_dim,
            scratchpad_size=scratchpad_size,
            scratchpad_dim=scratchpad_dim,
        )

        # Execution engine
        self.execution_engine = ActionExecutionEngine(
            action_space=self.action_space,
            working_memory=self.working_memory,
            primitive_dim=register_dim,
        )

        # Action predictor
        self.action_predictor = ActionPredictor(
            action_space=self.action_space,
            register_dim=register_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # Input encoder (default: simple scalar encoder)
        if input_encoder is not None:
            self.input_encoder = input_encoder
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(1, register_dim),
                nn.Tanh(),  # normalize inputs to [-1, 1]
            )

        # Output decoder (default: linear from register dim)
        if output_decoder is not None:
            self.output_decoder = output_decoder
        else:
            self.output_decoder = nn.Sequential(
                nn.Linear(register_dim, register_dim // 2),
                nn.ReLU(),
                nn.Linear(register_dim // 2, register_dim // 4),
                nn.ReLU(),
                nn.Linear(register_dim // 4, 1),
                nn.Tanh(),  # bounded output in [-1, 1]
            )
        # Output scale factor for mapping [-1,1] -> reasonable range
        self.output_scale = nn.Parameter(torch.tensor(100.0))

    def encode_input(self, inputs: torch.Tensor) -> MemoryState:
        """
        Encode input tensors to initial working memory state.

        Args:
            inputs: (batch, input_len) tensor of scalars.

        Returns:
            Initial MemoryState.
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Initialize clean memory state
        state = self.working_memory.init_state(batch_size, device)

        # Encode each input into a register
        for i in range(min(inputs.shape[1], self.num_registers)):
            scalar = inputs[:, i:i+1]  # (batch, 1)
            encoded = self.input_encoder(scalar)  # (batch, register_dim)
            state = self.working_memory.update_register(state, i, encoded)

        return state

    def decode_output(self, state: MemoryState, output_reg: int = 0) -> torch.Tensor:
        """
        Decode output from final working memory state.

        Args:
            state: Final MemoryState.
            output_reg: Which register to read for output.

        Returns:
            (batch, 1) predicted scalar.
        """
        reg_val = self.working_memory.read_register(state, output_reg)  # (batch, dim)
        return self.output_decoder(reg_val) * self.output_scale  # (batch, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        target_trace: Optional[List[int]] = None,
        execution_mode: str = "train",
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run the Actformer computation loop.

        Args:
            inputs: (batch, input_len) input tensor.
            target_trace: Optional ground-truth flat action indices for teacher forcing.
            execution_mode: 'train' (soft branches) or 'infer' (hard branches).
            temperature: Sampling temperature for autoregressive mode.

        Returns:
            output: (batch, 1) predicted output.
            info: Dict with action history, log_probs, losses, etc.
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Initialize state
        state = self.encode_input(inputs)
        action_history: List[int] = []
        log_probs: List[torch.Tensor] = []
        step_losses: List[Optional[torch.Tensor]] = []
        actions_taken = 0

        # Determine loop length
        if target_trace is not None:
            loop_length = min(len(target_trace), self.max_steps)
        else:
            loop_length = self.max_steps

        for step in range(loop_length):
            # Predict next action
            logits = self.action_predictor(state, action_history)  # (1, flat_vocab)

            if target_trace is not None and step < len(target_trace):
                # Teacher forcing: use ground-truth action
                target_idx = target_trace[step]
                target_tensor = torch.tensor([target_idx], device=device)
                log_prob = F.log_softmax(logits, dim=-1).gather(
                    1, target_tensor.unsqueeze(-1)
                ).squeeze(-1)

                # Compute CE loss for this step
                loss = F.cross_entropy(logits, target_tensor)
                step_losses.append(loss)

                action_token = target_tensor
            else:
                # Autoregressive: sample from policy
                action_token, log_prob = self.action_predictor.sample_action(
                    logits, temperature=temperature
                )

            action_idx = action_token[0].item()
            action_history.append(action_idx)
            log_probs.append(log_prob)
            actions_taken += 1

            # Decode and execute action
            action = self.action_space.decode_flat_token(action_idx)

            # Handle macro actions: expand to primitives if macro detected
            op = ActionType(action.op_id) if action.op_id < len(ActionType) else None
            if op is not None and op == ActionType.CALL_TOOL and action.modifier in self.action_space.macro_library:
                macro = self.action_space.macro_library[action.modifier]
                state = self.execution_engine.execute_macro(
                    macro.expand(), state, execution_mode
                )
            else:
                state = self.execution_engine.execute(action, state, execution_mode)

            # Check for halt
            if state.halt_flag is not None and state.halt_flag.any():
                break

            # Increment step counter in state
            from dataclasses import replace as dc_replace
            state = dc_replace(state, step=step + 1)

        # Decode output
        output = self.decode_output(state)

        info = {
            'action_history': action_history,
            'log_probs': log_probs,
            'step_losses': step_losses,
            'actions_taken': actions_taken,
            'final_state': state,
            'memory_summary': self.working_memory.summary(state),
        }

        return output, info

    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        log_probs: List[torch.Tensor],
        reward: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute combined supervised + optional RL loss.

        Args:
            output: (batch, 1) model output.
            target: (batch, 1) target value.
            log_probs: List of log-prob tensors.
            reward: Optional reward for RL fine-tuning.

        Returns:
            Combined scalar loss.
        """
        # Supervised output loss
        supervised_loss = F.mse_loss(output, target)

        if reward is not None and log_probs:
            # RL policy gradient loss
            rl_loss = -sum(lp.mean() for lp in log_probs) * reward
            return supervised_loss + 0.01 * rl_loss

        return supervised_loss