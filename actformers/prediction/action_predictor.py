"""
Action Predictor — dual attention mechanism for predicting the next action.

Combines:
  1. State attention: attends over current working memory registers.
  2. History attention: TransformerDecoder over previous action embeddings.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from actformers.core.action_space import (
    ActionToken,
    ActionSpace,
    FactorizedActionEmbedding,
)
from actformers.core.working_memory import MemoryState

__all__ = ["ActionPredictor"]


class ActionPredictor(nn.Module):
    """
    Predicts the next action given the current memory state and action history.

    Architecture:
      - State encoder: MLP over flattened register state → context vector.
      - History decoder: TransformerDecoderLayer over action embeddings,
        cross-attending to the state context.
      - Output head: Linear projection to flat vocab size (action distribution).

    Supports both teacher-forcing (ground-truth trace provided) and
    autoregressive (sample from policy) modes.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        register_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_history: int = 100,
    ):
        super().__init__()
        self.action_space = action_space
        self.register_dim = register_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_history = max_history

        # Factorized action embedding
        self.action_embed = FactorizedActionEmbedding(
            embed_dim=hidden_dim,
            op_vocab=action_space.op_vocab,
            arg_vocab=action_space.arg_vocab,
            mod_vocab=action_space.mod_vocab,
        )

        # Positional encoding for action history
        self.pos_encoding = nn.Parameter(
            self._sinusoidal_encoding(max_history, hidden_dim)
        )

        # State encoder: register state → context
        self.state_encoder = nn.Sequential(
            nn.Linear(register_dim * action_space.num_registers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: hidden → flat vocab logits
        self.output_head = nn.Linear(hidden_dim, action_space.flat_vocab_size)

    @staticmethod
    def _sinusoidal_encoding(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, dim)

    def encode_state(self, state: MemoryState) -> torch.Tensor:
        """
        Encode the current working memory state into a context vector.

        Args:
            state: MemoryState with registers (batch, num_regs, reg_dim).

        Returns:
            (batch, 1, hidden_dim) context vector.
        """
        batch_size = state.registers.shape[0]
        flat_regs = state.registers.reshape(batch_size, -1)  # (batch, num_regs * reg_dim)
        context = self.state_encoder(flat_regs)  # (batch, hidden_dim)
        return context.unsqueeze(1)  # (batch, 1, hidden_dim)

    def encode_history(
        self,
        action_history: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode the action history as embeddings.

        Args:
            action_history: List of flat action indices.
            device: Torch device.

        Returns:
            (batch, seq_len, hidden_dim) tensor.
        """
        if not action_history:
            return torch.zeros(1, 0, self.hidden_dim, device=device)

        tokens = [self.action_space.decode_flat_token(idx) for idx in action_history]
        emb = self.action_embed.embed_tokens(tokens)  # (seq, hidden_dim)
        seq_len = emb.shape[0]

        # Add positional encoding: (seq, dim) + (seq, dim) → (seq, dim)
        emb = emb + self.pos_encoding[0, :seq_len, :]

        return emb.unsqueeze(0)  # (1, seq, hidden_dim)

    def forward(
        self,
        state: MemoryState,
        action_history: List[int],
    ) -> torch.Tensor:
        """
        Predict logits for the next action.

        Args:
            state: Current MemoryState.
            action_history: List of previously taken flat action indices.

        Returns:
            (1, flat_vocab_size) logits for next action.
        """
        device = state.registers.device

        # Encode state as context
        context = self.encode_state(state)  # (batch, 1, hidden)

        # Encode action history
        history_emb = self.encode_history(action_history, device)  # (1, seq, hidden)

        if history_emb.shape[1] > 0:
            # Cross-attend: history attends to state context
            # TransformerDecoder expects (tgt, memory) both 3D: (batch, seq, dim)
            decoded = self.decoder(history_emb, context)  # (batch, seq, hidden)
            # Take last position's output
            next_action_repr = decoded[:, -1:, :]  # (batch, 1, hidden)
        else:
            # No history yet: use state context directly
            next_action_repr = context  # (batch, 1, hidden)

        # Project to vocab logits
        logits = self.output_head(next_action_repr.squeeze(1))  # (1, flat_vocab)
        return logits

    def sample_action(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the logits distribution.

        Args:
            logits: (batch, flat_vocab) unnormalized logits.
            temperature: Sampling temperature (< 1 = sharper, > 1 = softer).

        Returns:
            action_token: (batch,) sampled flat action index.
            log_prob: (batch,) log probability of the sampled action.
        """
        probs = F.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action_token = dist.sample()
        log_prob = dist.log_prob(action_token)
        return action_token, log_prob

    def teacher_forcing_step(
        self,
        state: MemoryState,
        action_history: List[int],
        target_action: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single teacher-forcing step: predict logits for the target action.

        Returns:
            logits: (batch, flat_vocab)
            log_prob: (batch,) log prob of target action.
        """
        logits = self.forward(state, action_history)
        log_probs = F.log_softmax(logits, dim=-1)
        target_tensor = torch.tensor([target_action], device=logits.device)
        log_prob = log_probs.gather(1, target_tensor.unsqueeze(-1)).squeeze(-1)
        return logits, log_prob