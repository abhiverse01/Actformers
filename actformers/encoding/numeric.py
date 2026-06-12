"""
Numeric Input Encoder — decomposes integers into digits with positional
embedding, then uses a small transformer encoder to produce per-register
representations.

This gives the model correct inductive bias for digit-by-digit processing.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

__all__ = ["NumericInputEncoder"]


class NumericInputEncoder(nn.Module):
    """
    Encodes integer inputs into register-file representations.

    For each input number:
      1. Decompose into base-10 digits.
      2. Embed each digit (0–9) with a learned embedding.
      3. Add positional embeddings (position within the number).
      4. Pass through a small transformer encoder.
      5. Map each digit representation to the register dimension.

    Each number occupies a contiguous register range. Pointers to each
    number's range are stored in the pointer file.

    Args:
        max_digits: Maximum number of digits per number.
        register_dim: Dimension of each register (working memory).
        digit_embed_dim: Internal embedding dimension for digits.
        num_heads: Number of attention heads in the transformer.
        num_layers: Number of transformer encoder layers.
        max_numbers: Maximum number of input numbers (batch dimension).
    """

    def __init__(
        self,
        max_digits: int = 20,
        register_dim: int = 64,
        digit_embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_numbers: int = 4,
    ):
        super().__init__()
        self.max_digits = max_digits
        self.register_dim = register_dim
        self.digit_embed_dim = digit_embed_dim
        self.max_numbers = max_numbers

        # Digit embedding: 10 possible digits (0-9) + special sentinel for padding
        self.digit_embed = nn.Embedding(11, digit_embed_dim)
        # Positional embedding within a number
        self.pos_embed = nn.Embedding(max_digits, digit_embed_dim)

        # Small transformer encoder for digit sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=digit_embed_dim,
            nhead=num_heads,
            dim_feedforward=digit_embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project from digit_embed_dim to register_dim
        self.output_proj = nn.Linear(digit_embed_dim, register_dim)

    def _decompose_digits(self, number: int) -> List[int]:
        """Decompose a non-negative integer into base-10 digits (LSB-first)."""
        if number < 0:
            number = abs(number)
        if number == 0:
            return [0]
        digits = []
        while number > 0:
            digits.append(number % 10)
            number //= 10
        return digits  # LSB first

    def forward(
        self,
        numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of integer tensors into register representations.

        Args:
            numbers: (batch, num_numbers) float tensor of integers.

        Returns:
            (batch, num_numbers * max_digits, register_dim) tensor.
        """
        batch_size, num_numbers = numbers.shape
        all_digit_embeds = []

        for num_idx in range(num_numbers):
            num_batch = numbers[:, num_idx].long().tolist()
            batch_digit_indices = []

            for n in num_batch:
                digits = self._decompose_digits(int(n))
                # Pad to max_digits, LSB-first
                padded = digits + [10] * (self.max_digits - len(digits))  # 10 = PAD
                batch_digit_indices.append(padded[:self.max_digits])

            digit_tensor = torch.tensor(
                batch_digit_indices, dtype=torch.long, device=numbers.device
            )  # (batch, max_digits)

            # Embed digits
            d_embed = self.digit_embed(digit_tensor)  # (batch, max_digits, dim)

            # Add positional embeddings
            positions = torch.arange(self.max_digits, device=numbers.device).unsqueeze(0)
            d_embed = d_embed + self.pos_embed(positions)

            # Transformer encoding
            d_encoded = self.transformer(d_embed)  # (batch, max_digits, dim)

            # Project to register dimension
            d_out = self.output_proj(d_encoded)  # (batch, max_digits, register_dim)
            all_digit_embeds.append(d_out)

        # Concatenate all numbers' digit representations
        result = torch.cat(all_digit_embeds, dim=1)  # (batch, num_numbers * max_digits, reg_dim)
        return result

    def get_register_ranges(self, num_numbers: int) -> List[tuple]:
        """
        Return the register index ranges for each input number.

        Each number occupies max_digits consecutive registers starting at
        num_idx * max_digits.
        """
        ranges = []
        for i in range(num_numbers):
            start = i * self.max_digits
            end = start + self.max_digits
            ranges.append((start, end))
        return ranges


class SimpleScalarEncoder(nn.Module):
    """
    Fallback encoder that maps each scalar to a single register.
    Used for simple tasks where digit decomposition is overkill.
    """

    def __init__(self, register_dim: int = 64):
        super().__init__()
        self.register_dim = register_dim
        self.encoder = nn.Sequential(
            nn.Linear(1, register_dim // 2),
            nn.ReLU(),
            nn.Linear(register_dim // 2, register_dim),
        )

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalars: (batch, num_inputs) float tensor.
        Returns:
            (batch, num_inputs, register_dim) tensor.
        """
        # Reshape to merge batch and num_inputs for per-element encoding
        orig_shape = scalars.shape
        flat = scalars.reshape(-1, 1)
        encoded = self.encoder(flat)
        return encoded.reshape(*orig_shape, -1)