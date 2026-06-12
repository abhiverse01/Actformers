"""
Output Decoder — reconstructs numeric output from working memory state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["OutputDecoder"]


class OutputDecoder(nn.Module):
    """
    Decodes the final working memory state into a numeric output.

    Reads from a specified output register and maps to a scalar.
    Supports digit-by-digit reconstruction for multi-digit outputs.
    """

    def __init__(self, register_dim: int = 64, num_output_digits: int = 20):
        super().__init__()
        self.register_dim = register_dim
        self.num_output_digits = num_output_digits

        # Simple scalar decoder: register → single output value
        self.scalar_decoder = nn.Sequential(
            nn.Linear(register_dim, register_dim // 2),
            nn.ReLU(),
            nn.Linear(register_dim // 2, 1),
        )

        # Digit-level decoder for multi-digit output
        self.digit_decoder = nn.Linear(register_dim, 10)  # 10 classes (0-9)

    def decode_scalar(self, state_registers: torch.Tensor, output_reg: int = 0) -> torch.Tensor:
        """
        Decode a single register to a scalar.

        Args:
            state_registers: (batch, num_registers, register_dim)
            output_reg: Which register to read.

        Returns:
            (batch, 1) predicted scalar value.
        """
        reg_val = state_registers[:, output_reg, :]  # (batch, dim)
        return self.scalar_decoder(reg_val)  # (batch, 1)

    def decode_digits(
        self,
        state_registers: torch.Tensor,
        num_digits: int,
        start_reg: int = 0,
    ) -> torch.Tensor:
        """
        Decode multiple registers as digits.

        Args:
            state_registers: (batch, num_registers, register_dim)
            num_digits: Number of digit registers to decode.
            start_reg: Starting register index.

        Returns:
            (batch, num_digits) digit logits.
        """
        digits = []
        for i in range(min(num_digits, state_registers.shape[1] - start_reg)):
            reg_val = state_registers[:, start_reg + i, :]
            digit_logits = self.digit_decoder(reg_val)  # (batch, 10)
            digits.append(digit_logits)

        if not digits:
            return torch.zeros(state_registers.shape[0], 0, device=state_registers.device)

        return torch.stack(digits, dim=1)  # (batch, num_digits, 10)

    def forward(
        self,
        state_registers: torch.Tensor,
        output_mode: str = "scalar",
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            state_registers: (batch, num_registers, register_dim)
            output_mode: 'scalar' or 'digits'
        """
        if output_mode == "scalar":
            return self.decode_scalar(state_registers, kwargs.get("output_reg", 0))
        elif output_mode == "digits":
            return self.decode_digits(
                state_registers,
                kwargs.get("num_digits", self.num_output_digits),
                kwargs.get("start_reg", 0),
            )
        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")