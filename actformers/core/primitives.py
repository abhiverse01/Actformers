"""
Actformer Differentiable Primitives — the computational building blocks.

Every primitive is a pure function (batch, dim) → (batch, dim) that is
fully differentiable so that gradients flow through the execution engine
during training.

Where an operation has a naturally discrete aspect (e.g. DIGIT_EXTRACT
extracting the n-th digit), we use **straight-through estimators**:
forward pass is discrete, backward pass pretends the operation was continuous.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DifferentiablePrimitives",
]


class DifferentiablePrimitives(nn.Module):
    """
    Collection of differentiable computational primitives.

    Convention: all ``operate`` methods take two tensor arguments of shape
    ``(batch, dim)`` and return a tensor of the same shape, unless otherwise
    noted.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        # Learned scale/shift for digit pack/unpack to keep values in range
        self.digit_scale = nn.Parameter(torch.ones(dim))

    # ----------------------------------------------------------------
    # Arithmetic (fully differentiable)
    # ----------------------------------------------------------------

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise addition. dL/da = dL/dout, dL/db = dL/dout."""
        return a + b

    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise subtraction. dL/da = dL/dout, dL/db = -dL/dout."""
        return a - b

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise multiplication. dL/da = dL/dout * b, dL/db = dL/dout * a."""
        return a * b

    def divide_safe(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Element-wise division with safe denominator clamping.
        dL/db flows through 1/(b + eps).
        """
        return a / (b + eps)

    def mod_op(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Element-wise modulo. Uses the identity a % b = a - b * floor(a / (b+eps)).
        Gradient flows through the floor operation (piecewise-constant, so gradient
        is zero almost everywhere — this is a known limitation).
        """
        return a - b * torch.floor(a / (b + eps))

    # ----------------------------------------------------------------
    # Comparison (soft — sigmoid-based)
    # ----------------------------------------------------------------

    def compare(self, a: torch.Tensor, b: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Soft comparison: returns ≈1 where a > b, ≈0 where a < b.

        Uses sigmoid((a - b) / T).  Lower temperature → sharper, closer to
        hard comparison but with vanishing gradients.

        Gradient: d/da = σ'((a-b)/T) / T, which is well-behaved everywhere.
        """
        return torch.sigmoid((a - b) / temperature)

    def max_soft(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Soft maximum: differentiable approximation of max(a, b).

        Uses the log-sum-exp trick:  max ≈ log(e^a + e^b) - log(2).
        Gradient is a weighted average of the two inputs.
        """
        return torch.logsumexp(torch.stack([a, b], dim=0), dim=0) - math.log(2)

    def min_soft(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Soft minimum via: min(a,b) = a + b - max(a,b).
        """
        return a + b - self.max_soft(a, b)

    # ----------------------------------------------------------------
    # Digit operations (straight-through estimators)
    # ----------------------------------------------------------------

    def digit_extract(
        self,
        value: torch.Tensor,
        digit_pos: int,
        base: int = 10,
    ) -> torch.Tensor:
        """
        Extract the *digit_pos*-th digit from each element of *value*.

        Forward (discrete):
            digit = (floor(|value|) // base^pos) % base  →  normalised to [0, 1]

        Backward (straight-through):
            Passes gradient through as if the operation were identity.

        Args:
            value: (batch, dim) — each element encodes a scalar in its first component.
            digit_pos: which digit to extract (0 = ones, 1 = tens, …).

        Returns:
            (batch, dim) with the extracted digit in the first component,
            zeros elsewhere (for clean register semantics).
        """
        # Forward: discrete extraction
        with torch.no_grad():
            v = value[:, 0:1].abs()
            divisor = base ** digit_pos
            digit = (torch.floor(v / divisor) % base) / base  # normalised to [0, 1]

        # Straight-through: pretend gradient flows through identity
        result = torch.zeros_like(value)
        result[:, 0:1] = digit + (value[:, 0:1] - value[:, 0:1].detach())
        return result

    def digit_pack(
        self,
        digit_tensor: torch.Tensor,
        digit_pos: int,
        accumulator: torch.Tensor,
        base: int = 10,
    ) -> torch.Tensor:
        """
        Pack a digit into position *digit_pos* of an accumulator.

        Forward (discrete):
            acc = acc + digit_value * base^pos

        Backward (straight-through):
            Gradient flows through as identity w.r.t. digit_tensor.
        """
        with torch.no_grad():
            digit_val = (digit_tensor[:, 0:1] * base).round() * (base ** digit_pos)
        # Straight-through
        new_acc = accumulator.clone()
        new_acc[:, 0:1] = accumulator[:, 0:1] + digit_val + (digit_tensor[:, 0:1] - digit_tensor[:, 0:1].detach())
        return new_acc

    # ----------------------------------------------------------------
    # Shift operations
    # ----------------------------------------------------------------

    def shift_left(self, value: torch.Tensor, shift: int = 1, base: float = 10.0) -> torch.Tensor:
        """Multiply by base^shift (equivalent to left-shifting digits). Differentiable."""
        return value * (base ** shift)

    def shift_right(self, value: torch.Tensor, shift: int = 1, base: float = 10.0) -> torch.Tensor:
        """Divide by base^shift (equivalent to right-shifting digits). Differentiable."""
        return value / (base ** shift)

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    def scalar_to_embedding(self, scalar: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Map a (batch, 1) scalar tensor to (batch, dim) by broadcasting
        into the first component and zeroing the rest.
        """
        emb = torch.zeros(scalar.shape[0], dim, device=scalar.device, dtype=scalar.dtype)
        emb[:, 0:1] = scalar
        return emb
