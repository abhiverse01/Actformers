"""
Evaluation metrics for Actformer.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

__all__ = [
    "exact_match_accuracy",
    "per_digit_accuracy",
    "ood_generalization_gap",
]


def exact_match_accuracy(
    predictions: List[int],
    targets: List[int],
) -> float:
    """
    Fraction of predictions that exactly match the target.

    This is the strictest metric — the model must produce the exact integer.
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def per_digit_accuracy(
    predictions: List[int],
    targets: List[int],
) -> float:
    """
    Fraction of individual digits that are correct.

    This gives partial credit for numbers that are close but not exact.
    """
    if not predictions:
        return 0.0
    total_digits = 0
    correct_digits = 0

    for pred, target in zip(predictions, targets):
        pred_str = str(abs(pred))
        target_str = str(abs(target))
        max_len = max(len(pred_str), len(target_str))
        pred_str = pred_str.zfill(max_len)
        target_str = target_str.zfill(max_len)
        for pc, tc in zip(pred_str, target_str):
            total_digits += 1
            if pc == tc:
                correct_digits += 1

    return correct_digits / max(total_digits, 1)


def ood_generalization_gap(
    train_digits_accuracy: float,
    test_digits_accuracies: Dict[int, float],
) -> Dict[str, float]:
    """
    Compute the OOD generalization gap.

    The gap is the difference between in-distribution accuracy and each
    out-of-distribution test level.

    The key metric: does accuracy hold at 2x, 5x, 10x train scale?

    Returns:
        Dict mapping test_digit_level to accuracy gap.
    """
    gaps = {}
    for test_digits, accuracy in test_digits_accuracies.items():
        gaps[f"gap_{test_digits}x"] = train_digits_accuracy - accuracy
    return gaps