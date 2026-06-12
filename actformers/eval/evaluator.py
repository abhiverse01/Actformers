"""
Generalization Evaluator — OOD testing for the core thesis.

The primary research result is the OOD generalization table:
  Train digits | Test 1x | Test 2x | Test 5x | Test 10x
  2            | 98%     | ???     | ???     | ???
  3            | 99%     | ???     | ???     | ???

Transformers score near 0% at 5x.  Actformer should score >95% at any scale.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from actformers.core.model import Actformer
from actformers.eval.metrics import exact_match_accuracy, per_digit_accuracy, ood_generalization_gap

__all__ = ["GeneralizationEvaluator"]


class GeneralizationEvaluator:
    """
    Evaluates OOD generalization by training on N-digit problems and
    testing on M-digit problems where M > N.
    """

    def __init__(
        self,
        model: Actformer,
        execution_mode: str = "infer",
        device: Optional[torch.device] = None,
        output_scale: float = 100.0,
    ):
        self.model = model
        self.execution_mode = execution_mode
        self.device = device or torch.device("cpu")
        self.output_scale = output_scale

    def eval_addition(
        self,
        test_dataset,
        max_trace_length: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate addition performance on a test dataset.

        Returns:
            Dict with 'exact_match', 'per_digit', 'num_samples'.
        """
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for sample in test_dataset:
                inp = sample['input'].unsqueeze(0).to(self.device)
                target = sample['output'].item()

                output, info = self.model(inp, execution_mode=self.execution_mode)
                pred = round(output.item() * self.output_scale)

                predictions.append(pred)
                targets.append(int(target * self.output_scale))

        em_acc = exact_match_accuracy(predictions, targets)
        pd_acc = per_digit_accuracy(predictions, targets)

        return {
            'exact_match': em_acc,
            'per_digit': pd_acc,
            'num_samples': len(predictions),
            'predictions': predictions[:10],
            'targets': targets[:10],
        }

    def eval_ood(
        self,
        train_digits: int,
        test_digit_levels: List[int],
        dataset_builder,
        samples_per_level: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Full OOD evaluation: test at multiple digit scales.

        Args:
            train_digits: Number of digits used during training.
            test_digit_levels: List of digit counts to test at (e.g. [2, 4, 6, 10]).
            dataset_builder: Callable(num_samples, min_digits, max_digits) → Dataset.
            samples_per_level: Samples per test level.

        Returns:
            Nested dict: {digit_level: {metric_name: value}}
        """
        results = {}

        # Test at 1x (in-distribution)
        test_ds = dataset_builder(samples_per_level, train_digits, train_digits)
        results[train_digits] = self.eval_addition(test_ds)

        # Test at OOD levels
        for level in test_digit_levels:
            if level > train_digits:
                test_ds = dataset_builder(samples_per_level, level, level)
                results[level] = self.eval_addition(test_ds)

        # Compute gaps
        train_acc = results[train_digits]['exact_match']
        test_accs = {k: v['exact_match'] for k, v in results.items() if k != train_digits}
        gaps = ood_generalization_gap(train_acc, test_accs)

        return {
            'per_level': results,
            'gaps': gaps,
        }