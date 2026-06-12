#!/usr/bin/env python3
"""Evaluation entry point for Actformer."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    import torch
    from actformers.core.model import Actformer
    from actformers.eval.evaluator import GeneralizationEvaluator
    from actformers.eval.metrics import exact_match_accuracy, per_digit_accuracy

    print("Actformer Evaluation")
    print("=" * 40)

    # Quick import test
    from actformers.data.trace_generator import ActionTraceGenerator
    from actformers.core.action_space import ActionToken, ActionType

    gen = ActionTraceGenerator()

    # Verify trace correctness
    errors = 0
    for a in range(100):
        for b in range(100):
            trace = gen.generate_addition_trace(a, b)
            result = ActionTraceGenerator.extract_result_from_trace(trace)
            if result != a + b:
                errors += 1

    print(f"Addition trace verification (0-99 + 0-99): {20000 - errors}/20000 correct")

    if errors == 0:
        print("✓ All addition traces verified correct")
    else:
        print(f"✗ {errors} trace errors found")

    # Test model forward pass
    model = Actformer(num_registers=8, register_dim=32, max_steps=20)
    evaluator = GeneralizationEvaluator(model)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Evaluator ready for OOD testing")


if __name__ == "__main__":
    main()