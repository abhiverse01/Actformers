#!/usr/bin/env python3
"""
Training entry point for Actformer.

Usage:
    python scripts/train.py                  # defaults
    python scripts/train.py model=small       # override config
    python scripts/train.py task=multiplication
"""

from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    import torch
    from actformers.core.action_space import ActionSpace
    from actformers.core.model import Actformer
    from actformers.data.trace_generator import ActionTraceGenerator
    from actformers.training.supervised_trainer import SupervisedTrainer
    from actformers.training.losses import ActionCrossEntropyLoss
    from actformers.data.datasets import AdditionDataset

    device = torch.device("cpu")

    # Create model
    model = Actformer(
        num_registers=8,
        register_dim=32,
        scratchpad_size=64,
        scratchpad_dim=32,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        max_steps=20,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")

    # Create action space for dataset
    action_space = ActionSpace(num_registers=8)

    # Create dataset
    dataset = AdditionDataset(
        num_samples=1000,
        min_digits=1,
        max_digits=2,
        action_space=action_space,
        max_trace_length=20,
    )
    print(f"Dataset created with {len(dataset)} samples")

    # Quick forward pass test
    sample = dataset[0]
    inp = sample['input'].unsqueeze(0).to(device)
    trace = sample['flat_trace'][:sample['trace_length']].tolist()

    output, info = model(inp, target_trace=trace, execution_mode="train")
    print(f"\nForward pass OK:")
    print(f"  Output shape: {output.shape}")
    print(f"  Actions taken: {info['actions_taken']}")
    print(f"  Memory utilization: {info['memory_summary']['memory_utilization']:.2%}")

    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        action_loss=ActionCrossEntropyLoss(),
        lr=1e-4,
    )

    print("\n✓ Training setup complete. Ready to train.")
    print("Use the CurriculumTrainer for phased training.")
    print("Use the RLTrainer for reinforcement learning fine-tuning.")


if __name__ == "__main__":
    main()