"""
PyTorch Dataset classes for Actformer tasks.

Each dataset wraps a task sampler and produces dictionaries:
  {
    'input': Tensor of shape (input_len,),   — the problem operands
    'output': Tensor of shape (1,),           — the expected result
    'trace': List[ActionToken],              — ground-truth action trace
    'trace_length': int,                     — len(trace)
    'flat_trace': LongTensor,                — flat-encoded trace indices
    'operands': Tuple[int, ...],             — raw operand values (for logging)
  }
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from actformers.core.action_space import ActionToken, ActionSpace
from actformers.data.tasks.addition import AdditionTask
from actformers.data.tasks.subtraction import SubtractionTask
from actformers.data.tasks.multiplication import MultiplicationTask
from actformers.data.tasks.digit_reversal import DigitReversalTask

__all__ = [
    "AdditionDataset",
    "SubtractionDataset",
    "MultiplicationDataset",
    "DigitReversalDataset",
    "MultiTaskDataset",
]


class AdditionDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10000,
        min_digits: int = 1,
        max_digits: int = 5,
        action_space: Optional[ActionSpace] = None,
        max_trace_length: int = 100,
        normalize: bool = True,
        input_scale: float = 100.0,
    ):
        self.task = AdditionTask(min_digits=min_digits, max_digits=max_digits)
        self.num_samples = num_samples
        self.action_space = action_space
        self.max_trace_length = max_trace_length
        self.normalize = normalize
        self.input_scale = input_scale
        self._cache: List[Dict] = []
        self._build_cache()

    def _build_cache(self) -> None:
        for _ in range(self.num_samples):
            a, b, c, trace = self.task.sample()
            self._cache.append({
                'a': a, 'b': b, 'result': c,
                'trace': trace,
                'trace_length': len(trace),
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self._cache[idx]
        trace = item['trace']
        flat = [self.action_space.encode_token_flat(t) for t in trace] if self.action_space else []
        # Pad to max_trace_length
        if len(flat) < self.max_trace_length:
            flat = flat + [0] * (self.max_trace_length - len(flat))
        inp_a = item['a'] / self.input_scale if self.normalize else float(item['a'])
        inp_b = item['b'] / self.input_scale if self.normalize else float(item['b'])
        out_val = item['result'] / self.input_scale if self.normalize else float(item['result'])
        return {
            'input': torch.tensor([inp_a, inp_b], dtype=torch.float),
            'output': torch.tensor([out_val], dtype=torch.float),
            'trace': trace,
            'trace_length': item['trace_length'],
            'flat_trace': torch.tensor(flat[:self.max_trace_length], dtype=torch.long),
            'operands': (item['a'], item['b']),
        }


class SubtractionDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10000,
        min_digits: int = 1,
        max_digits: int = 5,
        action_space: Optional[ActionSpace] = None,
        max_trace_length: int = 100,
    ):
        self.task = SubtractionTask(min_digits=min_digits, max_digits=max_digits)
        self.num_samples = num_samples
        self.action_space = action_space
        self.max_trace_length = max_trace_length
        self._cache = []
        self._build_cache()

    def _build_cache(self):
        for _ in range(self.num_samples):
            a, b, c, trace = self.task.sample()
            self._cache.append({'a': a, 'b': b, 'result': c, 'trace': trace, 'trace_length': len(trace)})

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        item = self._cache[idx]
        trace = item['trace']
        flat = [self.action_space.encode_token_flat(t) for t in trace] if self.action_space else []
        if len(flat) < self.max_trace_length:
            flat = flat + [0] * (self.max_trace_length - len(flat))
        return {
            'input': torch.tensor([item['a'], item['b']], dtype=torch.float),
            'output': torch.tensor([item['result']], dtype=torch.float),
            'trace': trace,
            'trace_length': item['trace_length'],
            'flat_trace': torch.tensor(flat[:self.max_trace_length], dtype=torch.long),
            'operands': (item['a'], item['b']),
        }


class MultiplicationDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10000,
        min_digits: int = 1,
        max_digits: int = 3,
        action_space: Optional[ActionSpace] = None,
        max_trace_length: int = 200,
    ):
        self.task = MultiplicationTask(min_digits=min_digits, max_digits=max_digits)
        self.num_samples = num_samples
        self.action_space = action_space
        self.max_trace_length = max_trace_length
        self._cache = []
        self._build_cache()

    def _build_cache(self):
        for _ in range(self.num_samples):
            a, b, c, trace = self.task.sample()
            self._cache.append({'a': a, 'b': b, 'result': c, 'trace': trace, 'trace_length': len(trace)})

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        item = self._cache[idx]
        trace = item['trace']
        flat = [self.action_space.encode_token_flat(t) for t in trace] if self.action_space else []
        if len(flat) < self.max_trace_length:
            flat = flat + [0] * (self.max_trace_length - len(flat))
        return {
            'input': torch.tensor([item['a'], item['b']], dtype=torch.float),
            'output': torch.tensor([item['result']], dtype=torch.float),
            'trace': trace,
            'trace_length': item['trace_length'],
            'flat_trace': torch.tensor(flat[:self.max_trace_length], dtype=torch.long),
            'operands': (item['a'], item['b']),
        }


class DigitReversalDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10000,
        min_digits: int = 1,
        max_digits: int = 5,
        action_space: Optional[ActionSpace] = None,
        max_trace_length: int = 50,
    ):
        self.task = DigitReversalTask(min_digits=min_digits, max_digits=max_digits)
        self.num_samples = num_samples
        self.action_space = action_space
        self.max_trace_length = max_trace_length
        self._cache = []
        self._build_cache()

    def _build_cache(self):
        for _ in range(self.num_samples):
            n, rev, trace = self.task.sample()
            self._cache.append({'n': n, 'result': rev, 'trace': trace, 'trace_length': len(trace)})

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        item = self._cache[idx]
        trace = item['trace']
        flat = [self.action_space.encode_token_flat(t) for t in trace] if self.action_space else []
        if len(flat) < self.max_trace_length:
            flat = flat + [0] * (self.max_trace_length - len(flat))
        return {
            'input': torch.tensor([item['n']], dtype=torch.float),
            'output': torch.tensor([item['result']], dtype=torch.float),
            'trace': trace,
            'trace_length': item['trace_length'],
            'flat_trace': torch.tensor(flat[:self.max_trace_length], dtype=torch.long),
            'operands': (item['n'],),
        }


class MultiTaskDataset(Dataset):
    """Combines multiple task datasets with interleaved sampling."""

    def __init__(
        self,
        datasets: List[Dataset],
        task_probs: Optional[List[float]] = None,
    ):
        self.datasets = datasets
        self.task_probs = task_probs or [1.0 / len(datasets)] * len(datasets)
        self.cumulative_sizes = []
        cumsum = 0
        for ds in datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)
        self.total = cumsum

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Round-robin across datasets
        ds_idx = idx % len(self.datasets)
        sample_idx = idx // len(self.datasets) % len(self.datasets[ds_idx])
        item = self.datasets[ds_idx][sample_idx]
        item['task_id'] = ds_idx
        return item