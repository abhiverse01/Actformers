#!/usr/bin/env python3
"""
Visualize action traces as readable strings.

Usage:
    python scripts/visualize_actions.py 123 456
    python scripts/visualize_actions.py --task subtraction 100 50
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from actformers.core.action_space import ActionType
from actformers.data.trace_generator import ActionTraceGenerator


def format_trace(trace) -> str:
    """Format an action trace as a readable string."""
    lines = []
    for i, action in enumerate(trace):
        op = ActionType(action.op_id) if action.op_id < len(ActionType) else None
        if op is None:
            lines.append(f"  Step {i:3d}: NOP")
            continue
        if op == ActionType.LOAD:
            lines.append(f"  Step {i:3d}: LOAD {action.modifier} → reg[{action.arg1}]")
        elif op == ActionType.OUTPUT:
            lines.append(f"  Step {i:3d}: OUTPUT reg[{action.arg0}] → pos {action.modifier}")
        elif op == ActionType.ADD:
            carry = " (with carry)" if action.modifier == 1 else ""
            lines.append(f"  Step {i:3d}: ADD reg[{action.arg0}] + reg[{action.arg1}] → reg[{action.arg2}]{carry}")
        elif op == ActionType.SUBTRACT:
            lines.append(f"  Step {i:3d}: SUB reg[{action.arg0}] - reg[{action.arg1}] → reg[{action.arg2}]")
        elif op == ActionType.MULTIPLY:
            lines.append(f"  Step {i:3d}: MUL reg[{action.arg0}] × reg[{action.arg1}] → reg[{action.arg2}]")
        elif op == ActionType.HALT:
            lines.append(f"  Step {i:3d}: HALT")
        elif op == ActionType.DIGIT_EXTRACT:
            lines.append(f"  Step {i:3d}: DIGIT_EXTRACT reg[{action.arg0}] → reg[{action.arg1}], pos {action.arg2}")
        else:
            lines.append(f"  Step {i:3d}: {op.name}({action.arg0}, {action.arg1}, {action.arg2}, mod={action.modifier})")
    return "\n".join(lines)


def main():
    gen = ActionTraceGenerator()
    task = "addition"

    if len(sys.argv) < 3:
        print("Usage: python visualize_actions.py [--task TASK] a b")
        print("Tasks: addition, subtraction, multiplication, reversal")
        sys.exit(1)

    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        task = sys.argv[idx + 1]
        args = [x for x in sys.argv[1:] if x not in (sys.argv[idx], sys.argv[idx+1])]

    a, b = int(args[0]), int(args[1])

    if task == "addition":
        trace = gen.generate_addition_trace(a, b)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        print(f"Addition: {a} + {b} = {result}")
    elif task == "subtraction":
        if a < b:
            a, b = b, a
        trace = gen.generate_subtraction_trace(a, b)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        print(f"Subtraction: {a} - {b} = {result}")
    elif task == "multiplication":
        trace = gen.generate_multiplication_trace(a, b)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        print(f"Multiplication: {a} × {b} = {result}")
    elif task == "reversal":
        trace = gen.generate_digit_reversal_trace(a)
        result = ActionTraceGenerator.extract_result_from_trace(trace)
        print(f"Reversal: {a} → {result}")
    else:
        print(f"Unknown task: {task}")
        sys.exit(1)

    print(f"\nAction trace ({len(trace)} steps):")
    print(format_trace(trace))


if __name__ == "__main__":
    main()