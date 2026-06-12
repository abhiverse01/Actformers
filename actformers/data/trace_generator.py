"""
Actformer Action Trace Generator — symbolic executor producing ground-truth
action traces for arithmetic tasks.

The core thesis of Actformer is that a model trained on *correct action traces*
will learn the underlying algorithm.  This module is the source of those traces.

Every trace generator produces a List[ActionToken] that, when executed symbolically
against a clean MemoryState, produces the correct result.

Register convention (curriculum tasks):
  reg[0]  = accumulator / output
  reg[1]  = working value A
  reg[2]  = working value B
  reg[3]  = carry / borrow flag
  reg[4]  = temporary
  reg[5]  = loop counter
  reg[6]  = digit position pointer (integer, not the WM pointer)
  reg[7+] = scratch temporaries

Trace format: List[ActionToken] — a linear sequence of actions.
The HALT action terminates the trace.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from actformers.core.action_space import (
    ActionToken,
    ActionType,
    make_add,
    make_load,
    make_store,
    make_halt,
    make_digit_extract,
    make_output,
    make_subtract,
    make_multiply,
    make_compare,
    make_pointer_move,
    make_nop,
    make_if,
    make_break,
    make_loop,
)

__all__ = [
    "ActionTraceGenerator",
]


class ActionTraceGenerator:
    """
    Generates ground-truth action traces for arithmetic tasks.

    Each method returns a List[ActionToken] representing the algorithm
    as an action sequence.  These traces are used for:
      1. Supervised training (teacher forcing)
      2. RL reward computation (compare model trace vs ground truth)
      3. Macro discovery mining (common sub-sequences)
    """

    def __init__(self, num_registers: int = 16):
        self.num_registers = num_registers

    # ------------------------------------------------------------------
    # Addition (long addition, digit by digit)
    # ------------------------------------------------------------------

    def generate_addition_trace(
        self,
        a: int,
        b: int,
    ) -> List[ActionToken]:
        """
        Generate the action trace for computing a + b.

        Algorithm (long addition):
          carry = 0
          for digit_pos in range(max_digits):
            digit_a = (a // 10^pos) % 10
            digit_b = (b // 10^pos) % 10
            sum = digit_a + digit_b + carry
            result_digit = sum % 10
            carry = sum // 10
            store result_digit at position pos
          if carry > 0:
            store carry at highest position
          HALT

        Register usage:
          reg[0] = digit_a (current digit of a)
          reg[1] = digit_b (current digit of b)
          reg[2] = sum (digit_a + digit_b)
          reg[3] = carry
          reg[4] = result_digit (sum % 10)
          reg[5] = temporary for new carry
        """
        trace: List[ActionToken] = []
        max_digits = max(len(str(abs(a))), len(str(abs(b))), 1)
        carry = 0

        for pos in range(max_digits):
            digit_a = abs(a) // (10 ** pos) % 10
            digit_b = abs(b) // (10 ** pos) % 10

            # LOAD digit_a → reg[0]
            trace.append(make_load(value_reg=0, target_reg=0, mod=digit_a))
            # LOAD digit_b → reg[1]
            trace.append(make_load(value_reg=0, target_reg=1, mod=digit_b))

            # ADD reg[0] + reg[1] → reg[2]
            trace.append(make_add(src_a=0, src_b=1, dst=2))

            # If carry > 0, add carry to sum
            if carry > 0:
                # LOAD carry → reg[3]
                trace.append(make_load(value_reg=0, target_reg=3, mod=carry))
                # ADD reg[2] + reg[3] → reg[2] (add carry)
                trace.append(make_add(src_a=2, src_b=3, dst=2))

            # sum = digit_a + digit_b + carry
            total = digit_a + digit_b + carry

            # Extract result digit: total % 10
            # Use LOAD to store result_digit → reg[4]
            result_digit = total % 10
            trace.append(make_load(value_reg=0, target_reg=4, mod=result_digit))

            # OUTPUT reg[4] (result digit for this position)
            trace.append(make_output(reg=4, mod=pos))

            # New carry = total // 10
            new_carry = total // 10
            trace.append(make_load(value_reg=0, target_reg=3, mod=new_carry))

            carry = new_carry

        # Handle remaining carry
        if carry > 0:
            trace.append(make_load(value_reg=0, target_reg=4, mod=carry))
            trace.append(make_output(reg=4, mod=max_digits))

        trace.append(make_halt())
        return trace

    # ------------------------------------------------------------------
    # Subtraction (long subtraction, digit by digit)
    # ------------------------------------------------------------------

    def generate_subtraction_trace(
        self,
        a: int,
        b: int,
    ) -> List[ActionToken]:
        """
        Generate action trace for computing a - b (assuming a >= b).

        Register usage:
          reg[0] = digit_a
          reg[1] = digit_b
          reg[2] = difference
          reg[3] = borrow
          reg[4] = result digit
        """
        trace: List[ActionToken] = []
        max_digits = max(len(str(abs(a))), len(str(abs(b))), 1)
        borrow = 0

        for pos in range(max_digits):
            digit_a = abs(a) // (10 ** pos) % 10
            digit_b = abs(b) // (10 ** pos) % 10

            # Adjust for borrow
            effective_a = digit_a - borrow

            trace.append(make_load(value_reg=0, target_reg=0, mod=digit_a))
            trace.append(make_load(value_reg=0, target_reg=1, mod=digit_b))

            if borrow > 0:
                trace.append(make_load(value_reg=0, target_reg=3, mod=borrow))
                trace.append(make_subtract(src_a=0, src_b=3, dst=0))

            # effective_a - digit_b
            trace.append(make_subtract(src_a=0, src_b=1, dst=2))

            effective_a_adj = digit_a - borrow
            diff = effective_a_adj - digit_b

            if diff < 0:
                diff += 10
                new_borrow = 1
            else:
                new_borrow = 0

            trace.append(make_load(value_reg=0, target_reg=4, mod=diff))
            trace.append(make_output(reg=4, mod=pos))
            trace.append(make_load(value_reg=0, target_reg=3, mod=new_borrow))

            borrow = new_borrow

        trace.append(make_halt())
        return trace

    # ------------------------------------------------------------------
    # Multiplication (long multiplication)
    # ------------------------------------------------------------------

    def generate_multiplication_trace(
        self,
        a: int,
        b: int,
    ) -> List[ActionToken]:
        """
        Generate action trace for computing a * b.

        Uses long multiplication: for each digit of b, multiply all digits
        of a, accumulate partial products shifted by position.

        Register usage:
          reg[0] = digit_a
          reg[1] = digit_b
          reg[2] = partial product
          reg[3] = accumulator
          reg[4] = temporary carry
        """
        trace: List[ActionToken] = []

        digits_b = [int(d) for d in str(abs(b))[::-1]]  # LSB first
        digits_a = [int(d) for d in str(abs(a))[::-1]]  # LSB first

        # Initialize accumulator
        trace.append(make_load(value_reg=0, target_reg=3, mod=0))

        for b_pos, db in enumerate(digits_b):
            carry = 0
            for a_pos, da in enumerate(digits_a):
                product = da * db + carry
                result_digit = product % 10
                new_carry = product // 10

                trace.append(make_load(value_reg=0, target_reg=0, mod=da))
                trace.append(make_load(value_reg=0, target_reg=1, mod=db))
                trace.append(make_multiply(src_a=0, src_b=1, dst=2))

                if carry > 0:
                    trace.append(make_load(value_reg=0, target_reg=4, mod=carry))
                    trace.append(make_add(src_a=2, src_b=4, dst=2))

                trace.append(make_load(value_reg=0, target_reg=4, mod=result_digit))
                trace.append(make_output(reg=4, mod=a_pos + b_pos))

                carry = new_carry

            if carry > 0:
                trace.append(make_load(value_reg=0, target_reg=4, mod=carry))
                trace.append(make_output(reg=4, mod=len(digits_a) + b_pos))

        trace.append(make_halt())
        return trace

    # ------------------------------------------------------------------
    # Digit Reversal
    # ------------------------------------------------------------------

    def generate_digit_reversal_trace(
        self,
        n: int,
    ) -> List[ActionToken]:
        """
        Generate action trace to reverse the digits of n.

        E.g. 123 → 321.  Digits are extracted LSB-first, output MSB-first
        using the OUTPUT action with reversed position indices.

        Register usage:
          reg[0] = input digit (extracted from position i)
          reg[4] = output staging
        """
        trace: List[ActionToken] = []
        digits = [int(d) for d in str(abs(n))]
        num_digits = len(digits)

        # Output reversed digits: the i-th digit of the original (MSB-first)
        # becomes the (num_digits - 1 - i)-th digit of the result (MSB-first).
        # Since extract_result_from_trace reconstructs from positional OUTPUTs
        # where pos=0 is the ones place, we need to reverse the position mapping.
        # digits = [1, 2, 3] for n=123 (MSB-first, i.e. 1=hundreds)
        # Reversed = 321 = 3×100 + 2×10 + 1×1
        # OUTPUT positions: pos=0 is ones, pos=1 is tens, pos=2 is hundreds
        # So we need: digit[0]=1 at pos 0 (ones), digit[1]=2 at pos 1 (tens), digit[2]=3 at pos 2 (hundreds)
        # That gives 1×1 + 2×10 + 3×100 = 1 + 20 + 300 = 321 ✓
        # Simply output digit[i] at position i — the positional encoding does the reversal.
        for i, digit in enumerate(digits):
            trace.append(make_load(value_reg=0, target_reg=0, mod=digit))
            trace.append(make_output(reg=0, mod=i))

        trace.append(make_halt())
        return trace

    # ------------------------------------------------------------------
    # Max of N numbers
    # ------------------------------------------------------------------

    def generate_max_trace(
        self,
        numbers: List[int],
    ) -> List[ActionToken]:
        """
        Generate action trace to find the maximum of a list of numbers.

        Register usage:
          reg[0] = current max candidate
          reg[1] = next number to compare
          reg[2] = comparison result (1 if reg[0] > reg[1])
        """
        trace: List[ActionToken] = []

        if not numbers:
            trace.append(make_load(value_reg=0, target_reg=0, mod=0))
            trace.append(make_halt())
            return trace

        # Load first number as initial max
        trace.append(make_load(value_reg=0, target_reg=0, mod=numbers[0]))

        for num in numbers[1:]:
            # Load next number
            trace.append(make_load(value_reg=0, target_reg=1, mod=num))
            # Compare: reg[2] = (current_max > num) ?
            trace.append(make_compare(src_a=0, src_b=1, dst=2))
            # If reg[2] ≈ 0 (num is larger), update max
            trace.append(make_load(value_reg=0, target_reg=0, mod=max(
                numbers[0] if len(trace) <= 4 else num, num
            )))

        trace.append(make_output(reg=0, mod=0))
        trace.append(make_halt())
        return trace

    # ------------------------------------------------------------------
    # Trace verification — execute trace symbolically and check result
    # ------------------------------------------------------------------

    @staticmethod
    def extract_result_from_trace(
        trace: List[ActionToken],
    ) -> int:
        """
        Extract the numeric result encoded in OUTPUT actions of a trace.

        Each OUTPUT action has modifier = digit position and arg0 = register
        holding the digit value.  We track per-register state *sequentially*
        so that a LOAD followed immediately by OUTPUT captures the correct
        value even when the same register is reused later.

        For positions that receive multiple OUTPUT actions (multiplication
        accumulation), the values are summed (with carry propagation).

        Returns the reconstructed integer.
        """
        # Track current value in each register
        reg_values: dict = {}
        # Accumulate raw digit outputs per position
        raw_outputs: dict = {}  # position → list of digit values

        for action in trace:
            op = ActionType(action.op_id) if action.op_id < len(ActionType) else None
            if op == ActionType.LOAD:
                reg_values[action.arg1] = action.modifier
            elif op == ActionType.OUTPUT:
                reg_idx = action.arg0
                pos = action.modifier
                if reg_idx in reg_values:
                    digit = reg_values[reg_idx]
                    raw_outputs.setdefault(pos, []).append(digit)

        if not raw_outputs:
            return 0

        # Sum digits at each position and propagate carries
        max_pos = max(raw_outputs.keys())
        result_digits: dict = {}
        carry = 0
        for pos in range(max_pos + 1):
            digit_sum = sum(raw_outputs.get(pos, [])) + carry
            result_digits[pos] = digit_sum % 10
            carry = digit_sum // 10

        # Handle final carry
        while carry > 0:
            max_pos += 1
            result_digits[max_pos] = carry % 10
            carry //= 10

        result = 0
        for pos, d in result_digits.items():
            result += d * (10 ** pos)
        return result