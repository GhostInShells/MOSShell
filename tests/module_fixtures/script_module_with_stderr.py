"""Fixture module that writes to stderr.

Used to validate script_channel stderr handling strategies.
"""

import sys


def add(a: int, b: int) -> int:
    print("hello from stderr", file=sys.stderr)
    return a + b
