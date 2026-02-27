"""Fixture module that prints to stdout.

Used to ensure script_channel transport ignores non-protocol stdout lines.
"""


def add(a: int, b: int) -> int:
    print(f"add({a}, {b})")
    return a + b
