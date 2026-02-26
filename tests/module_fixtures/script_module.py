"""Fixture script module for ModuleChannel path-loading tests."""

counter = 0


def add(a: int, b: int) -> int:
    return a + b


def inc() -> int:
    global counter
    counter += 1
    return counter
