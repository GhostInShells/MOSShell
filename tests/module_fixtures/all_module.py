"""Fixture module that defines __all__ exports."""

__all__ = ["only_this"]


def only_this() -> str:
    return "ok"


def not_exported() -> str:
    return "no"
