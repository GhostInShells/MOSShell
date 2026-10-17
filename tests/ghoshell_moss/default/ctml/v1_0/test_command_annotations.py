"""A declared type must bind — the annotation is the contract, not a comment.

CTML parses attribute values as literals, so a numeric id arrives as an int and
a one-digit danmaku arrives as an int too. The command declares ``str``; binding
must hand it a string. This module carries `from __future__ import annotations`
on purpose: it is the condition under which the coercion silently stopped
working, and every node module carries it.
"""
from __future__ import annotations

import pytest

from ghoshell_moss.core.blueprint.channel_builder import new_command


async def _handle(id: str) -> str:
    """Stand-in for a node command that takes an item handle."""
    return f"{type(id).__name__}:{id}"


async def _ids(ids: str) -> str:
    """Stand-in for a command that also accepts a real sequence."""
    return f"{type(ids).__name__}:{ids}"


@pytest.mark.asyncio
async def test_a_declared_str_binds_a_number_to_a_string():
    """id="2" reaches the parameter typed ``str`` as the string "2".

    Before this, the signature carried the *string* 'str' as its annotation (see
    the module docstring), `callable('str')` was False, so the value stayed the
    int 2 — the command then looked up "2" with 2 and reported "no item 2"
    while item 2 was on screen.
    """
    cmd = new_command(_handle)
    assert await cmd(2) == "str:2"


@pytest.mark.asyncio
async def test_a_sequence_given_to_a_str_parameter_keeps_its_shape():
    """A real sequence is not flattened into a string.

    ``ids="0,1"`` is parsed by CTML into the tuple (0, 1) and the channel decides
    how to read it (the screen node splits it). ``str((0, 1))`` would turn it
    into "(0, 1)" and lose the elements.
    """
    cmd = new_command(_ids)
    assert await cmd((0, 1)) == "tuple:(0, 1)"


async def _ghost_type(value: "NoSuchType") -> str:  # noqa: F821
    """Annotation names a type that only exists under TYPE_CHECKING."""
    return f"{type(value).__name__}:{value}"


@pytest.mark.asyncio
async def test_an_unresolvable_annotation_falls_back_instead_of_killing_the_channel():
    """Resolving annotations is only worth it if failing to resolve them is survivable.

    A TYPE_CHECKING-only import — or a forward reference the module never imports —
    would otherwise make the command unusable and, worse, take the whole channel's
    interface down with it. Fall back to the raw signature instead.
    """
    cmd = new_command(_ghost_type)
    assert await cmd("x") == "str:x"
