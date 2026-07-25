"""Injection stub behavior — pre-swap namespaces must stay reflectable."""

from __future__ import annotations

import pytest

from ghoshell_moss.agents.injections import get_ctx, get_file_editor


def test_dunder_probe_fails_softly():
    # hasattr only swallows AttributeError. If dunder access raised
    # RuntimeError, any reflection pass over a pre-swap namespace
    # (e.g. rendering an agent's instruction without running it) would die.
    stub = get_file_editor()
    assert not hasattr(stub, "__prompt__")
    assert not hasattr(stub, "__wrapped__")


def test_capability_access_raises_helpful_error():
    stub = get_ctx()
    with pytest.raises(RuntimeError, match="unbound sandbox injection"):
        stub.keys()


def test_repr_names_the_injection():
    assert repr(get_file_editor()) == "<Unbound file_editor>"
    assert repr(get_ctx()) == "<Unbound ctx>"
