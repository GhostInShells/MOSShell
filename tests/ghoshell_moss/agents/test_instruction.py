"""Instruction assembly — meta + source + __interfaces__ expansion."""

from __future__ import annotations

from types import ModuleType

import pytest

from ghoshell_moss.agents._instruction import (
    INTERFACES_ATTR,
    META_INSTRUCTION,
    Promptable,
    assemble_instruction,
    reflect_element,
)


def _fake_module(name: str = "probe", **attrs) -> ModuleType:
    m = ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_no_interfaces_yields_no_appendix():
    src = '"""Hello."""\nimport math\n'
    out = assemble_instruction(name="probe", source=src, module=_fake_module())
    assert "```python" in out
    assert src.rstrip() in out
    # No appendix header at all — empty declaration means empty
    assert "Interfaces declared" not in out
    assert "<interface:" not in out


def test_class_element_reflects_public_methods():
    class Widget:
        """A widget."""
        def spin(self, speed: int) -> None: ...
        def _private(self) -> None: ...  # must not appear

    m = _fake_module(__interfaces__=[Widget])
    out = assemble_instruction(name="p", source='"""x"""', module=m)
    assert "<interface:Widget>" in out
    assert "spin" in out
    assert "_private" not in out


def test_promptable_element_uses_self_prompt():
    class Handle:
        __name__ = "Handle"
        def __prompt__(self) -> str:
            return "runtime-computed description here"

    inst = Handle()
    assert isinstance(inst, Promptable)
    m = _fake_module(__interfaces__=[inst])
    out = assemble_instruction(name="p", source='"""x"""', module=m)
    assert "runtime-computed description here" in out
    assert "<interface:Handle>" in out


def test_non_list_interfaces_raises():
    m = _fake_module(__interfaces__="not a list")
    with pytest.raises(TypeError, match="list or tuple"):
        assemble_instruction(name="p", source='"""x"""', module=m)


def test_reflect_element_matches_appendix_routing():
    """get_interface (sandbox-injected) uses the same routing as __interfaces__."""
    class Gadget:
        def act(self) -> None: ...

    out_from_get = reflect_element(Gadget)
    m = _fake_module(__interfaces__=[Gadget])
    out_from_appendix = assemble_instruction(name="p", source='"""x"""', module=m)
    # The class body from get_interface must be a substring of the appendix
    assert out_from_get.strip() in out_from_appendix


def test_meta_names_the_agent():
    out = assemble_instruction(name="translator", source='"""t"""', module=_fake_module())
    assert "`translator`" in out
    assert "sandbox_exec" in out
    assert "get_interface" in out
