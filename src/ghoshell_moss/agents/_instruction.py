"""
Instruction assembly — the agent-family lens on "source is prompt".

An agent's instruction is a single string composed from three parts:

1. META — a short narrative telling the model where it is (the file it sees
   IS its definition, running in a sandbox, sandbox_exec is its only verb).
   Kept intentionally minimal: it does not describe capabilities, it names
   the situation.

2. SOURCE — the agent .py file verbatim. Its docstring is the task briefing;
   its imports are the capability declaration; comments ARE prompt. Nothing
   in this layer is added or paraphrased.

3. INTERFACES — expansion of the file's `__interfaces__` declaration.
   Only what the author explicitly listed gets an appendix block. No
   auto-scan of imports, no attr registry. Empty declaration → empty
   appendix (not even a header).

The model can pull further detail on demand via a `get_interface(obj)`
function injected into the sandbox — the reflection default is pull, not
push. That function is described in META and does not appear in the
instruction otherwise.

`prompt_sha` covers the FULL composed instruction, not the .py file bytes:
a meta template change or interfaces expansion change both count as
behaviour changes — attribution must not lie about that.

Backstage: this module never appears in an agent's instruction.
"""

from __future__ import annotations

import hashlib
import inspect
from types import ModuleType
from typing import Any, Callable, Protocol, runtime_checkable

from ghoshell_moss.core.codex._reflect import reflect_class_with_public_methods
from ghoshell_moss.core.codex._utils import get_callable_definition

__all__ = [
    "Promptable",
    "META_INSTRUCTION",
    "INTERFACES_ATTR",
    "assemble_instruction",
    "prompt_sha",
    "reflect_element",
]

INTERFACES_ATTR = "__interfaces__"

# The situating narrative. Reads Python fluently — everything else the model
# already knows from pre-training. Explains: the file it sees is what it IS,
# the sandbox is its body, and how to act.
#
# Deliberately does NOT describe: what Python is, what a sandbox is
# semantically, what tools look like, how memento works, what the interfaces
# appendix means. All redundant given the source and the tool's own signature.
META_INSTRUCTION = """\
You are MOSS agent `{name}`.

Your definition is the Python file below — compiled once and alive in a
sandbox you operate from within. Its module docstring is your task briefing;
its imports and top-level bindings are your entire capability surface.
Nothing outside this namespace exists for you: your imports are your
authorization.

```python
{source}
```
{interfaces_block}
You act by calling the `sandbox_exec` tool with Python code. The namespace
is cumulative — variables persist across calls, like a REPL. To inspect any
value you were given, call `get_interface(value)` inside the sandbox; it
returns the same interface view you would see with `moss codex get-interface`.

When you have your final answer, reply in plain text instead of calling
`sandbox_exec`.
"""


@runtime_checkable
class Promptable(Protocol):
    """
    Anything the assembler will ask for a self-description.

    Implement this on a runtime-constructed object (e.g. a config instance,
    a live handle) to have its prompt block generated dynamically. Note:
    dynamic elements break the "source alone determines instruction" story
    — prompt_sha still covers the composed result, so attribution stays
    honest, but the composed text can no longer be reproduced from the .py
    alone.
    """

    def __prompt__(self) -> str: ...


def _reflect_interface_element(element: Any) -> str:
    """One item from `__interfaces__` → its prompt block."""
    if isinstance(element, Promptable):
        return element.__prompt__().rstrip()
    if inspect.isclass(element):
        return reflect_class_with_public_methods(element).rstrip()
    if inspect.isfunction(element) or inspect.ismethod(element):
        return get_callable_definition(element).rstrip()
    if inspect.ismodule(element):
        # A whole module in __interfaces__ is unusual — the situating story
        # is that authors declare *interfaces*, not modules. Fall through
        # to str-of-repr so it is at least visible.
        return f"# module {element.__name__}"
    return str(element)


def _element_label(element: Any) -> str:
    """Best-effort name for the <interface:...> tag."""
    name = getattr(element, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(element).__name__


def _render_interfaces_block(module: ModuleType) -> str:
    """Read __interfaces__ and expand. Empty declaration → empty string."""
    declared = getattr(module, INTERFACES_ATTR, None)
    if not declared:
        return ""
    if not isinstance(declared, (list, tuple)):
        raise TypeError(
            f"{INTERFACES_ATTR} must be a list or tuple, got {type(declared).__name__}"
        )

    blocks: list[str] = []
    for element in declared:
        rendered = _reflect_interface_element(element)
        if not rendered.strip():
            continue
        label = _element_label(element)
        blocks.append(f"<interface:{label}>\n{rendered}\n</interface:{label}>")

    if not blocks:
        return ""

    body = "\n\n".join(blocks)
    return (
        "\nInterfaces declared in your file (`__interfaces__`), expanded here "
        "so you don't have to look them up:\n\n"
        f"{body}\n"
    )


def assemble_instruction(*, name: str, source: str, module: ModuleType) -> str:
    """Compose the full instruction the model will receive as system text."""
    return META_INSTRUCTION.format(
        name=name,
        source=source.rstrip(),
        interfaces_block=_render_interfaces_block(module),
    )


def prompt_sha(instruction: str) -> str:
    """SHA-256 of the composed instruction (not of the .py source alone).

    Kept short (16 hex chars) — enough to disambiguate across a single
    branch's history, cheap in metadata. Full hash is deterministic if
    you need it: hashlib.sha256(instruction.encode()).hexdigest().
    """
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]


def reflect_element(value: Any) -> str:
    """
    The `get_interface` function injected into agent sandboxes.

    Same routing as __interfaces__ expansion, but on demand: the model
    passes a live value (a function, a class it can see, an injected
    object) and gets back the same interface view a developer would see
    from `moss codex get-interface`.
    """
    return _reflect_interface_element(value)


# Sandbox-injected alias — the model calls this by the friendly name.
get_interface: Callable[[Any], str] = reflect_element
