"""Sandbox builtins isolation — exec must not poison other sandboxes or the host.

Protocol commitment: the builtins dict a sandbox execs against is owned by that
sandbox. Code writing into `__builtins__` (e.g. `__builtins__['x'] = ...` or
`__builtins__.pop('len')`) must be contained — it cannot corrupt the shared
SANDBOX_BUILTINS constant, another sandbox's namespace, or the process builtins.
"""

from __future__ import annotations

import builtins

from ghoshell_moss.core.codex.sandbox import Sandbox


def test_root_sandbox_exec_cannot_poison_another_sandbox():
    s1 = Sandbox()
    s2 = Sandbox()

    r = s1.exec("__builtins__['sabotaged'] = True; __builtins__.pop('len', None)")
    assert r.exception is None

    r2 = s2.exec("__result__ = len([1, 2, 3])")
    assert r2.exception is None
    assert r2.returns == 3


def test_root_sandbox_exec_cannot_poison_future_sandboxes():
    Sandbox().exec("__builtins__.pop('len', None)")

    fresh = Sandbox().exec("__result__ = len('abc')")
    assert fresh.exception is None
    assert fresh.returns == 3


def test_unrestricted_sandbox_cannot_poison_process_builtins():
    sb = Sandbox(builtins=None)
    r = sb.exec("__builtins__['sabotaged'] = True; __builtins__.pop('len', None)")
    assert r.exception is None

    # Host process builtins untouched — len still resolves process-wide.
    assert 'len' in builtins.__dict__


def test_child_sandbox_mutation_is_contained_from_parent_builtins():
    parent = Sandbox()
    child = Sandbox(parent=parent)

    r = child.exec("__builtins__.pop('len', None)")
    assert r.exception is None

    # Another root sandbox unaffected by the parent/child family's mutation.
    probe = Sandbox().exec("__result__ = len([1])")
    assert probe.exception is None
    assert probe.returns == 1
