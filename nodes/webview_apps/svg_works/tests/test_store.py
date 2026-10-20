"""The store is the truth here: the page is only its projection.

These tests pin the three properties the whole design rests on — ops are append-only,
the cursor is a position (not a deletion), and a snapshot is a fold of the log.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[1] / "main.py"


def _load():
    name = "svg_works_main"
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolve their module through sys.modules while decorating, so the
    # module has to be registered before exec — otherwise @dataclass blows up.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


store_mod = _load()


def test_ops_are_append_only_and_numbered_from_one():
    store = store_mod.WorkStore()
    store.open("w")
    a = store.record("w", "insert", "#stage", {"html": "<g/>"}, "skeleton")
    b = store.record("w", "set", "[data-addr=arm]", {"attr": "fill", "value": "#fff"}, "")
    assert (a.n, b.n) == (1, 2)
    assert [op.n for op in store.get("w").ops] == [1, 2]


def test_checkout_moves_the_cursor_without_dropping_ops():
    store = store_mod.WorkStore()
    store.open("w")
    for _ in range(3):
        store.record("w", "insert", "#stage", {"html": "<g/>"}, "")
    store.checkout("w", 1)
    work = store.get("w")
    assert work.cursor == 1
    assert len(work.ops) == 3, "ops after the cursor must survive — checkout is a position"
    store.checkout("w", 99)
    assert store.get("w").cursor == 3, "a cursor past the end clamps to the end"


def test_new_op_lands_at_the_end_and_advances_the_cursor():
    store = store_mod.WorkStore()
    store.open("w")
    store.record("w", "insert", "#stage", {"html": "<g/>"}, "")
    store.checkout("w", 0)
    op = store.record("w", "set", "#stage", {"attr": "fill", "value": "#000"}, "")
    # Appending after a checkout is how a fork is expressed: the log keeps growing and
    # the reader follows the newest op.
    assert op.n == 2
    assert store.get("w").cursor == 2


def test_snapshot_is_the_fold_of_the_current_work():
    store = store_mod.WorkStore()
    store.open("a", "first")
    store.record("a", "insert", "#stage", {"html": "<g/>"}, "")
    store.open("b", "second")
    snap = store.snapshot()
    assert snap["current"] == "b"
    assert snap["ops"] == [], "a snapshot carries the current work's ops, not every work's"
    assert [w["name"] for w in snap["works"]] == ["a", "b"]


def test_unknown_work_is_reported_not_silently_created():
    store = store_mod.WorkStore()
    assert store.record("nope", "insert", "#stage", {"html": ""}, "") is None
    assert store.checkout("nope", 1) is None
