"""The log is the truth here; the current state is only its fold.

These tests pin the three properties the mechanism rests on: every change is a new
event, folding is order-preserving, and each idea keeps a readable `.md` beside the log.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[1] / "main.py"


def _load():
    name = "ideas_main"
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module          # pydantic resolves models through sys.modules
    spec.loader.exec_module(module)
    return module


ideas_mod = _load()


def _store(tmp_path: Path):
    store = ideas_mod.IdeaStore()
    store.add_root(str(tmp_path))
    store.load()
    return store


def test_born_lands_as_an_event_and_a_readable_file(tmp_path):
    store = _store(tmp_path)
    root = store.roots[0]
    store.append(root, "born", "idea-1", {"text": "第一条", "priority": 2.0})
    assert [ev.op for ev in store.events] == ["born"]
    assert store.ideas["idea-1"].text == "第一条"
    assert store.ideas["idea-1"].priority == 2.0
    md = root / "ideas" / "idea-1.md"
    assert md.exists() and "第一条" in md.read_text(encoding="utf-8")


def test_state_and_priority_are_appended_not_overwritten(tmp_path):
    store = _store(tmp_path)
    root = store.roots[0]
    store.append(root, "born", "idea-1", {"text": "x"})
    store.append(root, "state", "idea-1", {"state": "active"})
    store.append(root, "priority", "idea-1", {"priority": 3.5})
    # the current state is the last word, but the record keeps every word
    assert store.ideas["idea-1"].state == "active"
    assert store.ideas["idea-1"].priority == 3.5
    assert [ev.op for ev in store.events] == ["born", "state", "priority"]


def test_fold_after_reload_matches_the_live_state(tmp_path):
    store = _store(tmp_path)
    root = store.roots[0]
    store.append(root, "born", "a", {"text": "a"})
    store.append(root, "born", "b", {"text": "b"})
    store.append(root, "link", "a", {"dst": "b", "kind": "depend", "weight": 2.0})
    store.append(root, "tag", "a", {"tag": "screen"})
    before = {k: (v.state, v.priority, len(v.links), tuple(v.tags)) for k, v in store.ideas.items()}

    fresh = ideas_mod.IdeaStore()
    fresh.add_root(str(tmp_path))
    fresh.load()
    after = {k: (v.state, v.priority, len(v.links), tuple(v.tags)) for k, v in fresh.ideas.items()}

    assert after == before, "replaying the log must rebuild exactly the state we were holding"


def test_a_broken_line_does_not_eat_the_record(tmp_path):
    store = _store(tmp_path)
    root = store.roots[0]
    store.append(root, "born", "a", {"text": "a"})
    log = root / "ideas" / "ideas.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    store.append(root, "state", "a", {"state": "done"})
    fresh = ideas_mod.IdeaStore()
    fresh.add_root(str(tmp_path))
    fresh.load()
    assert fresh.ideas["a"].state == "done"
