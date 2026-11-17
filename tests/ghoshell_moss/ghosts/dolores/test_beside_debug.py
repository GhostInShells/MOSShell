"""``ghost.beside.debug`` — how the self-test surface behaves.

Two promises are tested here. The sandbox one: a file may define a pydantic model and may not
reach the machine through an import it was not granted. The command one: it never holds the
turn — the call answers with a receipt, and the validated answer (or the failure) arrives later
as a delivery of its own.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from ghoshell_moss.ghosts.dolores.mindset.beside.debug import (
    build_debug_channel,
    load_debug_model,
)

_SOURCE = """
from pydantic import BaseModel, Field

class Answer(BaseModel):
    ok: bool
    note: str = Field(default="", description="why")
"""

#: The shape every ego commit stores: turns only, no seqs.
_REF = {
    "session_id": "s1",
    "start_turn": 0,
    "end_turn": 1,
    "start_seq": None,
    "end_seq": None,
    "preset": None,
    "title": None,
    "cwd": None,
    "created": "2026-11-17T00:00:00Z",
}


class _Answer(BaseModel):
    score: float
    issues: list[str]


class _Memento:
    """Only the one read the command makes: a commit's stored session reference."""

    def resolve_commit(self, coord):
        if coord != "1-1":
            return None
        return SimpleNamespace(ref=SimpleNamespace(metadata={"ref": _REF}))


class _Tasks:
    """Stands in for ``EmanationTasks``: records the ask, then answers or fails on cue."""

    def __init__(self, result):
        self.result = result
        self.calls: list[tuple] = []

    async def request(self, prompt, model, *, anchor, timeout=None):
        self.calls.append((prompt, model, anchor, timeout))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _model_ref(tmp_path) -> str:
    path = tmp_path / "answer.py"
    path.write_text(
        "from pydantic import BaseModel\n\n"
        "class Answer(BaseModel):\n"
        "    score: float\n"
        "    issues: list[str]\n"
    )
    return f"{path}:Answer"


# ── the model reference ────────────────────────────────────────────────────────────────────


def test_module_ref_resolves_an_importable_class():
    cls, box = load_debug_model("pydantic:BaseModel", source=False)

    assert cls is BaseModel
    assert box is None


def test_source_ref_compiles_a_model_in_the_sandbox(tmp_path):
    path = tmp_path / "m.py"
    path.write_text(_SOURCE)

    cls, box = load_debug_model(f"{path}:Answer", source=True)
    try:
        assert issubclass(cls, BaseModel)
        assert cls(ok=True).note == ""
    finally:
        box.close()


def test_source_ref_refuses_an_ungranted_import(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("import os\nfrom pydantic import BaseModel\n")

    with pytest.raises(ValueError, match="refused 'os'"):
        load_debug_model(f"{path}:X", source=True)


def test_source_ref_must_resolve_to_a_model(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("class Nope:\n    pass\n")

    with pytest.raises(TypeError, match="BaseModel"):
        load_debug_model(f"{path}:Nope", source=True)


# ── the command: a receipt now, the result later ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_command_returns_a_receipt_and_delivers_the_answer_later(tmp_path, monkeypatch):
    """The turn is not held: the call answers with a receipt, the answer arrives as a delivery."""
    from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

    tasks = _Tasks(_Answer(score=6.0, issues=["ground is stale"]))
    chan = build_debug_channel(tasks, _Memento())
    sent = []
    monkeypatch.setattr(CommandUtil, "send_signal", lambda signal: sent.append(signal))

    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command(
            "emanation_task",
            args=("1-1", "how is the ground?", _model_ref(tmp_path)),
            kwargs={"source": True},
        )
        for _ in range(200):
            if sent:
                break
            await asyncio.sleep(0.01)

    assert receipt.startswith("[emanation] asking 1-1")
    assert len(sent) == 1
    assert sent[0].name == "mind/delivery"
    text = sent[0].messages[0].to_content_string()
    assert '"score": 6.0' in text
    assert "ground is stale" in text


@pytest.mark.asyncio
async def test_a_failure_is_delivered_instead_of_disappearing(tmp_path, monkeypatch):
    """A task that could not answer comes back as content; the caller is not left guessing."""
    from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

    tasks = _Tasks(RuntimeError("the model answered in prose"))
    chan = build_debug_channel(tasks, _Memento())
    sent = []
    monkeypatch.setattr(CommandUtil, "send_signal", lambda signal: sent.append(signal))

    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command(
            "emanation_task",
            args=("1-1", "how is the ground?", _model_ref(tmp_path)),
            kwargs={"source": True},
        )
        for _ in range(200):
            if sent:
                break
            await asyncio.sleep(0.01)

    assert receipt.startswith("[emanation] asking 1-1")
    assert len(sent) == 1
    text = sent[0].messages[0].to_content_string()
    assert "failed" in text
    assert "the model answered in prose" in text


@pytest.mark.asyncio
async def test_a_bad_model_reference_is_refused_in_the_receipt(monkeypatch):
    """A wrong model is a mistake in the call, not a result: it never starts a task."""
    from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

    tasks = _Tasks(_Answer(score=1.0, issues=[]))
    chan = build_debug_channel(tasks, _Memento())
    sent = []
    monkeypatch.setattr(CommandUtil, "send_signal", lambda signal: sent.append(signal))

    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command(
            "emanation_task", args=("1-1", "q", "json:JSONEncoder")
        )
        await asyncio.sleep(0.05)

    assert receipt.startswith("error: TypeError")
    assert tasks.calls == []
    assert sent == []


@pytest.mark.asyncio
async def test_an_unknown_coord_is_refused_before_any_work():
    tasks = _Tasks(_Answer(score=1.0, issues=[]))
    chan = build_debug_channel(tasks, _Memento())

    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command(
            "emanation_task", args=("9-9", "q", "pydantic:BaseModel")
        )

    assert receipt == "error: no commit at `9-9`"
    assert tasks.calls == []
