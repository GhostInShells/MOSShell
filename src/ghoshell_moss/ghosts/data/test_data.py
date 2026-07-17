"""Data Ghost memory tests. No network calls."""

from pathlib import Path

import pytest
from ghoshell_container import Container
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.test import TestModel

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.ghost import GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.core.memento import MementoRef
from ghoshell_moss.ghosts.mock import MockArticulator
from ghoshell_moss.message import Message

from ._memory import DataMemory
from ._meta import DataMeta


def _moment(user: str, assistant: str) -> Moment:
    return Moment(
        percepts={"user": [Message.new().with_content(user)]},
        logos=assistant,
    )


def _history_text(history) -> str:
    return "\n".join(str(getattr(part, "content", "")) for message in history for part in message.parts)


def _container(home: Path) -> Container:
    container = Container()
    container.set(GhostWorkspace, GhostWorkspace(home=home, source=None))
    container.set(LoggerItf, get_moss_logger())
    return container


class TestDataMemory:
    def test_empty_memory(self, tmp_path: Path):
        memory = DataMemory(tmp_path / "memento", "data")
        assert memory.model_history() == []
        assert memory.messages() == []
        assert memory.inspect()["staging_count"] == 0

    def test_completed_moment_round_trip(self, tmp_path: Path):
        root = tmp_path / "memento"
        memory = DataMemory(root, "data", auto_commit_every=0)
        memory.remember(_moment("代号是琥珀-731", "我记住了。"))
        history = memory.model_history()
        assert [type(item) for item in history] == [ModelRequest, ModelResponse]
        text = _history_text(history)
        assert "琥珀-731" in text
        assert "我记住了" in text
        memory.close()

        reopened = DataMemory(root, "data", auto_commit_every=0)
        assert "琥珀-731" in _history_text(reopened.model_history())

    def test_auto_commit_and_folded_summary(self, tmp_path: Path):
        root = tmp_path / "memento"
        memory = DataMemory(
            root,
            "data",
            detail_n=1,
            auto_commit_every=1,
        )
        first = memory.remember(_moment("first", "one"))
        second = memory.remember(_moment("second", "two"))
        assert first is not None
        assert second is not None
        assert memory.branch.staging() == []
        assert len(memory.branch.own_commits()) == 2

        history = memory.model_history()
        text = _history_text(history)
        assert first.id in text
        assert "[extractive mechanical index]" in text
        assert "first" in text
        assert "one" in text
        assert "second" in text
        assert "two" in text

        memory.close()
        reopened = DataMemory(root, "data", detail_n=1, auto_commit_every=1)
        assert "first" in _history_text(reopened.model_history())

    def test_messages_keep_commit_reference(self, tmp_path: Path):
        memory = DataMemory(
            tmp_path / "memento",
            "data",
            detail_n=1,
            auto_commit_every=1,
        )
        first = memory.remember(_moment("first", "one"))
        memory.remember(_moment("second", "two"))
        summaries = [message for message in memory.messages() if MementoRef.read(message)]
        assert len(summaries) == 1
        assert MementoRef.read(summaries[0]).commit_id == first.id

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"detail_n": 0}, "detail_n"),
            ({"summary_m": -2}, "summary_m"),
            ({"auto_commit_every": -1}, "auto_commit_every"),
        ],
    )
    def test_invalid_policy(self, tmp_path: Path, kwargs: dict, message: str):
        with pytest.raises(ValueError, match=message):
            DataMemory(tmp_path / "memento", "data", **kwargs)


class TestDataGhost:
    @pytest.mark.asyncio
    async def test_articulate_then_reopen(self, tmp_path: Path):
        root = tmp_path / "persistent-memory"
        meta = DataMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="stored answer"),
            memory_root=root,
            auto_commit_every=0,
        )
        ghost = meta.factory(_container(tmp_path / "workspace"))
        moment = _moment("restart-secret-842", "")
        articulator = MockArticulator(moment)

        async with ghost:
            logos = "".join([part async for part in ghost.articulate(articulator)])
            moment.logos = logos  # GhostRuntime does this before the exit hook.
            ghost.on_articulate_exit(articulator, logos, None)
        assert logos == "stored answer"

        reopened = meta.factory(_container(tmp_path / "workspace"))
        history = reopened.memory.model_history()
        assert "restart-secret-842" in _history_text(history)
        assert "stored answer" in _history_text(history)
        await reopened.__aexit__(None, None, None)

    def test_failed_articulation_is_not_remembered(self, tmp_path: Path):
        meta = DataMeta(
            soul_content="be exact",
            model=TestModel(),
            memory_root=tmp_path / "memento",
        )
        ghost = meta.factory(_container(tmp_path / "workspace"))
        moment = _moment("must-not-persist", "partial")
        ghost.on_articulate_exit(
            MockArticulator(moment),
            "partial",
            RuntimeError("model failed"),
        )
        assert ghost.memory.branch.staging() == []
        assert ghost.inspect_context()["memory_write"] == "skipped_on_error"

    def test_default_root_is_ghost_workspace(self, tmp_path: Path):
        meta = DataMeta(soul_content="be exact", model=TestModel())
        ghost = meta.factory(_container(tmp_path))
        assert ghost.memory.root == tmp_path / "memento"
        assert ghost.memory.owner == "data"

    def test_relative_root_is_below_ghost_workspace(self, tmp_path: Path):
        meta = DataMeta(
            soul_content="be exact",
            model=TestModel(),
            memory_root="custom-memory",
        )
        ghost = meta.factory(_container(tmp_path))
        assert ghost.memory.root == tmp_path / "custom-memory"
