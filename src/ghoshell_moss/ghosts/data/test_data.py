"""Data Ghost memory tests. No network calls."""

import asyncio
from pathlib import Path

import pytest
from ghoshell_container import Container
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.test import TestModel

from ghoshell_moss.contracts.configs import ConfigStore, YamlConfigStore
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.ghost import GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.core.memento import MementoRef
from ghoshell_moss.ghosts.mock import MockArticulator
from ghoshell_moss.message import Message

from ._config import MemoryConfig
from ._memory import DataMemory
from ._meta import DataMeta
from ._reflection import DataReflector


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

    def test_control_methods_preserve_frozen_history(self, tmp_path: Path):
        memory = DataMemory(tmp_path / "memento", "data", auto_commit_every=0)
        memory.remember(_moment("first", "one"))
        view = memory.semantic_commit("first anchor")
        memory.reinterpret(str(view.seq), "corrected anchor")
        assert "corrected anchor" in memory.describe_commit(str(view.seq))
        child = memory.fork(str(view.seq), "alternate")
        assert child.meta.name == "alternate"
        assert memory.branch.all_commits()[0].id == view.id
        memory.close()

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

    def test_memento_control_surface_is_exposed_from_ghost_channel(self, tmp_path: Path):
        ghost = DataMeta(soul_content="be exact", model=TestModel()).factory(_container(tmp_path))
        assert ghost.channel().name() == "ghost"
        assert set(ghost.channel().main_state().own_commands()) >= {
            "memory_inspect",
            "memory_log",
            "memory_show",
            "memory_commit",
            "memory_reinterpret",
            "memory_fork",
            "memory_switch",
        }

    @pytest.mark.asyncio
    async def test_reflection_rewrites_note_without_touching_moment(self, tmp_path: Path):
        memory = DataMemory(tmp_path / "memento", "data", auto_commit_every=1)
        view = memory.remember(_moment("我喜欢短回答", "明白。"))
        assert view is not None
        reflector = DataReflector(
            Agent(model=TestModel(custom_output_text="用户偏好简洁回答。")),
            max_summary_chars=100,
            max_source_chars=1000,
        )
        reflected = await reflector.reflect(memory, view, Container())
        assert reflected is not None
        assert reflected.summary() == "用户偏好简洁回答。"
        assert "我喜欢短回答" in memory.commit_transcript(view.id, max_chars=1000)
        assert len(memory.branch.notes(view.id)) == 2
        memory.close()

    @pytest.mark.asyncio
    async def test_startup_chases_unreflected_mechanical_commit(self, tmp_path: Path):
        root = tmp_path / "memento"
        writer_meta = DataMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="stored answer"),
            memory_root=root,
            auto_commit_every=1,
            reflection_enabled=False,
        )
        writer = writer_meta.factory(_container(tmp_path / "workspace"))
        moment = _moment("remember startup", "")
        articulator = MockArticulator(moment)
        async with writer:
            logos = "".join([part async for part in writer.articulate(articulator)])
            moment.logos = logos
            writer.on_articulate_exit(articulator, logos, None)
        assert writer.memory.reflection_candidates()

        reader_meta = DataMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="normal answer"),
            reflection_model=TestModel(custom_output_text="启动追赶完成。"),
            memory_root=root,
            auto_commit_every=1,
        )
        reader = reader_meta.factory(_container(tmp_path / "workspace"))
        async with reader:
            for _ in range(20):
                if not reader.memory.reflection_candidates():
                    break
                await asyncio.sleep(0.01)
            assert reader.memory.reflection_candidates() == []
        assert reader.memory.branch.head().summary() == "启动追赶完成。"

    @pytest.mark.asyncio
    async def test_reflection_candidates_include_legacy_empty_commit(self, tmp_path: Path):
        memory = DataMemory(tmp_path / "memento", "data", auto_commit_every=0)
        memory.remember(_moment("legacy fact", "legacy answer"))
        view = memory.branch.commit("", kind="mechanical", by="data")
        assert [candidate.id for candidate in memory.reflection_candidates()] == [view.id]
        reflector = DataReflector(
            Agent(model=TestModel(custom_output_text="历史空摘要已补齐。")),
            max_summary_chars=100,
            max_source_chars=1000,
        )
        reflected = await reflector.reflect(memory, view, Container())
        assert reflected is not None
        assert reflected.summary() == "历史空摘要已补齐。"
        assert memory.reflection_candidates() == []
        memory.close()

    def test_memory_config_is_persisted_policy(self, tmp_path: Path):
        container = _container(tmp_path / "workspace")
        store = YamlConfigStore(LocalStorage(tmp_path / "configs"))
        store.save(MemoryConfig(detail_n=3, auto_commit_every=1, reflection_enabled=False))
        container.set(ConfigStore, store)
        ghost = DataMeta(soul_content="be exact", model=TestModel()).factory(container)
        assert ghost.memory.inspect()["detail_n"] == 3
        assert ghost.memory.inspect()["auto_commit_every"] == 1
        assert ghost.inspect_state()["reflection"]["enabled"] is False

    def test_relative_root_is_below_ghost_workspace(self, tmp_path: Path):
        meta = DataMeta(
            soul_content="be exact",
            model=TestModel(),
            memory_root="custom-memory",
        )
        ghost = meta.factory(_container(tmp_path))
        assert ghost.memory.root == tmp_path / "custom-memory"
