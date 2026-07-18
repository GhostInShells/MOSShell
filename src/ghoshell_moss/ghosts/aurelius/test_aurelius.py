"""Aurelius Ghost memory tests. No network calls."""

import asyncio
from pathlib import Path

import pytest
from ghoshell_container import Container
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from ghoshell_moss.contracts.configs import ConfigStore, YamlConfigStore
from ghoshell_moss.contracts.desktop import PathOutsideRootError
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.ghost import GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.core.memento import MementoRef
from ghoshell_moss.ghosts.mock import MockArticulator
from ghoshell_moss.message import Message

from ._config import MemoryConfig
from ._knowledge import MemoryProjection
from ._memory import AureliusMemory
from ._meta import AureliusMeta
from ._reflection import AureliusReflector


def _moment(user: str, assistant: str = "") -> Moment:
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


class TestAureliusMemory:
    def test_empty_memory(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius")
        assert memory.model_history() == []
        assert memory.messages() == []
        assert memory.inspect()["staging_count"] == 0

    def test_completed_moment_round_trip(self, tmp_path: Path):
        root = tmp_path / "memento"
        memory = AureliusMemory(root, "aurelius", auto_commit_every=0)
        memory.remember(_moment("代号是琥珀-731", "我记住了。"))
        history = memory.model_history()
        assert [type(item) for item in history] == [ModelRequest, ModelResponse]
        text = _history_text(history)
        assert "琥珀-731" in text
        assert "我记住了" in text
        memory.close()

        reopened = AureliusMemory(root, "aurelius", auto_commit_every=0)
        assert "琥珀-731" in _history_text(reopened.model_history())

    def test_auto_commit_and_folded_summary(self, tmp_path: Path):
        root = tmp_path / "memento"
        memory = AureliusMemory(
            root,
            "aurelius",
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
        reopened = AureliusMemory(root, "aurelius", detail_n=1, auto_commit_every=1)
        assert "first" in _history_text(reopened.model_history())

    def test_messages_keep_commit_reference(self, tmp_path: Path):
        memory = AureliusMemory(
            tmp_path / "memento",
            "aurelius",
            detail_n=1,
            auto_commit_every=1,
        )
        first = memory.remember(_moment("first", "one"))
        memory.remember(_moment("second", "two"))
        summaries = [message for message in memory.messages() if MementoRef.read(message)]
        assert len(summaries) == 1
        assert MementoRef.read(summaries[0]).commit_id == first.id

    def test_control_methods_preserve_frozen_history(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=0)
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
            AureliusMemory(tmp_path / "memento", "aurelius", **kwargs)


class TestAureliusGhost:
    @pytest.mark.asyncio
    async def test_articulate_then_reopen(self, tmp_path: Path):
        root = tmp_path / "persistent-memory"
        meta = AureliusMeta(
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
        meta = AureliusMeta(
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
        meta = AureliusMeta(soul_content="be exact", model=TestModel())
        ghost = meta.factory(_container(tmp_path))
        assert ghost.memory.root == tmp_path / "memento"
        assert ghost.memory.owner == "aurelius"

    def test_memento_control_surface_is_exposed_from_ghost_channel(self, tmp_path: Path):
        ghost = AureliusMeta(soul_content="be exact", model=TestModel()).factory(_container(tmp_path))
        assert ghost.channel().name() == "ghost"
        assert set(ghost.channel().main_state().own_commands()) >= {
            "memory_inspect",
            "memory_log",
            "memory_show",
            "memory_commit",
            "memory_reinterpret",
            "memory_fork",
            "memory_switch",
            "memory_recall",
            "memory_claims",
            "desktop_open",
            "desktop_pin",
            "desktop_update",
            "desktop_frame",
        }

    @pytest.mark.asyncio
    async def test_reflection_rewrites_note_without_touching_moment(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        view = memory.remember(_moment("我喜欢短回答", "明白。"))
        assert view is not None
        reflector = AureliusReflector(
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
        writer_meta = AureliusMeta(
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

        reader_meta = AureliusMeta(
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
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=0)
        memory.remember(_moment("legacy fact", "legacy answer"))
        view = memory.branch.commit("", kind="mechanical", by="aurelius")
        assert [candidate.id for candidate in memory.reflection_candidates()] == [view.id]
        reflector = AureliusReflector(
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
        ghost = AureliusMeta(soul_content="be exact", model=TestModel()).factory(container)
        assert ghost.memory.inspect()["detail_n"] == 3
        assert ghost.memory.inspect()["auto_commit_every"] == 1
        assert ghost.inspect_state()["reflection"]["enabled"] is False
        assert ghost.inspect_state()["knowledge"]["active_count"] == 0

    def test_relative_root_is_below_ghost_workspace(self, tmp_path: Path):
        meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(),
            memory_root="custom-memory",
        )
        ghost = meta.factory(_container(tmp_path))
        assert ghost.memory.root == tmp_path / "custom-memory"


class TestAureliusKnowledge:
    def test_projection_rebuilds_claims_with_stable_memento_evidence(self, tmp_path: Path):
        root = tmp_path / "memento"
        memory = AureliusMemory(root, "aurelius", auto_commit_every=1)
        memory.remember(
            _moment(
                "本轮测试代号是 AMBER-731，所属环境是 staging。",
                "测试代号是 ORBIT-004。",
            )
        )
        projection = MemoryProjection(memory)
        snapshot = projection.snapshot()
        active = {claim.key: claim for claim in snapshot.claims if claim.status == "active"}
        assert active["test.run.code"].value == "AMBER-731"
        assert active["test.run.environment"].value == "staging"
        assert active["test.run.code"].evidence_refs[0].commit_id is not None
        assert active["test.run.code"].evidence_refs[0].note_seq == 0
        assert active["test.run.code"].evidence_refs[0].quote == "本轮测试代号是 AMBER-731"
        rejected = [item for item in snapshot.candidates if item.value == "ORBIT-004"]
        assert len(rejected) == 1
        assert rejected[0].promoted is False
        assert rejected[0].evidence.origin == "assistant"
        claim_ids = {claim.id for claim in snapshot.claims}
        memory.close()

        reopened = AureliusMemory(root, "aurelius", auto_commit_every=1)
        rebuilt = MemoryProjection(reopened).snapshot()
        assert {claim.id for claim in rebuilt.claims} == claim_ids
        reopened.close()

    def test_recall_isolates_fields_and_verifier_rejects_orbit_substitution(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        memory.remember(_moment("本轮测试代号是 AMBER-731，所属环境是 staging。"))
        memory.remember(_moment("ORBIT-004 的校验词是“雪松”。"))
        projection = MemoryProjection(memory)
        packet = projection.recall("本轮测试代号和所属环境是什么？")
        assert packet.status == "ok"
        assert {claim.key for claim in packet.claims} == {"test.run.code", "test.run.environment"}
        assert {claim.value for claim in packet.claims} == {"AMBER-731", "staging"}
        assert "ORBIT-004" not in projection.render_packet(packet)
        assert projection.verify("AMBER-731，staging。", packet).accepted is True
        assert projection.verify("amber-731，STAGING。", packet).accepted is True
        rejected = projection.verify("ORBIT-004，staging。", packet)
        assert rejected.accepted is False

        production = AureliusMemory(tmp_path / "production", "aurelius", auto_commit_every=1)
        production.remember(_moment("本轮测试代号是 GOLD-101，所属环境是 production。"))
        production_projection = MemoryProjection(production)
        production_packet = production_projection.recall("本轮测试代号和所属环境是什么？")
        assert production_projection.verify("GOLD-101，production。", production_packet).accepted is True

    def test_reflection_and_logos_never_promote_active_claims(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        view = memory.remember(_moment("测试代号是 AMBER-731。", "测试代号是 ORBIT-004。"))
        assert view is not None
        memory.apply_reflection(view.id, "测试代号是 REFLECT-999。")
        snapshot = MemoryProjection(memory).snapshot()
        active_codes = [
            claim.value
            for claim in snapshot.claims
            if claim.key == "test.run.code" and claim.status == "active"
        ]
        assert active_codes == ["AMBER-731"]
        rejected = {item.value: item for item in snapshot.candidates if not item.promoted}
        assert rejected["ORBIT-004"].evidence.origin == "assistant"
        assert rejected["REFLECT-999"].evidence.origin == "reflection"

    def test_only_configured_tool_source_can_promote(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        memory.remember(
            Moment(
                percepts={
                    "trusted_tool": [Message.new().with_content("project.build.status = passed")],
                    "command": [Message.new().with_content("project.release.status = shipped")],
                }
            )
        )
        projection = MemoryProjection(memory)
        snapshot = projection.snapshot()
        assert [claim.key for claim in snapshot.claims] == ["project.build.status"]
        claim = snapshot.claims[0]
        assert claim.evidence_refs[0].origin == "trusted_tool"
        assert claim.confidence == 0.98
        assert projection.recall("project.build.status 是什么？").status == "ok"

    def test_correction_supersedes_old_value_and_unresolved_change_conflicts(self, tmp_path: Path):
        corrected = AureliusMemory(tmp_path / "corrected", "aurelius", auto_commit_every=1)
        corrected.remember(_moment("我住在杭州。"))
        corrected.remember(_moment("我之前说我住在杭州，现更正为苏州。"))
        projection = MemoryProjection(corrected)
        snapshot = projection.snapshot()
        cities = {claim.value: claim for claim in snapshot.claims if claim.key == "user.city"}
        assert cities["杭州"].status == "superseded"
        assert cities["苏州"].status == "active"
        assert cities["杭州"].id in cities["苏州"].supersedes
        assert projection.recall("我住在哪里？").claims[0].value == "苏州"

        conflicting = AureliusMemory(tmp_path / "conflicting", "aurelius", auto_commit_every=1)
        conflicting.remember(_moment("我住在杭州。"))
        conflicting.remember(_moment("我住在苏州。"))
        assert MemoryProjection(conflicting).recall("我住在哪里？").status == "conflict"
        conflicting.remember(_moment("更正：我住在杭州。"))
        resolved = MemoryProjection(conflicting).recall("我住在哪里？")
        assert resolved.status == "ok"
        assert resolved.claims[0].value == "杭州"

    def test_unknown_fact_returns_safe_packet(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius")
        packet = MemoryProjection(memory).recall("我的护照号是什么？")
        assert packet.status == "unknown"
        assert "没有找到" in packet.safe_answer()

    @pytest.mark.asyncio
    async def test_runtime_buffers_memory_answer_and_blocks_invalid_model_output(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        root = tmp_path / "memento"
        meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="ORBIT-004，staging。"),
            memory_root=root,
            auto_commit_every=1,
            reflection_enabled=False,
            desktop_enabled=False,
        )
        ghost = meta.factory(_container(workspace))
        source = _moment(
            "本轮测试代号是 AMBER-731，所属环境是 staging。",
            "测试代号是 ORBIT-004。",
        )
        ghost.on_articulate_exit(MockArticulator(source), source.logos, None)

        query = _moment("本轮测试代号和所属环境是什么？")
        async with ghost:
            answer = "".join([part async for part in ghost.articulate(MockArticulator(query))])
        assert answer == "记忆证据校验未通过，暂不回答该事实。"
        assert ghost.inspect_context()["recall"]["status"] == "ok"
        assert ghost.inspect_context()["answer_verification"]["accepted"] is False

        valid_meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="AMBER-731，staging。"),
            memory_root=root,
            auto_commit_every=1,
            reflection_enabled=False,
            desktop_enabled=False,
        )
        valid = valid_meta.factory(_container(workspace))
        async with valid:
            answer = "".join([part async for part in valid.articulate(MockArticulator(query))])
        assert answer == "AMBER-731，staging。"
        assert valid.inspect_context()["answer_verification"]["accepted"] is True


class TestAureliusDesktop:
    @pytest.mark.asyncio
    async def test_ground_instruction_and_frame_are_ephemeral_model_context(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "DESKTOP.md").write_text("回答仓库问题时说明当前 Pin。\n", encoding="utf-8")
        spec = workspace / "spec.md"
        spec.write_text("version one\n", encoding="utf-8")
        captured: dict[str, str] = {}

        async def stream(messages, info):
            captured["messages"] = _history_text(messages)
            captured["instructions"] = str(info.instructions)
            yield "grounded"

        meta = AureliusMeta(
            soul_content="be exact",
            model=FunctionModel(stream_function=stream),
            reflection_enabled=False,
            desktop_root=workspace,
        )
        ghost = meta.factory(_container(workspace))
        async with ghost:
            ground = ghost.desktop.primary
            assert ground is not None
            ground.pin("spec.md:1-1", "current spec")
            assert "version one" in await ghost.desktop.context()
            spec.write_text("version two\n", encoding="utf-8")
            assert "changed on disk" in await ghost.desktop.context()
            update = await ghost.desktop.update(ground.label, "spec.md:1-1")
            assert update.changed is True
            response = "".join(
                [part async for part in ghost.articulate(MockArticulator(_moment("当前规格是什么？")))]
            )
            assert response == "grounded"
            assert ghost.knowledge is not None
            assert ghost.knowledge.snapshot().claims == []
            with pytest.raises(PathOutsideRootError):
                ground.pin("../secret.md")

        assert "回答仓库问题时说明当前 Pin" in captured["instructions"]
        assert "version two" in captured["messages"]
        assert ghost.inspect_context()["ground_context_chars"] > 0
