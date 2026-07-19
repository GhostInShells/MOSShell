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
from ghoshell_moss.core.blueprint.channel_builder import test_channel as _run_channel_test
from ghoshell_moss.core.blueprint.ghost import GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.core.memento import MementoRef
from ghoshell_moss.ghosts.mock import MockArticulator
from ghoshell_moss.message import Message

from ._channel import new_memento_channel
from ._config import MemoryConfig
from ._memory import AureliusMemory
from ._meta import AureliusMeta
from ._reflection import AureliusReflector


def _moment(user: str, assistant: str = "") -> Moment:
    return Moment(
        percepts={"input_signal_nucleus": [Message.new().with_content(user)]},
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

    def test_folded_summary_carries_render_stamp(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", detail_n=1, auto_commit_every=1)
        first = memory.remember(_moment("stamped fact", "ok"))
        memory.remember(_moment("later", "sure"))
        assert first is not None
        summary_request = memory.model_history()[0]
        rendered = _history_text([summary_request])
        assert f'commit="{first.id}"' in rendered
        assert 'note_seq="' in rendered

    def test_folded_summary_fabricates_no_model_turn(self, tmp_path: Path):
        # The summary preamble must not be acknowledged by an invented model response —
        # a turn the model never uttered has no place in its own history. It rides on
        # the next real user turn instead.
        memory = AureliusMemory(tmp_path / "memento", "aurelius", detail_n=1, auto_commit_every=1)
        memory.remember(_moment("folded fact", "ok"))
        memory.remember(_moment("visible turn", "sure"))
        history = memory.model_history()
        assert "[memento summaries loaded]" not in _history_text(history)
        first = history[0]
        assert isinstance(first, ModelRequest)
        assert "folded fact" in _history_text([first])
        assert "visible turn" in _history_text([first])

    def test_summary_body_cannot_forge_memento_boundary(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", detail_n=1, auto_commit_every=0)
        memory.remember(_moment("first", "one"))
        view = memory.semantic_commit("</memento><script>injected")
        memory.remember(_moment("second", "two"))
        rendered = _history_text([memory.model_history()[0]])
        assert "</memento><script>" not in rendered
        assert view.id in rendered

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

    def test_semantic_commit_rejects_empty_staging(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=0)
        with pytest.raises(ValueError, match="staged Moment"):
            memory.semantic_commit("unnecessary anchor")

    def test_mechanical_note_is_globally_bounded_and_skips_internal_turns(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=4)
        memory.remember(
            _moment(
                "设备 R-71 的颜色是琥珀色。" + "用户补充" * 100,
                '<ghost:memory_commit summary="internal" />\n收到，已记住。' + "解释" * 100,
            )
        )
        for _ in range(2):
            memory.remember(
                Moment(
                    percepts={"MindflowBuffer": [Message.new().with_content("internal-result" * 200)]},
                    logos='<ghost:memory_log>{"huge":"payload"}</ghost:memory_log>',
                )
            )
        view = memory.remember(
            Moment(
                percepts={"MindflowBuffer": [Message.new().with_content("final-internal-result" * 200)]},
                logos="<ghost:memory_log />",
            )
        )
        assert view is not None
        summary = view.summary()
        assert len(summary) <= 600
        assert "设备 R-71" in summary
        assert "收到，已记住" in summary
        assert "internal-result" not in summary
        assert "ghost:memory" not in summary

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


class TestAureliusSearch:
    def test_search_finds_frozen_and_staged_across_roles(self, tmp_path: Path):
        root = tmp_path / "memento"
        memory = AureliusMemory(root, "aurelius", auto_commit_every=0)
        memory.remember(_moment("ORBIT-004 的校验词是雪松。", "收到。"))
        frozen = memory.semantic_commit("ORBIT-004 校验词锚点")
        assert frozen is not None
        memory.remember(_moment("顺便记一下松木不是雪松。"))  # stays in staging

        input_hits = memory.search("ORBIT-004")
        assert input_hits
        assert input_hits[0].commit_id == frozen.id
        assert input_hits[0].frozen is True
        assert "雪松" in input_hits[0].snippet

        logos_hits = memory.search("收到")
        assert any(hit.role == "logos" for hit in logos_hits)

        staged_hits = memory.search("松木")
        assert staged_hits
        assert staged_hits[0].frozen is False
        assert staged_hits[0].commit_id is None

    def test_search_is_case_insensitive_and_bounded_and_survives_reopen(self, tmp_path: Path):
        root = tmp_path / "memento"
        memory = AureliusMemory(root, "aurelius", auto_commit_every=1)
        for index in range(5):
            memory.remember(_moment(f"passphrase token-{index}", "ok"))
        assert len(memory.search("PASSPHRASE", limit=3)) == 3
        with pytest.raises(ValueError, match="keyword"):
            memory.search("   ")
        memory.close()

        reopened = AureliusMemory(root, "aurelius", auto_commit_every=1)
        assert reopened.search("token-2")


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

    @pytest.mark.asyncio
    async def test_fact_question_streams_without_verifier_gate(self, tmp_path: Path):
        # The old regex verifier would buffer and reject this; now the model answers freely
        # and is expected to self-check with memory_search per the discipline instruction.
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="AMBER-731，staging。"),
            memory_root=tmp_path / "memento",
            auto_commit_every=1,
            reflection_enabled=False,
            curation_enabled=False,
            desktop_enabled=False,
        )
        ghost = meta.factory(_container(workspace))
        source = _moment("本轮测试代号是 AMBER-731，所属环境是 staging。", "记下了。")
        ghost.on_articulate_exit(MockArticulator(source), source.logos, None)
        async with ghost:
            answer = "".join([part async for part in ghost.articulate(MockArticulator(_moment("代号是什么？")))])
        assert answer == "AMBER-731，staging。"

    def test_failed_articulation_is_witnessed_as_failed(self, tmp_path: Path):
        # A failed frame is a trajectory event ("saw X, tried, errored"), not something
        # to erase. It must persist, tagged "failed" so it never reads as a finished turn.
        meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(),
            memory_root=tmp_path / "memento",
        )
        ghost = meta.factory(_container(tmp_path / "workspace"))
        moment = _moment("witness-this-failure", "partial")
        ghost.on_articulate_exit(
            MockArticulator(moment),
            "partial",
            RuntimeError("model failed"),
        )
        staging = ghost.memory.branch.staging()
        assert len(staging) == 1
        assert "failed" in staging[0].threads
        assert ghost.inspect_context()["memory_write"] == "staged_failed"

    def test_default_root_is_ghost_workspace(self, tmp_path: Path):
        meta = AureliusMeta(soul_content="be exact", model=TestModel())
        ghost = meta.factory(_container(tmp_path))
        assert ghost.memory.root == tmp_path / "memento"
        assert ghost.memory.owner == "aurelius"

    def test_memory_config_is_persisted_policy(self, tmp_path: Path):
        container = _container(tmp_path / "workspace")
        store = YamlConfigStore(LocalStorage(tmp_path / "configs"))
        store.save(MemoryConfig(detail_n=3, auto_commit_every=1, reflection_enabled=False))
        container.set(ConfigStore, store)
        ghost = AureliusMeta(soul_content="be exact", model=TestModel()).factory(container)
        assert ghost.memory.inspect()["detail_n"] == 3
        assert ghost.memory.inspect()["auto_commit_every"] == 1
        assert ghost.inspect_state()["reflection"]["enabled"] is False
        assert "curation" in ghost.inspect_state()

    def test_relative_root_is_below_ghost_workspace(self, tmp_path: Path):
        meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(),
            memory_root="custom-memory",
        )
        ghost = meta.factory(_container(tmp_path))
        assert ghost.memory.root == tmp_path / "custom-memory"

    def test_configured_model_wires_max_output_tokens(self, tmp_path: Path):
        # Regression: a configured (non-injected) model must carry the contract's
        # max_output_tokens as ModelSettings(max_tokens=...). Without it pydantic-ai
        # falls back to the provider default and rejects longer replies with
        # "token limit (provider default) exceeded" even when the prompt fits.
        from ghoshell_moss.contracts.llms import (
            LLMConfig,
            ModelConfig,
            Provider,
            ServiceConfig,
        )

        config = LLMConfig(
            default=Provider(
                service=ServiceConfig(
                    name="anthropic",
                    base_url="https://example.invalid",
                    api_key="sk-test-not-a-real-key",
                    protocol="anthropic",
                ),
                default=ModelConfig(model="claude-test", max_output_tokens=8192),
            )
        )
        container = _container(tmp_path / "workspace")
        store = YamlConfigStore(LocalStorage(tmp_path / "configs"))
        store.save(config)
        container.set(ConfigStore, store)

        captured: dict = {}
        meta = AureliusMeta(
            soul_content="be exact",
            on_agent_build=lambda agent: captured.update(settings=agent.model_settings),
        )
        meta.factory(container)
        assert captured["settings"]["max_tokens"] == 8192


class TestAureliusControlSurface:
    def test_retrieval_visible_admin_hidden_and_discipline_injected(self, tmp_path: Path):
        ghost = AureliusMeta(soul_content="be exact", model=TestModel()).factory(_container(tmp_path))
        assert ghost.channel().name() == "ghost"
        commands = ghost.channel().main_state().own_commands()
        assert set(commands) >= {
            "memory_search",
            "memory_log",
            "memory_show",
            "memory_inspect",
            "memory_staging",
            "memory_commit",
            "memory_reinterpret",
            "memory_fork",
            "memory_switch",
            "memory_reflect",
            "memory_curate",
            "memory_branches",
            "desktop_open",
            "desktop_pin",
            "desktop_update",
            "desktop_frame",
        }
        for name in ("memory_search", "memory_log", "memory_show"):
            assert commands[name].meta().visible is True
            # Read-to-answer commands must trigger the next Re-Act cycle, or a
            # search-then-answer turn settles with the hits unobserved and the ghost
            # goes silent on canonical-key questions.
            assert commands[name].meta().always_observe is True
        for name in ("memory_commit", "memory_fork", "memory_reinterpret", "memory_curate", "memory_reflect"):
            assert commands[name].meta().visible is False
            # Write/admin commands must NOT force observation — they are actions, not answers.
            assert commands[name].meta().always_observe is False
        assert commands["desktop_open"].meta().visible is True
        assert "memory_search" in ghost.system_prompt()
        assert "没有找到记忆证据" in ghost.system_prompt()

    @pytest.mark.asyncio
    async def test_memory_search_is_executable_from_ctml(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        memory.remember(_moment("设备 R-71 的颜色是琥珀色。", "收到。"))
        channel = new_memento_channel(memory)
        tasks = await _run_channel_test(channel, ctml='<ghost:memory_search keyword="R-71" />')
        assert len(tasks) == 1
        result = await tasks[0]
        assert result
        assert result[0]["snippet"]
        assert result[0]["frozen"] is True

    @pytest.mark.asyncio
    async def test_disabled_bypass_commands_report_truthfully(self, tmp_path: Path):
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        channel = new_memento_channel(memory, on_reflect=None, on_curate=None)
        for command in ("memory_reflect", "memory_curate"):
            tasks = await _run_channel_test(channel, ctml=f"<ghost:{command} />")
            assert "disabled" in await tasks[0]

    @pytest.mark.asyncio
    async def test_memory_reflect_schedules_from_ctml_worker_thread(self, tmp_path: Path):
        # Regression: CTML commands execute under asyncio.to_thread, where the reflect
        # handler used to call create_task with no running loop and crash. The ghost
        # must marshal the schedule back onto its own loop instead.
        meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="answer"),
            reflection_model=TestModel(custom_output_text="反思完成。"),
            memory_root=tmp_path / "memento",
            auto_commit_every=1,
            curation_enabled=False,
            desktop_enabled=False,
        )
        ghost = meta.factory(_container(tmp_path / "workspace"))
        source = _moment("记住这个。", "好的。")
        async with ghost:
            ghost.on_articulate_exit(MockArticulator(source), source.logos, None)
            for _ in range(20):
                if not ghost.memory.reflection_candidates():
                    break
                await asyncio.sleep(0.01)
            tasks = await _run_channel_test(ghost.channel(), ctml="<ghost:memory_reflect />")
            assert "scheduled" in await tasks[0]

    @pytest.mark.asyncio
    async def test_concurrent_remember_and_commit_do_not_corrupt(self, tmp_path: Path):
        # remember() runs on the event loop while CTML semantic_commit() runs on a worker
        # thread. Without serialization these two writers race on staging.jsonl. The lock
        # must keep every Moment accounted for exactly once across staging + commits.
        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=0)

        async def writer() -> None:
            for index in range(20):
                await asyncio.to_thread(memory.remember, _moment(f"fact-{index}", "ok"))

        async def committer() -> None:
            for _ in range(10):
                await asyncio.sleep(0)
                try:
                    await asyncio.to_thread(memory.semantic_commit, "anchor")
                except ValueError:
                    pass  # empty staging is a legal race outcome, not corruption

        await asyncio.gather(writer(), committer())
        memory.semantic_commit("final anchor") if memory.branch.staging() else None
        seen = {record.id for record in memory.branch.staging()}
        for view in memory.branch.all_commits():
            for record in memory.branch.commit_records(view.id):
                assert record.id not in seen, "a Moment appeared in two places — write race"
                seen.add(record.id)
        assert len(seen) == 20


class TestAureliusReflection:
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
            curation_enabled=False,
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
            curation_enabled=False,
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


class TestAureliusCuration:
    @pytest.mark.asyncio
    async def test_curator_writes_notes_from_frozen_trajectory(self, tmp_path: Path):
        from ._curation import AureliusCurator

        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        memory.remember(_moment("ORBIT-004 的校验词是雪松。", "收到。"))
        notes_path = tmp_path / "ground" / "facts.md"
        curator = AureliusCurator(
            Agent(model=TestModel(custom_output_text="校验词：雪松 (cmt_x)")),
            notes_path,
            max_source_chars=4000,
            max_notes_chars=2000,
        )
        written = await curator.curate(memory, Container())
        assert written == notes_path
        body = notes_path.read_text(encoding="utf-8")
        assert "雪松" in body
        assert "memory_show" in body  # provenance banner points back at frozen evidence

    @pytest.mark.asyncio
    async def test_curator_no_op_on_empty_trajectory(self, tmp_path: Path):
        from ._curation import AureliusCurator

        memory = AureliusMemory(tmp_path / "memento", "aurelius", auto_commit_every=1)
        curator = AureliusCurator(
            Agent(model=TestModel(custom_output_text="unused")),
            tmp_path / "ground" / "facts.md",
            max_source_chars=4000,
            max_notes_chars=2000,
        )
        assert await curator.curate(memory, Container()) is None

    @pytest.mark.asyncio
    async def test_curation_notes_are_pinned_into_ground(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        meta = AureliusMeta(
            soul_content="be exact",
            model=TestModel(custom_output_text="answer"),
            curation_model=TestModel(custom_output_text="- 稳定事实：X (cmt_1)"),
            memory_root=tmp_path / "memento",
            auto_commit_every=1,
            reflection_enabled=False,
            desktop_root=workspace,
        )
        ghost = meta.factory(_container(workspace))
        source = _moment("记住这个事实。", "好的。")
        ghost.on_articulate_exit(MockArticulator(source), source.logos, None)
        async with ghost:
            for _ in range(30):
                if (workspace / "facts.md").exists():
                    break
                await asyncio.sleep(0.01)
            assert (workspace / "facts.md").exists()
            context = await ghost.desktop.context()
        assert "稳定事实" in context


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
            curation_enabled=False,
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
            with pytest.raises(PathOutsideRootError):
                ground.pin("../secret.md")

        assert "回答仓库问题时说明当前 Pin" in captured["instructions"]
        assert "version two" in captured["messages"]
        assert ghost.inspect_context()["ground_context_chars"] > 0


class TestAureliusContextBudget:
    def _ghost(self, tmp_path: Path, *, model=None, budget: int = 0, **meta_kwargs):
        meta = AureliusMeta(
            soul_content="be exact",
            model=model or TestModel(custom_output_text="ok"),
            memory_root=tmp_path / "memento",
            reflection_enabled=False,
            curation_enabled=False,
            desktop_enabled=False,
            **meta_kwargs,
        )
        ghost = meta.factory(_container(tmp_path / "workspace"))
        if budget:
            ghost._context_budget_enabled = True
            ghost._context_input_budget = budget
        return ghost

    def test_estimator_charges_text_and_images_conservatively(self):
        from pydantic_ai import ImageUrl
        from pydantic_ai.messages import ModelRequest as BudgetRequest
        from pydantic_ai.messages import UserPromptPart as BudgetPrompt

        from ._budget import IMAGE_NOMINAL_TOKENS, estimate_history_tokens, estimate_text_tokens

        assert estimate_text_tokens("") == 0
        # CJK-leaning divisor: 100 chars must count as >= 40 tokens, never near 100/4.
        assert estimate_text_tokens("霜" * 100) >= 40
        image_like = BudgetRequest(parts=[BudgetPrompt(content=[ImageUrl(url="data:image/png;base64,AAAA")])])
        # Image billed by nominal cost, not payload length.
        assert estimate_history_tokens([image_like]) < IMAGE_NOMINAL_TOKENS + 100

    def test_overflow_classifier_matches_input_not_output_errors(self):
        from ._budget import is_context_overflow

        assert is_context_overflow(RuntimeError("prompt is too long: 210000 tokens > maximum context"))
        assert is_context_overflow(RuntimeError("Error code 400: context_length_exceeded"))
        # Output-side max_tokens issue and attention aborts must NOT be treated as overflow.
        assert not is_context_overflow(
            RuntimeError("Model token limit (provider default) exceeded before any response")
        )
        assert not is_context_overflow(RuntimeError("Attention is already aborted"))

    def test_budgeted_history_shrinks_window_until_it_fits(self, tmp_path: Path):
        ghost = self._ghost(tmp_path, budget=600)
        for index in range(8):
            ghost.memory.remember(_moment(f"长事实记录 {index}：" + "记" * 300, "回答" * 100))
        history, report = ghost._budgeted_history()
        assert report["shrunk"] is True
        assert report["estimated_tokens"] <= max(600 - ghost._context_fixed_overhead, 1) or (
            report["detail_n"] == ghost._context_min_detail_n and report["summary_m"] == 0
        )
        assert history  # never empties completely
        # Persisted policy is untouched; only this render shrank.
        assert ghost.memory.detail_n == 12

    def test_budget_disabled_renders_full_window(self, tmp_path: Path):
        ghost = self._ghost(tmp_path)  # injected TestModel → budget 0 → disabled
        ghost.memory.remember(_moment("一条记录", "好的"))
        history, report = ghost._budgeted_history()
        assert report["shrunk"] is False
        assert report["estimated_tokens"] is None
        assert history

    @pytest.mark.asyncio
    async def test_overflow_retries_with_halved_window_then_succeeds(self, tmp_path: Path):
        calls = {"count": 0}

        async def flaky(messages, info):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("input is too long for maximum context length")
            yield "recovered"

        ghost = self._ghost(tmp_path, model=FunctionModel(stream_function=flaky))
        ghost.memory.remember(_moment("历史一", "答一"))
        async with ghost:
            out = "".join(
                [part async for part in ghost.articulate(MockArticulator(_moment("问题")))]
            )
        assert out == "recovered"
        assert calls["count"] == 2
        assert ghost.inspect_context()["context_budget"]["overflow_retry"] is True

    @pytest.mark.asyncio
    async def test_non_overflow_error_propagates_without_retry(self, tmp_path: Path):
        calls = {"count": 0}

        async def broken(messages, info):
            calls["count"] += 1
            raise RuntimeError("Attention is already aborted")
            yield  # pragma: no cover

        ghost = self._ghost(tmp_path, model=FunctionModel(stream_function=broken))
        async with ghost:
            with pytest.raises(Exception, match="aborted"):
                async for _ in ghost.articulate(MockArticulator(_moment("问题"))):
                    pass
        assert calls["count"] == 1
