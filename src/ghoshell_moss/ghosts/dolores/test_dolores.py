"""Dolores Ghost 原型测试 — 骨架 + stubs 同步阶段.

覆盖:
- GhostMeta/Ghost ABC 契约
- stubs 同步三种路径: init / override / noop
- 构造无副作用 (写盘收敛到 __aenter__)
- session.output 提示在 init/override 时发出, noop 不发声
"""

import asyncio
from pathlib import Path

import pytest
import yaml
from ghoshell_container import Container

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.session.mock_session import MockSession


def _dolores_meta(**kwargs):
    from ._meta import DoloresMeta

    return DoloresMeta(**kwargs)


def _dolores(meta=None, *, home=None, session=None, matrix=None, shell=None, base_instruction=None):
    from ._runtime import Dolores

    return Dolores(
        meta=meta or _dolores_meta(),
        home=home,
        session=session,
        matrix=matrix,
        shell=shell,
        base_instruction=base_instruction,
    )


class TestDoloresMeta:
    def test_defaults(self):
        from .nucleus import DoloresEgoNucleusMeta

        meta = _dolores_meta()
        assert meta.name() == "dolores"
        assert meta.prototype() == "Dolores"
        # Dolores 默认挂载 ego 自醒 nucleus (self-wake 通道).
        metas = meta.nuclei_metas()
        assert len(metas) == 1
        assert isinstance(metas[0], DoloresEgoNucleusMeta)

    def test_nuclei_metas_fully_replaced_when_passed(self):
        """显式传 nuclei_metas 时完全替换默认, 不叠加."""
        from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
        from .nucleus import DoloresEgoNucleusMeta

        custom = DoloresEgoNucleusMeta()
        meta = _dolores_meta(nuclei_metas=[custom])
        assert meta.nuclei_metas() == [custom]

    def test_is_ghost_meta_abc(self):
        assert isinstance(_dolores_meta(), GhostMeta)

    def test_version_constant(self):
        assert isinstance(_dolores_meta().VERSION, str)
        assert _dolores_meta().VERSION

    def test_stubs_dir_contains_ground(self):
        stubs = _dolores_meta().stubs_dir()
        assert stubs.is_dir()
        assert (stubs / "GROUND.md").exists()

    def test_factory_returns_dolores(self):
        from ._runtime import Dolores

        meta = _dolores_meta()
        ghost = meta.factory(Container())
        assert isinstance(ghost, Dolores)
        assert isinstance(ghost, Ghost)
        assert ghost.meta is meta
        # 空 container 无 GhostWorkspace/Session/Matrix → 均为 None, 无副作用.
        assert ghost._home is None
        assert ghost._session is None
        assert ghost._matrix is None
        assert ghost._dsh_launcher is None

    def test_dsh_stubs_and_plugin_stub(self):
        dsh_stubs = _dolores_meta().dsh_stubs_dir()
        assert dsh_stubs.is_dir()
        assert (dsh_stubs / "profiles/web/package.json").exists()
        # plugin 源在独立 stub, 创建时复制进 dsh profile.
        assert _dolores_meta().dsh_plugin_stub().is_file()


class TestStubsSync:
    def test_construct_has_no_side_effect(self, tmp_path: Path):
        _dolores(home=tmp_path)
        assert not (tmp_path / ".dolores.yml").exists()
        assert not (tmp_path / "GROUND.md").exists()

    def test_init_creates_home_and_marker(self, tmp_path: Path):
        session = MockSession()
        ghost = _dolores(home=tmp_path, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        assert (tmp_path / "GROUND.md").exists()
        config = yaml.safe_load((tmp_path / ".dolores.yml").read_text())
        assert config["version"] == ghost._meta.VERSION
        # dirs 物化 + dsh_stubs 同步.
        assert (tmp_path / ".dsh").is_dir()
        assert (tmp_path / "skills").is_dir()
        assert (tmp_path / ".dsh/profiles/web/plugin.ts").exists()
        assert len(session.outputs) == 1
        assert session.outputs[0].role == "system"

    def test_override_on_version_mismatch(self, tmp_path: Path):
        (tmp_path / ".dolores.yml").write_text("version: dev_0\n")
        (tmp_path / "stale.txt").write_text("dynamic data")
        session = MockSession()
        ghost = _dolores(home=tmp_path, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        assert (tmp_path / "GROUND.md").exists()
        config = yaml.safe_load((tmp_path / ".dolores.yml").read_text())
        assert config["version"] == ghost._meta.VERSION
        # 动态数据文件不被 stubs 覆盖触碰.
        assert (tmp_path / "stale.txt").read_text() == "dynamic data"
        assert len(session.outputs) == 1
        assert "override" in session.outputs[0].messages_string()

    def test_noop_when_version_matches(self, tmp_path: Path):
        (tmp_path / ".dolores.yml").write_text(f"version: {_dolores_meta().VERSION}\n")
        (tmp_path / "GROUND.md").write_text("already here")
        session = MockSession()
        ghost = _dolores(home=tmp_path, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        # 版本一致 → 不覆盖, 不发声.
        assert (tmp_path / "GROUND.md").read_text() == "already here"
        assert session.outputs == []

    def test_home_none_is_noop(self):
        session = MockSession()
        ghost = _dolores(home=None, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())
        assert session.outputs == []


class TestDolores:
    def test_is_ghost_abc(self):
        assert isinstance(_dolores(), Ghost)

    def test_lifecycle_no_error(self):
        ghost = _dolores()

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

    def test_dsh_launcher_raises_before_start(self):
        ghost = _dolores()
        with pytest.raises(RuntimeError):
            ghost.dsh_launcher

    def test_no_dsh_launch_without_matrix(self):
        ghost = _dolores(home=None, session=MockSession())

        async def run():
            async with ghost:
                pass

        asyncio.run(run())
        # 无 matrix → 不拉起 dsh, launcher 保持 None.
        assert ghost._dsh_launcher is None
        with pytest.raises(RuntimeError):
            ghost.dsh_launcher


class TestDoloresArticulate:
    def test_yields_placeholder_logos(self):
        """无 ego (matrix=None) → think 只产出占位 logos, 不 crash."""
        ghost = _dolores()

        async def collect():
            return [delta async for delta in ghost.think(None)]

        assert asyncio.run(collect()) == [""]


class TestDoloresInstruction:
    def test_system_prompt_derives_two_meta_segments(self):
        """system_prompt = 原型元信息 + 身份描述, 从结构化 meta 派生, 无 baseline 时不含 baseline."""
        meta = _dolores_meta()
        text = _dolores(meta=meta).system_prompt()
        assert "prototype: Dolores" in text
        assert f"version: {meta.VERSION}" in text
        assert "name: dolores" in text
        assert "description:" in text

    def test_system_prompt_prepends_base_instruction(self):
        """baseline (MossSystemPrompter.base_instruction) 在两段之前."""
        text = _dolores(base_instruction="BASELINE").system_prompt()
        assert text.startswith("BASELINE")
        assert "prototype: Dolores" in text
        assert "name: dolores" in text

    @pytest.mark.asyncio
    async def test_ground_instruction_none_without_home(self):
        """无 home → 无 root ground, ground_instruction 返回 None."""
        ghost = _dolores()
        async with ghost:
            assert await ghost.ground_instruction() is None

    @pytest.mark.asyncio
    async def test_ground_instruction_renders_held_ground(self, tmp_path: Path):
        """home 存在 → stubs 同步落 GROUND.md, __aenter__ 打开 root ground, 渲染非空."""
        ghost = _dolores(home=tmp_path)
        async with ghost:
            text = await ghost.ground_instruction()
            assert text is not None
            assert text.strip() != ""


class TestDoloresMemories:
    """Dolores.memories — ground 渲染为第一条存在主义记忆."""

    @pytest.mark.asyncio
    async def test_memories_empty_without_home(self):
        """无 home → 无 ground 渲染, memories 返回空."""
        ghost = _dolores()
        async with ghost:
            assert ghost.memories() == []

    @pytest.mark.asyncio
    async def test_memories_returns_ground_as_first(self, tmp_path: Path):
        """home 存在 → ground 渲染文本包成 ground tag 的记忆, 是唯一一条."""
        ghost = _dolores(home=tmp_path)
        async with ghost:
            memories = ghost.memories()
            assert len(memories) == 1
            text = memories[0].to_content_string()
            assert "<ground>" in text and "</ground>" in text


class TestDoloresEgoNucleus:
    """DoloresEgoNucleus — BACKGROUND 挑战包 (发完丢), attended 加工成 INFO 运行包."""

    @pytest.mark.asyncio
    async def test_signal_produces_background_empty_impulse(self):
        from ghoshell_moss.core.blueprint.mindflow import Priority

        from .nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(new_dolores_ego_signal())

        assert len(impulses) == 1
        imp = impulses[0]
        assert imp.priority == Priority.BACKGROUND
        assert imp.messages == []
        # 默认 mode (空) = 正常仲裁, 非 silent buffer.
        assert imp.mode == ""

    @pytest.mark.asyncio
    async def test_attended_rewrites_to_info(self):
        from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority

        from .nucleus import DoloresEgoNucleus

        nucleus = DoloresEgoNucleus()
        async with nucleus:
            challenge = Impulse(source="dolores_ego_nucleus", priority=Priority.BACKGROUND)
            rewritten = nucleus.attended(challenge)

        assert rewritten is not None
        assert rewritten.priority == Priority.INFO
        assert rewritten.messages == []

    @pytest.mark.asyncio
    async def test_ignores_foreign_signal(self):
        from ghoshell_moss.core.blueprint.mindflow import Signal

        from .nucleus import DoloresEgoNucleus

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(Signal(name="some/other"))

        assert impulses == []

    @pytest.mark.asyncio
    async def test_meta_factory_builds_nucleus(self):
        from ghoshell_container import Container

        from .nucleus import DoloresEgoNucleus, DoloresEgoNucleusMeta

        nucleus = DoloresEgoNucleusMeta().factory(Container())
        assert isinstance(nucleus, DoloresEgoNucleus)
        assert nucleus.name() == "dolores_ego_nucleus"


class TestDoloresEgoSelfWake:
    """DoloresEgo 的 self-wake gate — articulate flag 决定 turn/start 是否自醒."""

    def _ego(self):
        from ._ego import DoloresEgo, DoloresEgoContext

        # gate 路径不触 dsh / 不读 ghost 反参 — ctx/launcher 传最小 dummy 即可.
        return DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                instruction="i",
            ),
        )

    @pytest.mark.asyncio
    async def test_turn_start_self_wakes_when_idle(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        await ego._on_session_activity(None)  # type: ignore[arg-type]  # gate 不读 event

        assert len(emitted) == 1
        assert emitted[0].name == "dolores/ego"

    @pytest.mark.asyncio
    async def test_turn_start_suppressed_when_articulating(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        ego.articulating = True
        await ego._on_session_activity(None)  # type: ignore[arg-type]

        assert emitted == []


# ── DoloresRun — thinking 交易 run 对象 (public + 可测) ─────────────


class FakeRunSession:
    """DoloresRun 的事件源 fake — on_session_event 注册/解绑 + 手动 emit."""

    def __init__(self):
        self.handlers: list = []

    def on_session_event(self, event_type, callback):
        self.handlers.append((event_type, callback))

        def dispose():
            self.handlers.remove((event_type, callback))

        return dispose

    async def emit(self, event):
        for _, cb in list(self.handlers):
            await cb(event)


class FakeRunEgo:
    """DoloresRun 的 ego fake — Duck-typed (session/articulating/enter_thinking/exit_thinking)."""

    def __init__(self, session):
        self.session = session
        self.articulating = False
        self.enter_calls = 0
        self.exit_calls = 0
        self.exit_yielded_values: list[bool] = []
        self.enter_error: Exception | None = None

    async def enter_thinking(self, thinking):
        self.enter_calls += 1
        if self.enter_error is not None:
            raise self.enter_error

    async def exit_thinking(self, *, yielded=False):
        self.exit_calls += 1
        self.exit_yielded_values.append(yielded)


class FakeRunThinking:
    def __init__(self):
        self.abort_reasons: list = []

    def abort(self, reason):
        self.abort_reasons.append(reason)


class TestDoloresRun:
    """DoloresRun 生命周期 + 事件消费 — public 类, 轻量 fake 即可验证."""

    def _run(self, session=None, ego=None, thinking=None):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent  # noqa: F401

        from ._run import DoloresRun

        session = session or FakeRunSession()
        return DoloresRun(
            ego=ego or FakeRunEgo(session),
            thinking=thinking or FakeRunThinking(),
        )

    @pytest.mark.asyncio
    async def test_aenter_binds_listener_and_aexit_cleans_up(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        run = self._run(session=session, ego=ego)
        async with run:
            assert ego.articulating is True
            assert len(session.handlers) == 1  # catch-all 监听已绑
            await asyncio.sleep(0)  # 让出 loop, enter task 跑
            assert ego.enter_calls == 1
        assert ego.articulating is False
        assert ego.exit_calls == 1
        assert ego.exit_yielded_values == [False]  # 非 yield 收线 → yielded=False
        assert len(session.handlers) == 0  # 解绑

    @pytest.mark.asyncio
    async def test_events_consumes_stream_and_breaks_on_turn_end(self):
        """enter 成功不塞毒丸 — 事件流持续到消费方在 turn/end break 收线."""
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, SessionEventMeta

        session = FakeRunSession()
        ego = FakeRunEgo(session)
        run = self._run(session=session, ego=ego)
        collected = []
        async with run:
            await session.emit(SessionEvent(meta=SessionEventMeta(type="turn/start", seq=1), data={"turn": 1}))
            await session.emit(SessionEvent(meta=SessionEventMeta(type="step/start", seq=2), data={"turn": 1, "step": 1}))
            await session.emit(SessionEvent(meta=SessionEventMeta(type="turn/end", seq=3), data={"turn": 1, "reason": "completed"}))
            async for event in run.events():  # 消费方在 turn/end break (与 Dolores.think 同构)
                collected.append(event.meta.type)
                if event.meta.type == "turn/end":
                    break
        assert collected == ["turn/start", "step/start", "turn/end"]

    @pytest.mark.asyncio
    async def test_yield_break_forwards_yielded_flag(self):
        """消费方认出 wait_next_moment 并 break → run.yielded 置位 → exit_thinking(yielded=True).

        plugin 侧据 yielded=True 绝不 cancel, 不依赖 dsh 侧 pendingYield 的时序竞态.
        """
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, SessionEventMeta
        from ghoshell_moss.ghosts.dolores._runtime import _is_yield_tool_call

        session = FakeRunSession()
        ego = FakeRunEgo(session)
        run = self._run(session=session, ego=ego)
        async with run:
            await session.emit(SessionEvent(
                meta=SessionEventMeta(type="tool/call", seq=1), data={"name": "wait_next_moment"}))
            async for event in run.events():
                if _is_yield_tool_call(event):
                    run.yielded = True  # 与 Dolores.think 的 break 分支同构
                    break
        assert ego.exit_yielded_values == [True]

    @pytest.mark.asyncio
    async def test_enter_error_propagates_and_aborts(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        ego.enter_error = RuntimeError("enter boom")
        run = self._run(session=session, ego=ego, thinking=thinking)
        with pytest.raises(RuntimeError, match="enter boom"):
            async with run:
                async for _ in run.events():
                    pass
        assert thinking.abort_reasons  # enter 异常 → thinking.abort

    @pytest.mark.asyncio
    async def test_consumer_exception_aborts_thinking(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        with pytest.raises(RuntimeError, match="consumer boom"):
            async with run:
                raise RuntimeError("consumer boom")
        assert thinking.abort_reasons


class TestYieldToolCall:
    """wait_next_moment (yield tool) 识别 — turn 边界信号, 触发 thinking exit."""

    def _event(self, event_type: str, data: dict):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, SessionEventMeta

        return SessionEvent(meta=SessionEventMeta(type=event_type, seq=1), data=data)

    def test_recognizes_yield_tool_call(self):
        from ._runtime import _is_yield_tool_call

        event = self._event("tool/call", {"name": "wait_next_moment"})
        assert _is_yield_tool_call(event) is True

    def test_ignores_other_tool_call(self):
        from ._runtime import _is_yield_tool_call

        event = self._event("tool/call", {"name": "moss_observe"})
        assert _is_yield_tool_call(event) is False

    def test_ignores_non_tool_call(self):
        from ._runtime import _is_yield_tool_call

        event = self._event("assistant/chunk", {})
        assert _is_yield_tool_call(event) is False


class TestObserveToolCall:
    """observe (主动观测) 识别 — interleaved thinking 的主动观测点, 内联返回 moment."""

    def _event(self, event_type: str, data: dict):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, SessionEventMeta

        return SessionEvent(meta=SessionEventMeta(type=event_type, seq=1), data=data)

    def test_recognizes_observe_tool_call(self):
        from ._runtime import _observe_tool_call

        event = self._event("tool/call", {"name": "observe", "callId": "c1"})
        call = _observe_tool_call(event)
        assert call is not None
        assert call.callId == "c1"

    def test_ignores_yield_tool_call(self):
        from ._runtime import _observe_tool_call

        event = self._event("tool/call", {"name": "wait_next_moment"})
        assert _observe_tool_call(event) is None

    def test_ignores_non_tool_call(self):
        from ._runtime import _observe_tool_call

        event = self._event("assistant/chunk", {})
        assert _observe_tool_call(event) is None


class TestDoloresMomentPayload:
    """DoloresEgo 的 moment 映射 — context (inject) + inputs (steer) 两条 message."""

    def _ego(self):
        from ._ego import DoloresEgo, DoloresEgoContext

        return DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                instruction="i",
            ),
        )

    def test_moment_payload_splits_context_and_inputs(self):
        from ghoshell_moss.core.blueprint.moment import Echoes, Moment
        from ghoshell_moss.message import Message

        moment = Moment(
            previous=Echoes(messages=[Message.new().with_content("echo")]),
            percepts={"test": [Message.new().with_content("percept")]},
            hint="hint text",
            command_logos="cmd!",
        )
        payload = self._ego()._moment_payload(moment)
        assert payload["moment_id"] == moment.id
        context_text = "".join(c["text"] for c in payload["context"] if c.get("type") == "text")
        inputs_text = "".join(c["text"] for c in payload["inputs"] if c.get("type") == "text")
        # context: echoes + executing (cmd!), 排除 percept/hint.
        assert "echo" in context_text and "cmd!" in context_text
        assert "percept" not in context_text and "hint text" not in context_text
        # inputs: percept 平铺 + hint (hint 排后).
        assert "percept" in inputs_text and "hint text" in inputs_text
        assert inputs_text.index("percept") < inputs_text.index("hint text")

    def test_moment_payload_empty_when_no_context_or_inputs(self):
        from ghoshell_moss.core.blueprint.moment import Moment

        payload = self._ego()._moment_payload(Moment())
        assert payload["context"] == []
        assert payload["inputs"] == []

    def test_moment_content_parts_concatenates_context_and_inputs(self):
        from ghoshell_moss.core.blueprint.moment import Echoes, Moment
        from ghoshell_moss.message import Message

        moment = Moment(
            previous=Echoes(messages=[Message.new().with_content("echo")]),
            percepts={"test": [Message.new().with_content("percept")]},
        )
        parts = self._ego()._moment_content_parts(moment)
        text = "".join(c["text"] for c in parts if c.get("type") == "text")
        assert "echo" in text
        assert "percept" in text

    def test_moment_content_parts_empty_when_no_content(self):
        from ghoshell_moss.core.blueprint.moment import Moment

        assert self._ego()._moment_content_parts(Moment()) == []


class TestDoloresEpochPayload:
    """DoloresEgo 的 epoch 槽位 — <epoch> 容器 (recap + baseline), epoch 变更时才返回."""

    def _ego(self):
        from ._ego import DoloresEgo, DoloresEgoContext

        return DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                instruction="i",
            ),
        )

    def _thinking(self, epoch):
        class _Observer:
            @property
            def epoch(self):
                return epoch

        class _Thinking:
            def __init__(self, observer):
                self.observer = observer

        return _Thinking(_Observer())

    def test_epoch_payload_renders_epoch_container(self):
        from ghoshell_moss.core.blueprint.moment import Epoch
        from ghoshell_moss.message import Message

        epoch = Epoch(
            id="e1",
            index=1,
            recap=[Message.new(tag="summary").with_content("past")],
            baseline={"facade": "channel tree"},
        )
        payload = self._ego()._epoch_payload(self._thinking(epoch))
        assert payload is not None
        text = "".join(c["text"] for c in payload if c.get("type") == "text")
        assert "<epoch" in text and 'index="1"' in text
        assert "<recap>" in text and "past" in text
        assert "<baseline>" in text and "<facade>" in text and "channel tree" in text

    def test_epoch_payload_none_when_epoch_unchanged(self):
        from ghoshell_moss.core.blueprint.moment import Epoch

        ego = self._ego()
        epoch = Epoch(id="e1", index=1, recap=[], baseline={})
        assert ego._epoch_payload(self._thinking(epoch)) is None  # 空 epoch → 不注入
        # 已记录的 epoch 再次进入 → None (不变更).
        epoch2 = Epoch(id="e1", index=1, recap=[], baseline={})
        assert ego._epoch_payload(self._thinking(epoch2)) is None
