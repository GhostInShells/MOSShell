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
        """无 trajectory (shell=None) → articulate 只产出占位 logos, 不 crash."""
        ghost = _dolores()

        async def collect():
            return [delta async for delta in ghost.think(None)]

        assert asyncio.run(collect()) == [""]


class TestDoloresTrajectory:
    def test_no_trajectory_without_shell(self):
        """无 shell → 不挂载 trajectory, articulate 静默占位."""
        ghost = _dolores()

        async def run():
            async with ghost:
                assert ghost._trajectory is None
                with pytest.raises(RuntimeError):
                    ghost.trajectory
                collected = [delta async for delta in ghost.think(None)]
                assert collected == [""]

        asyncio.run(run())

    @pytest.mark.asyncio
    async def test_mounts_trajectory_when_shell_running(self):
        """shell running → __aenter__ 挂载 trajectory, 句柄可访问且 is_running."""
        from ghoshell_moss.core.ctml.shell import new_ctml_shell

        shell = new_ctml_shell("dolores_traj_mount")
        ghost = _dolores(shell=shell)
        async with shell:
            async with ghost:
                assert ghost._trajectory is not None
                assert ghost.trajectory.is_running()

    @pytest.mark.asyncio
    async def test_skips_mount_when_shell_not_running(self):
        """shell 未启动 → 跳过挂载, trajectory 保持 None."""
        from ghoshell_moss.core.ctml.shell import new_ctml_shell

        shell = new_ctml_shell("dolores_traj_skip")
        ghost = _dolores(shell=shell)
        async with ghost:
            assert ghost._trajectory is None
            with pytest.raises(RuntimeError):
                ghost.trajectory

    @pytest.mark.asyncio
    async def test_articulate_emits_epoch_and_frame_to_output(self):
        """articulate 首帧写 epoch_start 全量 facade, 每轮写 trajectory frame → output."""
        from ghoshell_moss.core.ctml.shell import new_ctml_shell

        shell = new_ctml_shell("dolores_traj_output")
        session = MockSession()
        ghost = _dolores(shell=shell, session=session)
        async with shell:
            async with ghost:
                collected = [delta async for delta in ghost.think(None)]
        assert collected == [""]
        # output 面出现 trajectory 角色: epoch start (全量 facade) + frame (帧投影).
        traj_outputs = [o for o in session.outputs if o.role == "trajectory"]
        assert len(traj_outputs) >= 2
        assert "epoch start" in traj_outputs[0].log
        epoch_text = traj_outputs[0].messages_string()
        assert "<channel" in epoch_text
        frame_text = traj_outputs[1].messages_string()
        assert "<moss" in frame_text


class TestDoloresInstruction:
    def test_system_prompt_derives_two_meta_segments(self):
        """system_prompt = 原型元信息 + 身份描述, 从结构化 meta 派生, 无 baseline 时不含 baseline."""
        text = _dolores().system_prompt()
        assert "prototype: Dolores" in text
        assert "version: dev_1" in text
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


class TestDoloresEgoNucleus:
    """DoloresEgoNucleus 最小流程 — signal → info 级空 body 默认 mode impulse."""

    @pytest.mark.asyncio
    async def test_signal_produces_info_empty_default_impulse(self):
        from ghoshell_moss.core.blueprint.mindflow import Priority

        from .nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(new_dolores_ego_signal())

        assert len(impulses) == 1
        imp = impulses[0]
        assert imp.priority == Priority.INFO
        assert imp.messages == []
        # 默认 mode (空) = 正常仲裁, 非 silent buffer.
        assert imp.mode == ""

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
        from ._ego import DoloresEgo

        return DoloresEgo(None)  # gate 路径不读 ghost 反参, 传 None 即可

    @pytest.mark.asyncio
    async def test_turn_start_self_wakes_when_idle(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        await ego._on_turn_start(None)  # type: ignore[arg-type]  # gate 不读 event

        assert len(emitted) == 1
        assert emitted[0].name == "dolores/ego"

    @pytest.mark.asyncio
    async def test_turn_start_suppressed_when_articulating(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        ego.articulating = True
        await ego._on_turn_start(None)  # type: ignore[arg-type]

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
    """DoloresRun 的 ego fake — Duck-typed (session/_articulating/_rpc_*)."""

    def __init__(self, session):
        self.session = session
        self._articulating = False
        self.enter_calls = 0
        self.exit_calls = 0
        self.enter_error: Exception | None = None

    async def _rpc_thinking_enter(self, thinking):
        self.enter_calls += 1
        if self.enter_error is not None:
            raise self.enter_error

    async def _rpc_thinking_exit(self):
        self.exit_calls += 1


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
            assert ego._articulating is True
            assert len(session.handlers) == 1  # catch-all 监听已绑
            await asyncio.sleep(0)  # 让出 loop, enter task 跑
            assert ego.enter_calls == 1
        assert ego._articulating is False
        assert ego.exit_calls == 1
        assert len(session.handlers) == 0  # 解绑

    @pytest.mark.asyncio
    async def test_events_consumes_and_terminates_on_poison(self):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, SessionEventMeta

        session = FakeRunSession()
        ego = FakeRunEgo(session)
        run = self._run(session=session, ego=ego)
        collected = []
        async with run:
            await session.emit(SessionEvent(meta=SessionEventMeta(type="turn/start", seq=1), data={"turn": 1}))
            await session.emit(SessionEvent(meta=SessionEventMeta(type="step/start", seq=2), data={"turn": 1, "step": 1}))
            async for event in run.events():  # enter task 完成会塞毒丸 → 终止
                collected.append(event.meta.type)
        assert "turn/start" in collected
        assert "step/start" in collected

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
