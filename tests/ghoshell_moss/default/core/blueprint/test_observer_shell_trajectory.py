"""Observer 与 ShellTrajectory 的两帧装线单测.

ShellTrajectory (shell 运行时观测轨迹) 作为 Observer/Moments (感知观测轨迹)
的 shell 观测来源. 每个 round 一拆两帧, 走不同槽位:

- 帧1 (moment 生产时 / 执行前): trajectory 观测 → Moment.percepts (输入, 基于什么行动).
- 帧2 (执行结束后): trajectory 观测 → Echoes / add_echoes → 下一帧 Moment.previous (feedback, 行动带回什么).
- 执行过程中: 模型 logos 流 buffer 到 moment.logos (行动本身).

need_observe 完全由命令本身决定: 帧2 的 TrajectoryFrame.need_observe 是 bool,
由事件携带 (ShellTaskDoneEvent 来自 task_result().observe, InterpreterStoppedEvent
来自 interpretation().observe). 再传给 add_echoes → observer.need_observe() 反映命令语义.

moments (observer) 每轮只产一帧 (percepts 帧); trajectory 被 poll 两帧.

仿照 test_shell_trajectory.py 的真实 shell 集成风格.
"""

import pytest

from ghoshell_moss.core.blueprint.moment import BaseMomentsObserver, Moment
from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.message import Message


def _text(msg: Message) -> str:
    """取一条消息的首个文本内容, 简化断言."""
    for c in msg.contents:
        if "text" in c:
            return c["text"]
    return ""


def _texts(msgs) -> list[str]:
    return [_text(m) for m in msgs]


def _flatten(ctx: dict[str, list[Message]]) -> list[Message]:
    """把 facade.dynamic_context() 的 per-channel 映射压平成消息流."""
    return [msg for messages in ctx.values() for msg in messages]


def _new_shell(name: str, *, always_observe: bool = False):
    """构一个带 hello 命令 + 动态上下文的 shell."""
    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    from ghoshell_moss.core.ctml.shell import new_ctml_shell

    shell = new_ctml_shell(name)
    chan = new_channel(name="chan")
    phase = {"v": None}

    @chan.build.command(always_observe=always_observe)
    async def hello() -> str:
        return "world"

    @chan.build.context_messages
    async def context() -> list[Message]:
        return [Message.new().with_content(f"phase-{phase['v']}")]

    shell.main_channel.import_channels(chan)
    return shell, phase


@pytest.mark.asyncio
async def test_two_frame_separation_percepts_vs_echoes():
    """帧1 → 执行前 percepts; 帧2 → 执行后 result; logos 缓冲到 moment.

    驱动一条命令: 生产 moment (percepts = 执行前帧, 无 task-done) → 执行 ctml
    (logos 缓冲) → 帧2 作为 add_echoes (need_observe 由 shell status 决定) →
    只进入下一帧 moment 的 previous, 不污染当前 moment 的 percepts.
    """
    shell, _ = _new_shell("two_frame")
    obs = BaseMomentsObserver(max_size=10)
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            obs.with_dynamic_context_func("shell", lambda: _flatten(trajectory.facade.dynamic_context()))
            obs.with_percepts_buffer("shell", lambda: trajectory.pop_frame().project(with_dynamic=False))

            # 帧1: 执行前生产 moment.
            m1 = obs.observe()
            assert isinstance(m1, Moment)
            # percepts 是执行前帧: 尚无 task-done 结果.
            assert "world" not in _texts(m1.percepts["shell"])

            # 执行 ctml, logos 缓冲到 moment.
            async with shell.interpreter_in_ctx() as interp:
                interp.feed("<chan:hello />")
                interp.commit()
                await interp.wait_tasks(timeout=2)
            m1.logos = "<chan:hello />"
            assert m1.logos == "<chan:hello />"

            # 帧2: 执行结束后 → result (add_echoes). need_observe 是帧携带的 bool, 由命令决定.
            frame_b = trajectory.pop_frame()
            assert frame_b.need_observe is False  # hello 无 observe 请求.
            obs.add_echoes(frame_b.project(with_dynamic=False), need_observe=frame_b.need_observe)
            assert obs.need_observe() is False
            # 结果不进当前 moment 的 percepts (仍是执行前帧).
            assert "world" not in _texts(m1.percepts["shell"])

            # 下一帧 moment 捕获执行结果作 previous (feedback).
            m2 = obs.observe()
            assert any("world" in t for t in _texts(m2.previous_echoes_messages()))


@pytest.mark.asyncio
async def test_sequential_two_frame_loop_twenty_rounds():
    """同步时序: 20 轮 (执行前帧→percepts, 执行后帧→result), 验证两帧装线的不变式.

    每轮: refresh_metas → observe (帧1) → 跑命令 + buffer logos → pop 帧2 → add_echoes.
    need_observe 由帧2 的 TrajectoryFrame.need_observe 决定 (hello 无 observe 请求 → 恒 False).

    不变式:
    - 每帧 percepts 是执行前帧, 永远不含 task-done 结果 (那是 result).
    - 每个 round 的 logos 都 buffer 到对应 moment.
    - 结果零丢失: 前 rounds-1 个 moment 的 previous 各带一个 task-done.
    - 结果零重复: percepts 从不携带 task-done.
    - need_observe 反映命令: hello 恒为 False (status.need_observe == 0).
    """
    shell, phase = _new_shell("two_frame_loop")
    obs = BaseMomentsObserver(max_size=30)
    rounds = 20
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            obs.with_dynamic_context_func("shell", lambda: _flatten(trajectory.facade.dynamic_context()))
            obs.with_percepts_buffer("shell", lambda: trajectory.pop_frame().project(with_dynamic=False))

            for i in range(rounds):
                phase["v"] = str(i)
                # 上下文新鲜度依赖 metas 刷新, 必须显式触发.
                await shell.refresh_metas(timeout=2)

                # 帧1: 生产 moment (执行前).
                m = obs.observe()
                assert obs.need_observe() is False  # observe 已归零.
                # percepts 是无 task-done 的执行前帧.
                assert "world" not in _texts(m.percepts["shell"])
                # 动态上下文读到当前 phase.
                assert any(f"phase-{i}" in t for t in _texts(m.dynamic_context["shell"]))

                # 执行 ctml, logos 缓冲到 moment.
                async with shell.interpreter_in_ctx() as interp:
                    interp.feed("<chan:hello />")
                    interp.commit()
                    await interp.wait_tasks(timeout=2)
                m.logos = "<chan:hello />"

                # 帧2: 执行结束后 → result, 进下一 moment.previous. need_observe 由命令决定.
                frame_b = trajectory.pop_frame()
                assert frame_b.need_observe is False
                obs.add_echoes(frame_b.project(with_dynamic=False), need_observe=frame_b.need_observe)
                assert obs.need_observe() is False

            # 零重复帧: 恰好 rounds 个 moment.
            all_moments = obs.moments()
            assert len(all_moments) == rounds
            # 每轮 logos 已缓冲.
            assert all(m.logos == "<chan:hello />" for m in all_moments)
            # 结果是 feedback (previous), 不是 percepts: 任意 moment 的 percepts 不含 task-done.
            assert all("world" not in _texts(m.percepts["shell"]) for m in all_moments)
            # 零丢失: 前 rounds-1 个 moment 的 previous 各带一个 task-done 结果.
            world_in_previous = sum(
                sum(1 for t in _texts(m.previous_echoes_messages()) if "world" in t)
                for m in all_moments
            )
            assert world_in_previous == rounds - 1


@pytest.mark.asyncio
async def test_need_observe_derived_from_command():
    """need_observe 完全由命令决定: 从帧2 的 shell status 传染给 observer.

    普通 hello (无 observe 请求) → obs.need_observe() False;
    always_observe 命令 → obs.need_observe() True, 再 observe() 归零.
    这是独立测试点: 立刻断言 obs.need_observe() 反映命令语义.
    """
    # 普通命令: 无 observe 请求.
    shell_norm, _ = _new_shell("need_obs_norm")
    obs_norm = BaseMomentsObserver(max_size=5)
    async with shell_norm:
        async with MShellTrajectory(shell_norm) as trajectory:
            obs_norm.with_percepts_buffer("shell", lambda: trajectory.pop_frame().project(with_dynamic=False))
            async with shell_norm.interpreter_in_ctx() as interp:
                interp.feed("<chan:hello />")
                interp.commit()
                await interp.wait_tasks(timeout=2)
            fb = trajectory.pop_frame()
            assert fb.need_observe is False
            obs_norm.add_echoes(fb.project(with_dynamic=False), need_observe=fb.need_observe)
            assert obs_norm.need_observe() is False

    # always_observe 命令: 请求下一次 observe.
    shell_obs, _ = _new_shell("need_obs_yes", always_observe=True)
    obs_obs = BaseMomentsObserver(max_size=5)
    async with shell_obs:
        async with MShellTrajectory(shell_obs) as trajectory:
            obs_obs.with_percepts_buffer("shell", lambda: trajectory.pop_frame().project(with_dynamic=False))
            async with shell_obs.interpreter_in_ctx() as interp:
                interp.feed("<chan:hello />")
                interp.commit()
                await interp.wait_tasks(timeout=2)
            fb = trajectory.pop_frame()
            assert fb.need_observe is True
            obs_obs.add_echoes(fb.project(with_dynamic=False), need_observe=fb.need_observe)
            assert obs_obs.need_observe() is True
            # observe 归零.
            obs_obs.observe()
            assert obs_obs.need_observe() is False


# ============================================================
# ghost_runtime._wire_mindflow 装线 — 正式验证套件
# ============================================================
# 上面的旧测试走 with_percepts_buffer + 手工 add_echoes 的"两帧分开"路径.
# ghost_runtime (host/ghost_runtime.py _wire_mindflow) 用的是另几条装线:
#   trajectory.when_need_observe(_notify_moments_need_observe)
#   moments.on_moment_created(_on_moments_observing)
#   moments.on_epoch_creating(lambda _e: trajectory.new_epoch())   # 反向绑定
#   moments.with_epoch_baseline("facade", lambda: trajectory.epoch_start_point(refresh=False))
# 下面复刻这几条装线, 验证其行为契约. 不改动上面旧测试.


def _runtime_wire(obs: BaseMomentsObserver, trajectory: MShellTrajectory):
    """复刻 ghost_runtime._wire_mindflow 的装线回调."""
    def _on_moments_observing(moment: Moment) -> None:
        frame = trajectory.pop_frame()
        if moment.previous is not None:
            messages = frame.project(with_dynamic=False)
            moment.previous.add_echoes(messages, frame.need_observe)
            moment.previous.need_observe = frame.need_observe

    def _notify_moments_need_observe(e):
        # 只通知"该观察了", 不带内容 — add_echoes 已处理空信号置位.
        obs.add_echoes([], need_observe=True)

    obs.on_moment_created(_on_moments_observing)
    trajectory.when_need_observe(_notify_moments_need_observe)
    obs.on_epoch_creating(lambda _epoch: trajectory.new_epoch())
    obs.with_epoch_baseline("facade", lambda: trajectory.epoch_start_point(refresh=False))


@pytest.mark.asyncio
async def test_runtime_wiring_need_observe_notifies_observer():
    """always_observe 命令完成后, when_need_observe 通知链应让 obs.need_observe() 为 True.

    这是 _notify_moments_need_observe 的契约: shell 出现 need_observe 事件时,
    observer 置位观察信号, 驱动下一轮. 修复前因 when_need_observe 不 fire 而失效.
    """
    shell, _ = _new_shell("rt_notify", always_observe=True)
    obs = BaseMomentsObserver(max_size=5)
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            _runtime_wire(obs, trajectory)
            async with shell.interpreter_in_ctx() as interp:
                interp.feed("<chan:hello />")
                interp.commit()
                await interp.wait_tasks(timeout=2)
            # 通知链: need_observe 事件 -> _notify -> add_echoes([], True) -> 置位.
            assert obs.need_observe() is True


@pytest.mark.asyncio
async def test_runtime_wiring_frame_injected_into_previous_on_observe():
    """observe 时拉一帧轨迹, project 内容进 moment.previous (feedback), 不进 percepts.

    与旧观测面 (帧进 percepts) 的关键区别: ghost_runtime 把轨迹帧当上一轮反馈缝合.
    """
    shell, _ = _new_shell("rt_frame_inject")
    obs = BaseMomentsObserver(max_size=10)
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            _runtime_wire(obs, trajectory)
            async with shell.interpreter_in_ctx() as interp:
                interp.feed("<chan:hello />")
                interp.commit()
                await interp.wait_tasks(timeout=2)

            m = obs.observe()
            assert isinstance(m, Moment)
            assert m.previous is not None
            prev = _texts(m.previous_echoes_messages())
            assert any("world" in t for t in prev)  # task-done 反馈
            assert any("<moss at=" in t for t in prev)  # project 的 moss 容器
            # 帧不进 percepts: 那是 with_percepts_buffer 的路径.
            assert "world" not in _texts(m.percepts_messages())


@pytest.mark.asyncio
async def test_runtime_wiring_epoch_baseline_is_full_facade():
    """with_epoch_baseline 用 trajectory 全量 facade 作为 epoch baseline (facade 槽位)."""
    shell, _ = _new_shell("rt_epoch_baseline")
    obs = BaseMomentsObserver(max_size=5)
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            _runtime_wire(obs, trajectory)
            epoch = obs.new_epoch([])
            assert "facade" in epoch.baseline
            assert '<channel path="chan">' in epoch.baseline["facade"]
            # recap 槽位是前情提要, 不应再夹带 facade.
            assert epoch.recap == []


@pytest.mark.asyncio
async def test_runtime_wiring_epoch_creating_refreshes_trajectory_baseline():
    """反向绑定: moments.new_epoch → on_epoch_creating → trajectory.new_epoch.

    baseline 快照之后新增 channel, 若不反绑, epoch.baseline["facade"] 会读 stale
    baseline (不含新 channel); 反绑后 trajectory 先刷新, baseline facade 反映最新 metas.
    """
    from ghoshell_moss.core.py_channel import PyChannel

    shell, _ = _new_shell("rt_reverse_bind")
    obs = BaseMomentsObserver(max_size=5)
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            _runtime_wire(obs, trajectory)
            # baseline 快照之后运行时新增 channel.
            chan = PyChannel(name="chan2")

            @chan.build.command()
            async def hi() -> str:
                return "hi"

            shell.main_channel.add_virtual_channel(chan)
            await shell.refresh_metas()

            epoch = obs.new_epoch([])
            assert '<channel path="chan2">' in epoch.baseline["facade"]


@pytest.mark.asyncio
async def test_epoch_zero_delivers_late_channel_and_frame_echo_stays_empty():
    """链路: 晚进 metas 的 channel (如 mindflow 反身 channel) 全量 facade 在第零帧交付.

    真实运行时序: trajectory.__aenter__ 快照 baseline (此刻 mirror 未进 metas) → mirror 挂上
    shell + refresh (进 metas) → 首次 observer epoch (反向绑定 trajectory.new_epoch) 重快照
    baseline (含 mirror) → ``epoch.baseline['facade']`` 含 mirror (第零帧交付), epoch 之后首个
    frame 的 facade_delta 为空 — echo 不重复携带 mirror facade (反泄漏).

    对照昨天 mindflow 单测 (test_mindflow_channel_help_not_duplicated_across_frames): 它从不建
    observer epoch, trajectory 用 ``__aenter__`` 旧 baseline (无 mirror), 首帧 echo 泄漏 mirror
    全量 facade — 那是"缺 epoch"的伪像. 本测试显式建 observer epoch, 锁真实链路的反泄漏契约.
    """
    from ghoshell_moss.core.py_channel import PyChannel

    shell, _ = _new_shell("rt_mirror_epoch_zero")
    obs = BaseMomentsObserver(max_size=10)
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            _runtime_wire(obs, trajectory)
            mirror = PyChannel(name="mirror", description="mindflow mirror channel")

            @mirror.build.command()
            async def observe_me() -> str:
                return "ok"

            shell.main_channel.add_virtual_channel(mirror)
            await shell.refresh_metas()

            # 缺 observer epoch 时 (昨天 mindflow 单测的隐含前提): 首帧 diff 泄漏 mirror 全量.
            leak = trajectory.pop_frame()
            assert '<channel path="mirror">' in leak.facade_delta()

            # 真实链路: 首次 observer epoch → 反向绑定 trajectory.new_epoch 重快照 (含 mirror).
            epoch = obs.new_epoch([])
            # 第零帧交付: epoch baseline 含 mirror 全量 facade.
            assert '<channel path="mirror">' in epoch.baseline["facade"]
            # 反泄漏: epoch 之后首个 frame 的 facade_delta 为空, echo 不重复带 mirror.
            frame = trajectory.pop_frame()
            assert frame.facade_delta() == ""


@pytest.mark.asyncio
async def test_runtime_wiring_full_rounds_no_loss_dup():
    """跨 round 用 ghost_runtime 装线: 每轮 observe 恰拉一帧入 previous, 零丢失零重复."""
    shell, _ = _new_shell("rt_rounds")
    obs = BaseMomentsObserver(max_size=30)
    rounds = 8
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            _runtime_wire(obs, trajectory)
            for _ in range(rounds):
                async with shell.interpreter_in_ctx() as interp:
                    interp.feed("<chan:hello />")
                    interp.commit()
                    await interp.wait_tasks(timeout=2)
                obs.observe()

            all_m = obs.moments()
            assert len(all_m) == rounds
            # 帧从不进 percepts (零重复).
            assert all("world" not in _texts(m.percepts_messages()) for m in all_m)
            # 每轮一个 task-done 反馈 (零丢失).
            world_prev = sum(
                sum(1 for t in _texts(m.previous_echoes_messages()) if "world" in t)
                for m in all_m
            )
            assert world_prev == rounds