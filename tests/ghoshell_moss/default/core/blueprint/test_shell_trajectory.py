"""ShellTrajectory 单测.

两层:
- 纯函数级: facade_delta / status / event 的数据投影契约 (直接构造 ChannelMeta).
- 真实 shell 集成: tracer 事件捕获 / commit 纪律 / facade 接线 (new_ctml_shell 驱动).
"""

import asyncio
import datetime

import pytest

from ghoshell_moss.core.blueprint.shell_trajectory import (
    InterpreterStoppedEvent,
    MShellStatus,
    MShellTrajectory,
    ShellTaskDoneEvent,
    ShellKeyFrame,
)
from ghoshell_moss.core.concepts.channel import ChannelMeta

_BASE = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _meta(name: str = 'a', *, created: int = 0, **kwargs) -> ChannelMeta:
    return ChannelMeta(
        name=name,
        created=_BASE + datetime.timedelta(seconds=created),
        **kwargs,
    )


def _frame(previous: dict, metas: dict) -> ShellKeyFrame:
    return ShellKeyFrame(
        epoch_index=0,
        index=0,
        events=[],
        need_observe=False,
        status=MShellStatus(state='idle'),
        previous_metas=previous,
        metas=metas,
        created=0.0,
    )


# --- 纯函数: facade_delta (增 / 删 / 改 / 不变) ---


def test_facade_delta_removed_channel_emits_tombstone():
    """上一帧有、当前帧没有的 channel 必须 emit 墓碑, 否则模型保留已下线的表面."""
    frame = _frame({'a': _meta(created=0)}, {})
    delta = frame.facade_delta()
    assert '<channel path="a" removed/>' in delta


def test_facade_delta_added_channel_emits_full_facade():
    """新增 channel emit 完整 facade."""
    frame = _frame({}, {'a': _meta(help='new help', created=0)})
    delta = frame.facade_delta()
    assert '<channel path="a">' in delta
    assert 'new help' in delta


def test_facade_delta_changed_channel_emits_new_facade():
    """变更的 channel emit 新 facade, 不含旧内容."""
    frame = _frame(
        {'a': _meta(help='old help', created=0)},
        {'a': _meta(help='new help', created=1)},
    )
    delta = frame.facade_delta()
    assert 'new help' in delta
    assert 'old help' not in delta


def test_facade_delta_unchanged_emits_nothing():
    """facade 文本未变 (即使 created 变了) → 不发射."""
    frame = _frame(
        {'a': _meta(help='same', created=0)},
        {'a': _meta(help='same', created=1)},
    )
    assert frame.facade_delta() == ''


# --- 纯函数: MShellStatus.description ---


def test_moss_status_running_shows_counts():
    """running 状态展示命令计数 body."""
    status = MShellStatus(state='running', completed=2, failed=1)
    desc = status.description()
    assert 'running' in desc
    assert 'completed: 2' in desc
    assert 'failed: 1' in desc


def test_moss_status_idle_self_closes_without_counts():
    """非 running 状态自闭合, 计数被短路 (由 interpreter event 承载)."""
    status = MShellStatus(state='idle', completed=2)
    desc = status.description()
    assert '<status idle/>' in desc
    assert 'completed' not in desc


# --- 纯函数: InterpreterStoppedEvent.as_messages ---


def test_interpreter_stopped_event_renders_counts():
    """stop 事件渲染 completed/cancelled/failed 计数."""
    event = InterpreterStoppedEvent(
        index=0,
        created=0.0,
        state='done',
        completed=2,
        cancelled=1,
        failed=1,
    )
    messages = event.as_messages()
    assert len(messages) == 1
    content = messages[0].to_content_string()
    assert 'completed: 2' in content
    assert 'cancelled: 1' in content
    assert 'failed: 1' in content


def test_interpreter_stopped_event_renders_error():
    """error 状态渲染异常文本."""
    event = InterpreterStoppedEvent(index=0, created=0.0, state='error', error='boom')
    content = event.as_messages()[0].to_content_string()
    assert 'error: boom' in content


# --- 真实 shell 集成 ---


@pytest.mark.asyncio
async def test_trajectory_peek_captures_task_done():
    """驱动一条命令后, peek 的帧里应捕获 task-done 事件."""
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.py_channel import PyChannel

    shell = new_ctml_shell("traj_events")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:hello />")
                i.commit()
                await i.wait_tasks(timeout=2)

            frame = trajectory.peek()
            task_dones = [e for e in frame.events if isinstance(e, ShellTaskDoneEvent)]
            assert len(task_dones) >= 1, "task done 应被 tracer 捕获"


@pytest.mark.asyncio
async def test_trajectory_commit_drains_and_guards_stale():
    """peek 非破坏; commit 消费事件; 重复 commit 同帧返回 False."""
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.py_channel import PyChannel

    shell = new_ctml_shell("traj_commit")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:hello />")
                i.commit()
                await i.wait_tasks(timeout=2)

            # peek 非破坏: 两次 peek 事件一致.
            f1 = trajectory.peek()
            f2 = trajectory.peek()
            assert len(f1.events) == len(f2.events) > 0

            # commit 消费事件.
            assert trajectory.commit(f1) is True
            assert trajectory.peek().events == []

            # 重复 commit 同帧 → stale, 返回 False.
            assert trajectory.commit(f1) is False


@pytest.mark.asyncio
async def test_trajectory_epoch_start_point_renders_facade():
    """epoch 起点返回全量 facade, 包含 channel 的可变表面."""
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.py_channel import PyChannel

    shell = new_ctml_shell("traj_facade")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            facade = trajectory.epoch_start_point(refresh=False)
            assert '<channel path="chan">' in facade


@pytest.mark.asyncio
async def test_first_frame_is_delta_after_epoch_start_point():
    """epoch_start_point 含全量 facade ⟹ 首帧 facade_delta 为空 (不重复).

    锁「recap/epoch 起点(全量 facade) 与首帧(facade delta) 互斥」不变式:
    当 epoch 起点已把全量表面交付出去, 首帧就只应是 delta; 若首帧重新 emit
    全量, 就是重复.
    """
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.py_channel import PyChannel

    shell = new_ctml_shell("traj_first_delta")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            # epoch 起点: 全量 facade 含 chan.
            facade = trajectory.epoch_start_point(refresh=True)
            assert '<channel path="chan">' in facade

            # 首帧: facade_delta 应为空 — 表面已被 epoch 起点交付过.
            frame1 = trajectory.pop_frame()
            assert frame1.facade_delta() == ""


@pytest.mark.asyncio
async def test_first_frame_emits_channel_added_after_baseline():
    """baseline 之后新增 channel ⟹ 首帧只 emit 新增 channel (delta), 不重发旧表面.

    与上面相反的分支: epoch 起点不含该 channel (它在 baseline 快照之后才进来),
    首帧必须从头构建它, 但只构建这一个 delta, 不重复已有的.
    """
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.py_channel import PyChannel

    shell = new_ctml_shell("traj_first_new")
    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            # baseline 已快照 (无 chan). 之后运行时新增 chan.
            chan = PyChannel(name="chan")

            @chan.build.command()
            async def hello() -> str:
                return "world"

            shell.main_channel.add_virtual_channel(chan)
            await shell.refresh_metas()

            frame1 = trajectory.pop_frame()
            delta = frame1.facade_delta()
            assert '<channel path="chan">' in delta
            # 是 delta, 不重复: 该 channel 的 facade 只出现一次.
            assert delta.count('<channel path="chan">') == 1


@pytest.mark.asyncio
async def test_trajectory_empty_drain():
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.py_channel import PyChannel

    shell = new_ctml_shell("traj_facade")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    @chan.build.context_messages
    async def messages():
        return ["hello"]

    async with chan.bootstrap() as rtm:
        assert len(rtm.metas()) == 1
        messages = []
        for meta in rtm.metas().values():
            messages.extend(meta.context)
        assert len(messages) == 1

    shell.main_channel.import_channels(chan)

    async with shell:
        await shell.refresh_metas()
        dynamic_messages = []
        metas = shell.channel_metas()
        assert len(metas) == 2
        for path, meta in metas.items():
            dynamic_messages.extend(meta.context)

        assert len(dynamic_messages) > 0
        async with MShellTrajectory(shell) as trajectory:
            frame = trajectory.pop_frame()
            assert len(frame.dynamic_context_messages()) > 0
            assert len(frame.project(with_status=False)) > 0
            for i in range(10):
                # 不带 status 就没有数据.
                assert len(trajectory.pop_frame().project(with_status=False, with_dynamic=False)) == 0
                await shell.refresh_metas()


@pytest.mark.asyncio
async def test_trajectory_when_need_observe_fires_on_task_done():
    """when_need_observe 回调应在 need_observe 事件出现时被触发.

    当前 MShellEventTracer._append_event 只 append 事件, 从不调用 _need_observe_callbacks,
    导致 ghost_runtime 经 when_need_observe -> _notify_moments_need_observe 的通知链完全失效.
    驱动 always_observe 命令产生 need_observe task-done 后, 回调必须被触发.
    """
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.blueprint.channel_builder import new_channel

    shell = new_ctml_shell("traj_need_observe")
    chan = new_channel(name="chan")

    @chan.build.command(always_observe=True)
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            fired: list = []
            trajectory.when_need_observe(fired.append)

            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:hello />")
                i.commit()
                await i.wait_tasks(timeout=2)

            assert len(fired) > 0, "need_observe 事件应触发 when_need_observe 回调"
