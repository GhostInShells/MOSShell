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
    frame = _frame({}, {'a': _meta(notice='new help', created=0)})
    delta = frame.facade_delta()
    assert '<channel path="a">' in delta
    assert 'new help' in delta


def test_facade_delta_changed_channel_emits_new_facade():
    """变更的 channel emit 新 facade, 不含旧内容."""
    frame = _frame(
        {'a': _meta(notice='old help', created=0)},
        {'a': _meta(notice='new help', created=1)},
    )
    delta = frame.facade_delta()
    assert 'new help' in delta
    assert 'old help' not in delta


def test_facade_delta_unchanged_emits_nothing():
    """facade 文本未变 (即使 created 变了) → 不发射."""
    frame = _frame(
        {'a': _meta(notice='same', created=0)},
        {'a': _meta(notice='same', created=1)},
    )
    assert frame.facade_delta() == ''


def test_facade_delta_container_tag_is_facade_delta():
    """facade delta 消息用 <facade-delta> 容器 tag 包裹 (而非 <facade>)."""
    frame = _frame(
        {'a': _meta(notice='old help', created=0)},
        {'a': _meta(notice='new help', created=1)},
    )
    messages = frame.project(with_status=False, with_dynamic=False)
    facade_messages = [m for m in messages if m.meta.tag == 'facade-delta']
    assert len(facade_messages) == 1
    text = facade_messages[0].to_content_string()
    assert '<facade-delta>' in text
    assert '</facade-delta>' in text
    assert 'new help' in text
    assert 'old help' not in text  # delta, 非全量重发


def test_facade_delta_emits_only_changed_channel():
    """facade delta 只发变更的 channel, 未变更的 channel 不重发 (非全量重渲染)."""
    frame = _frame(
        {
            'a': _meta(notice='a old', created=0),
            'b': _meta(notice='b same', created=0),
        },
        {
            'a': _meta(notice='a new', created=1),
            'b': _meta(notice='b same', created=1),
        },
    )
    delta = frame.facade_delta()
    assert 'a new' in delta
    assert 'a old' not in delta
    assert 'b same' not in delta  # b 未变更, 不重发


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
    """stop 事件渲染 completed/cancelled/failed 计数, state 折进标签作裸词."""
    event = InterpreterStoppedEvent(
        index=0,
        created=0.0,
        state='done',
        completed=2,
        cancelled=1,
        failed=1,
        last_cancelled='chan:slow',
        last_failed='chan:boom',
    )
    messages = event.as_messages()
    assert len(messages) == 1
    content = messages[0].to_content_string()
    assert '<logos done>' in content
    assert 'completed 2' in content
    assert 'cancelled 1, last chan:slow' in content
    assert 'failed 1, last chan:boom' in content


def test_interpreter_stopped_event_renders_error():
    """error 状态渲染异常文本."""
    event = InterpreterStoppedEvent(index=0, created=0.0, state='error', error='boom')
    content = event.as_messages()[0].to_content_string()
    assert 'error: boom' in content


def test_interpreter_stopped_event_empty_settlement_returns_no_messages():
    """空结算 (无 completed/cancelled/failed/error) 无投影价值 → as_messages 返回空列表."""
    event = InterpreterStoppedEvent(index=0, created=0.0, state='done')
    assert event.as_messages() == []


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
async def test_trajectory_projects_interpreter_settlement_for_empty_result_command():
    """无返回值命令: ShellTaskDoneEvent 为空, 但 interpreter stop 的结算 (completed: 1) 必须投影.

    否则模型只能看到 bare <status idle/>, 无法感知命令是否真的执行 (speech 等无返回值动作).
    """
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.py_channel import PyChannel

    shell = new_ctml_shell("traj_interp_settlement")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def silent() -> None:
        return None

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:silent />")
                i.commit()
                await i.wait_tasks(timeout=2)

            frame = trajectory.pop_frame()
            stops = [e for e in frame.events if isinstance(e, InterpreterStoppedEvent)]
            assert len(stops) == 1
            assert stops[0].completed == 1
            # 帧投影必须携带结算, 而不是只剩 <status idle/>.
            texts = [m.to_content_string() for m in frame.project(with_dynamic=False)]
            assert any("completed 1" in t for t in texts)


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
async def test_virtual_child_removal_emits_tombstone():
    """virtual children 从树上消失 ⟹ 帧差分 emit 墓碑 (端到端).

    mesh channel 的器官投影由 virtual_children 回调供给: 回调不再返回某个 child 之后,
    下一帧必须显式告知模型它已下线, 否则模型会一直保留一个已经不存在的表面.
    上一帧 meta 表里有、当前帧没有的 path, 由 facade_delta 自动发墓碑 — 生产者无需
    产出任何 "removed" 文本 (那是 notice 片段级才需要的协作).
    """
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    from ghoshell_moss.core.concepts.channel import Channel

    shell = new_ctml_shell("traj_virtual_removed")
    mesh = new_channel(name="mesh")
    # 挂载名是回调返回的 key ('voice'); 子 channel 自己的名字 ('main') 不进路径.
    voice = new_channel(name="main")

    @voice.build.command()
    async def say(text: str) -> str:
        return text

    live: dict[str, Channel] = {"voice": voice}

    @mesh.build.virtual_children
    def _children() -> dict[str, Channel]:
        return dict(live)

    shell.main_channel.import_channels(mesh)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            # 前置: 基线 (epoch 起点) 里它是挂着的.
            assert 'path="mesh.voice"' in trajectory.epoch_start_point(refresh=False)

            live.clear()
            await shell.refresh_metas()
            frame = trajectory.pop_frame()
            assert '<channel path="mesh.voice" removed/>' in frame.facade_delta()


@pytest.mark.asyncio
async def test_unavailable_channel_emits_tombstone_then_reappears():
    """available 临时中断 ⟹ 有 -> 移除 -> 出现 (端到端).

    channel 的 available 函数返回 False 表示"临时不可用" — 表面必须从模型面消失,
    否则模型会继续对着一个已经关掉的 channel 讲话 (残留 affordance). available 恢复
    之后表面必须完整回来. 帧差分: 不可用那帧 emit 墓碑, 恢复那帧 emit 全量 facade,
    之后无变化不再重发.

    生产者只需翻转 available 函数, 不产出任何 "removed" 文本 — 墓碑由 facade_delta
    对 "上一帧有、当前帧没有" 的 path 自动生成.
    """
    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    from ghoshell_moss.core.ctml.shell import new_ctml_shell

    shell = new_ctml_shell("traj_available")
    chan = new_channel(name="chan")
    available = {"ok": True}
    chan.build.available(lambda: available["ok"])

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async def _advance():
        # 推进 tracer 事件索引, 让帧能被 commit (见 commit 的 index > last_index 守卫).
        async with shell.interpreter_in_ctx() as i:
            i.feed("<noop />")
            i.commit()
            await i.wait_tasks(timeout=2)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            # 有: 基线里它在.
            assert 'path="chan"' in trajectory.epoch_start_point(refresh=True)

            # 第一次刷新: available false -> 移除.
            available["ok"] = False
            await shell.refresh_metas()
            await _advance()
            frame = trajectory.pop_frame()
            assert '<channel path="chan" removed/>' in frame.facade_delta()

            # 第二次刷新: available true -> 出现.
            available["ok"] = True
            await shell.refresh_metas()
            await _advance()
            frame = trajectory.pop_frame()
            delta = frame.facade_delta()
            assert '<channel path="chan">' in delta
            assert 'hello' in delta  # 命令界面重新出现

            # 第三次刷新: 无变化 -> 不再重发该 channel.
            await shell.refresh_metas()
            await _advance()
            frame = trajectory.pop_frame()
            assert 'path="chan"' not in frame.facade_delta()


@pytest.mark.asyncio
async def test_frame_delta_sees_metas_regenerated_outside_shell_layer():
    """meta 在 shell 层之外重生成 (tree 级刷新) 时, 帧差分必须照样看见变化.

    facade 曾持有 meta 快照, 只在 shell 层的生成回调里更新; 而 channel 自刷 / virtual
    children 变化走的是 tree 级刷新, 不 fire 那个回调. 快照与活数据一分叉, 上一帧与当前帧
    就同源, notice 与 named notice 的变化会被 ``created`` 闸门一起吞掉. 这里用 tree 级
    refresh 复现该路径: 同一个 channel 的无名 notice 与有名片段同时变, 两者必须分别签发.
    """
    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    from ghoshell_moss.core.ctml.shell import new_ctml_shell

    shell = new_ctml_shell("traj_meta_outside_shell")
    chan = new_channel(name="chan")
    state = {"notice": "n0", "vision": "v0"}

    @chan.build.notice
    def _notice() -> str:
        return state["notice"]

    @chan.build.named_notices
    def _named() -> dict[str, str | None]:
        return {"vision": state["vision"]}

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            assert "n0" in trajectory.epoch_start_point(refresh=True)

            state["notice"] = "n1"
            state["vision"] = "v1"
            # 不走 shell.refresh_metas: 直接在 tree 上重生成这个节点的 meta.
            await shell.runtime.tree.refresh(chan.id(), wait=True)

            delta = trajectory.pop_frame().facade_delta()
            assert "<notice>" in delta
            assert "n1" in delta  # 无名 notice
            assert "<vision>v1</vision>" in delta  # 有名片段


@pytest.mark.asyncio
async def test_instruction_re_renders_at_epoch_start_point():
    """instruction 不再 startup 冻结 — refresh 重渲染, 新 epoch 的全量 facade 带新内容.

    锁「durable 面可演化」不变式: instruction 注册一个读可变状态的函数, 状态变了之后
    refresh + 重建 epoch, ``epoch_start_point`` 交付的全量 facade 必须是新 instruction
    文本 (而不是一直停在 startup 快照)。
    """
    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    from ghoshell_moss.core.ctml.shell import new_ctml_shell

    shell = new_ctml_shell("traj_instruction")
    chan = new_channel(name="chan")
    store = {"v": "initial"}
    chan.build.instruction(lambda: f"value={store['v']}")
    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            facade = trajectory.epoch_start_point(refresh=True)
            assert "value=initial" in facade
            store["v"] = "changed"
            await shell.refresh_metas()
            facade = trajectory.epoch_start_point(refresh=True)
            assert "value=changed" in facade
            assert "value=initial" not in facade


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


@pytest.mark.asyncio
async def test_trajectory_when_need_observe_survives_new_epoch():
    """when_need_observe 订阅必须跨 epoch 重建存活.

    真实运行时会重建 epoch: dolores ego 每帧 thinking 起点访问 observer.epoch
    (懒创建) → moments.new_epoch → on_epoch_creating → trajectory.new_epoch。
    new_epoch 会 close 旧 tracer 并换一个新实例; 订阅若绑在 tracer 实例上,
    重建后通知链静默失效 (D25: observe=True 不驱动下一帧, 只能靠界面输入)。
    """
    from ghoshell_moss.core.ctml.shell import new_ctml_shell
    from ghoshell_moss.core.blueprint.channel_builder import new_channel

    shell = new_ctml_shell("traj_need_observe_epoch")
    chan = new_channel(name="chan")

    @chan.build.command(always_observe=True)
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with MShellTrajectory(shell) as trajectory:
            fired: list = []
            trajectory.when_need_observe(fired.append)

            # epoch 重建 — 底层 tracer 被替换.
            trajectory.new_epoch()

            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:hello />")
                i.commit()
                await i.wait_tasks(timeout=2)

            assert len(fired) > 0, "epoch 重建后 need_observe 订阅应仍然生效"
