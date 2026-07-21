import asyncio

import pytest

from ghoshell_moss.core.blueprint.states_channel import (
    new_channel_state, new_stateful_channel_from_main, new_prime_channel,
    new_stateful_channel,
)
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.message import Message


# ============================================================
# StatefulChannel (PrimeChannel) 基础测试
# ============================================================


@pytest.mark.asyncio
async def test_main_state_commands():
    """main_state 的命令始终可用."""
    chan = new_prime_channel(name="main")

    @chan.build.command()
    async def foo() -> str:
        return "foo"

    async with chan.bootstrap() as runtime:
        cmd = runtime.get_command("foo")
        assert cmd is not None
        assert await cmd() == "foo"


@pytest.mark.asyncio
async def test_register_state_via_new_state():
    """new_state() 返回 ChannelStateBuilder, 注册为可切换 state."""
    chan = new_stateful_channel(name="main")
    st = chan.new_state("idle", "idle state")

    @st.command()
    async def relax() -> str:
        return "relaxing"

    assert "idle" in chan.states()
    assert chan.states()["idle"] is st


@pytest.mark.asyncio
async def test_register_state_via_with_state():
    """with_state() 接受外部 ChannelState."""
    chan = new_stateful_channel(name="main")
    ext_state = new_channel_state(name="external", description="external state")

    @ext_state.command()
    async def ext_cmd() -> str:
        return "external"

    chan.with_state(ext_state)
    assert "external" in chan.states()


@pytest.mark.asyncio
async def test_switch_state_basic():
    """切换到 state 后, 其命令出现在 runtime."""
    chan = new_stateful_channel(name="main")
    idle_st = chan.new_state("idle", "idle state")

    @idle_st.command()
    async def relax() -> str:
        return "relaxing"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("idle")
        cmd = runtime.get_command("relax")
        assert cmd is not None
        assert await cmd() == "relaxing"


@pytest.mark.asyncio
async def test_main_and_current_state_commands_coexist():
    """main_state 和 current_state 的命令同时存在."""
    chan = PyChannel(name="main")

    @chan.build.command()
    async def main_cmd() -> str:
        return "main"

    work_st = chan.new_state("work", "working state")

    @work_st.command()
    async def work_cmd() -> str:
        return "work"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        assert runtime.get_command("main_cmd") is not None
        assert runtime.get_command("work_cmd") is not None
        assert await runtime.get_command("main_cmd")() == "main"
        assert await runtime.get_command("work_cmd")() == "work"


@pytest.mark.asyncio
async def test_state_command_priority_main_wins():
    """main_state 和 current_state 有同名命令时, main 优先."""
    chan = PyChannel(name="main")

    @chan.build.command()
    async def foo() -> str:
        return "main"

    alt_st = chan.new_state("alt", "alt state")

    @alt_st.command()
    async def foo() -> str:
        return "alt"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("alt")
        cmd = runtime.get_command("foo")
        assert await cmd() == "main"


# ============================================================
# 状态切换与停止
# ============================================================


@pytest.mark.asyncio
async def test_stop_current_state():
    """stop_current_state 移除 current_state 的命令."""
    chan = PyChannel(name="main")

    @chan.build.command()
    async def main_cmd() -> str:
        return "main"

    work_st = chan.new_state("work", "working")

    @work_st.command()
    async def work_cmd() -> str:
        return "work"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        assert runtime.get_command("work_cmd") is not None

        await runtime.stop_current_state()
        assert runtime.get_command("main_cmd") is not None
        assert runtime.get_command("work_cmd") is None


@pytest.mark.asyncio
async def test_switch_between_multiple_states():
    """A → B → C 连续切换, 每次切换后只有最新 state 的命令生效."""
    chan = PyChannel(name="main")

    @chan.build.command()
    async def base() -> str:
        return "base"

    a_st = chan.new_state("a", "state a")
    b_st = chan.new_state("b", "state b")
    c_st = chan.new_state("c", "state c")

    @a_st.command()
    async def cmd_a() -> str:
        return "a"

    @b_st.command()
    async def cmd_b() -> str:
        return "b"

    @c_st.command()
    async def cmd_c() -> str:
        return "c"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("a")
        assert runtime.get_command("cmd_a") is not None
        assert runtime.get_command("cmd_b") is None

        await runtime.switch_state("b")
        assert runtime.get_command("cmd_a") is None
        assert runtime.get_command("cmd_b") is not None
        assert runtime.get_command("cmd_c") is None

        await runtime.switch_state("c")
        assert runtime.get_command("cmd_b") is None
        assert runtime.get_command("cmd_c") is not None
        # main_state 命令始终存在
        assert runtime.get_command("base") is not None


@pytest.mark.asyncio
async def test_switch_state_executes_command_in_new_state():
    """切换到 state 后, 命令在 state 的上下文中执行."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")

    @work_st.command()
    async def report() -> str:
        t = ChannelCtx.task()
        assert t is not None
        return "done"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        task = runtime.create_command_task("report")
        runtime.push_task(task)
        result = await task
        assert result == "done"


# ============================================================
# State 生命周期
# ============================================================


@pytest.mark.asyncio
async def test_state_on_startup_called():
    """switch_state 触发新 state 的 on_startup."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")
    started = []

    @work_st.startup
    async def on_start() -> None:
        started.append(1)

    async with chan.bootstrap() as runtime:
        assert len(started) == 0
        await runtime.switch_state("work")
        assert len(started) == 1


@pytest.mark.asyncio
async def test_state_on_close_called_on_stop():
    """stop_current_state 触发 current_state 的 on_close."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")
    closed = []

    @work_st.close
    async def on_close_fn() -> None:
        closed.append(1)

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        assert len(closed) == 0
        await runtime.stop_current_state()
        assert len(closed) == 1


@pytest.mark.asyncio
async def test_state_on_close_called_on_switch_away():
    """从 A 切换到 B 时, A 的 on_close 被调用."""
    chan = PyChannel(name="main")
    a_st = chan.new_state("a", "state a")
    b_st = chan.new_state("b", "state b")
    closed_a = []
    closed_b = []

    @a_st.close
    async def close_a() -> None:
        closed_a.append(1)

    @b_st.close
    async def close_b() -> None:
        closed_b.append(1)

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("a")
        await runtime.switch_state("b")
        assert len(closed_a) == 1
        assert len(closed_b) == 0


@pytest.mark.asyncio
async def test_state_on_running_created_as_task():
    """switch_state 为 current_state 创建 on_running task."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")
    running_started = asyncio.Event()
    running_done = asyncio.Event()

    @work_st.running
    async def on_run() -> None:
        running_started.set()
        await running_done.wait()

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        await running_started.wait()
        # on_running task 正在运行
        running_done.set()
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_state_on_running_cancelled_on_switch():
    """从 A 切换到 B 时, A 的 on_running task 被取消."""
    chan = PyChannel(name="main")
    a_st = chan.new_state("a", "state a")
    b_st = chan.new_state("b", "state b")
    a_cancelled = asyncio.Event()

    @a_st.running
    async def run_a() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            a_cancelled.set()

    @b_st.command()
    async def cmd_b() -> str:
        return "b"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("a")
        await asyncio.sleep(0.1)
        await runtime.switch_state("b")
        await asyncio.sleep(0.1)
        assert a_cancelled.is_set()


@pytest.mark.asyncio
async def test_main_state_on_startup_called_once():
    """main_state 的 on_startup 只在 bootstrap 时调用一次."""
    chan = PyChannel(name="main")
    started = []

    @chan.build.startup
    async def on_start() -> None:
        started.append(1)

    async with chan.bootstrap() as runtime:
        pass
    assert len(started) == 1


@pytest.mark.asyncio
async def test_main_state_on_close_called_once():
    """main_state 的 on_close 在 channel 关闭时调用."""
    chan = PyChannel(name="main")
    closed = []

    @chan.build.close
    async def on_close_fn() -> None:
        closed.append(1)

    assert len(closed) == 0
    async with chan.bootstrap() as runtime:
        pass
    assert len(closed) == 1


# ============================================================
# State context / instruction 合并
# ============================================================


@pytest.mark.asyncio
async def test_state_context_messages_merged():
    """meta 包含 main_state 和 current_state 的 context messages."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")

    @chan.build.context_messages
    async def main_ctx() -> list[Message]:
        return [Message.new().with_content("main-ctx")]

    @work_st.context_messages
    async def work_ctx() -> list[Message]:
        return [Message.new().with_content("work-ctx")]

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert len(meta.context) >= 2

        texts = []
        for msg in meta.context:
            for c in msg.contents:
                from ghoshell_moss.message import Text
                texts.append(Text.from_content(c).text)
        assert "main-ctx" in texts
        assert "work-ctx" in texts


@pytest.mark.asyncio
async def test_state_instruction_from_main_state():
    """main_state 的 instruction 出现在 meta 中."""
    chan = PyChannel(name="main")

    @chan.build.instruction
    async def instr() -> str:
        return "main-instruction"

    async with chan.bootstrap() as runtime:
        meta = runtime.self_meta()
        assert meta.instruction == "main-instruction"


# ============================================================
# State children
# ============================================================


@pytest.mark.asyncio
async def test_current_state_children_appear_as_virtual():
    """current_state 的 children 在 virtual_sub_channels 中."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")
    sub = PyChannel(name="sub")

    work_st.import_channels(sub)

    @sub.build.command()
    async def sub_cmd() -> str:
        return "sub"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        virtual = runtime.virtual_sub_channels()
        assert "sub" in virtual


# ============================================================
# 边界情况
# ============================================================


@pytest.mark.asyncio
async def test_switch_to_same_state_noop():
    """切换到当前 state 是空操作."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")
    started = []

    @work_st.startup
    async def on_start() -> None:
        started.append(1)

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        assert len(started) == 1
        result = await runtime.switch_state("work")
        assert "already running" in result
        assert len(started) == 1


@pytest.mark.asyncio
async def test_switch_to_nonexistent_state():
    """切换到不存在的 state 返回错误消息."""
    chan = PyChannel(name="main")

    async with chan.bootstrap() as runtime:
        result = await runtime.switch_state("nonexistent")
        assert "not found" in result


@pytest.mark.asyncio
async def test_stop_when_no_current_state():
    """没有 current_state 时 stop_current_state 安全返回."""
    chan = PyChannel(name="main")

    async with chan.bootstrap() as runtime:
        result = await runtime.stop_current_state()
        assert "no current state" in result or result == ""


@pytest.mark.asyncio
async def test_states_meta_contains_state_list():
    """ChannelMeta 包含 states 字典和 current_state 名称."""
    chan = PyChannel(name="main")
    chan.new_state("idle", "idle state")
    chan.new_state("work", "working state")

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert "idle" in meta.states
        assert "work" in meta.states
        assert meta.states["idle"] == "idle state"
        assert meta.states["work"] == "working state"


@pytest.mark.asyncio
async def test_states_meta_current_state():
    """ChannelMeta.current_state 反映当前 state 名称."""
    chan = PyChannel(name="main")
    chan.new_state("work", "working")

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert meta.current_state == "work"


@pytest.mark.asyncio
async def test_is_dynamic_when_states_exist():
    """有 switchable states 时 channel 是 dynamic 的."""
    chan = PyChannel(name="main")
    assert not chan.build.is_dynamic()

    chan.new_state("work", "working")
    async with chan.bootstrap() as runtime:
        # 有 states 时 runtime 应该是 dynamic
        assert runtime.is_dynamic()


@pytest.mark.asyncio
async def test_switch_state_auto_registered_command():
    """有 states 时, switch_state 命令自动注册."""
    chan = PyChannel(name="main")
    chan.new_state("work", "working")

    async with chan.bootstrap() as runtime:
        # switch_state 和 stop_current_state 命令应自动可用
        all_cmds = runtime.own_commands()
        cmd_names = list(all_cmds.keys())
        assert "switch_state" in cmd_names


@pytest.mark.asyncio
async def test_stop_current_state_auto_registered_when_active():
    """有 current_state 时 stop_current_state 命令可用."""
    chan = PyChannel(name="main")
    chan.new_state("work", "working")

    async with chan.bootstrap() as runtime:
        # 没有 current_state 时 stop_current_state 不可用
        cmds_before = runtime.own_commands()
        assert "stop_current_state" not in cmds_before

        await runtime.switch_state("work")
        cmds_after = runtime.own_commands()
        assert "stop_current_state" in cmds_after


# ============================================================
# BaseStateChannel (非 Prime, 仅 Stateful)
# ============================================================


@pytest.mark.asyncio
async def test_base_state_channel_from_builder():
    """BaseStateChannel 包装一个 ChannelState, 可以直接 bootstrap."""
    builder = new_channel_state(name="base", description="a base state channel")

    @builder.command()
    async def greet() -> str:
        return "hello"

    chan = new_stateful_channel_from_main(builder)
    async with chan.bootstrap() as runtime:
        cmd = runtime.get_command("greet")
        assert cmd is not None
        assert await cmd() == "hello"


@pytest.mark.asyncio
async def test_base_state_channel_with_switchable_states():
    """BaseStateChannel 也可以注册和切换 state."""
    main_st = new_channel_state(name="root", description="root state")

    @main_st.command()
    async def root_cmd() -> str:
        return "root"

    chan = new_stateful_channel_from_main(main_st)
    alt_st = chan.new_state("alt", "alternative")

    @alt_st.command()
    async def alt_cmd() -> str:
        return "alt"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("alt")
        assert runtime.get_command("root_cmd") is not None
        assert runtime.get_command("alt_cmd") is not None
        assert await runtime.get_command("alt_cmd")() == "alt"


# ============================================================
# switch_state / stop_current_state 命令通过 execute_task 执行
# ============================================================


@pytest.mark.asyncio
async def test_switch_state_via_task():
    """switch_state 可以通过 push_task 执行."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")

    @work_st.command()
    async def work_cmd() -> str:
        return "work"

    async with chan.bootstrap() as runtime:
        task = runtime.create_command_task("switch_state", args=("work",))
        runtime.push_task(task)
        result = await task
        assert "started current state" in result
        assert runtime.get_command("work_cmd") is not None


@pytest.mark.asyncio
async def test_stop_current_state_via_task():
    """stop_current_state 可以通过 push_task 执行."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")

    @work_st.command()
    async def work_cmd() -> str:
        return "work"

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        task = runtime.create_command_task("stop_current_state")
        runtime.push_task(task)
        result = await task
        assert "stopped" in result
        assert runtime.get_command("work_cmd") is None


# ============================================================
# auto-switch to '' state on startup
# ============================================================


@pytest.mark.asyncio
async def test_auto_switch_empty_string_state_on_startup():
    """on_startup 中 '' state 自动激活."""
    chan = PyChannel(name="main")
    default_st = chan.new_state("", "default state")

    @default_st.command()
    async def default_cmd() -> str:
        return "default"

    async with chan.bootstrap() as runtime:
        # on_startup 已触发, '' state 应自动激活
        cmd = runtime.get_command("default_cmd")
        assert cmd is not None
        assert await cmd() == "default"


# ============================================================
# on_close 对 current_state 的处理 (疑似 bug 验证)
# ============================================================


@pytest.mark.asyncio
async def test_current_state_on_close_called_on_channel_shutdown():
    """channel 关闭时, current_state.on_close 被调用."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")
    current_closed = []

    @work_st.close
    async def work_close() -> None:
        current_closed.append(1)

    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        # current_state 处于激活状态

    # channel 关闭后, current_state.on_close 被调用
    assert len(current_closed) == 1


# ============================================================
# switch_state command 注册 bug 验证
# ============================================================


@pytest.mark.asyncio
async def test_switch_state_command_is_correct_function():
    """验证 switch_state 命令注册的是正确的函数."""
    chan = PyChannel(name="main")
    work_st = chan.new_state("work", "working")

    @work_st.command()
    async def work_cmd() -> str:
        return "work"

    async with chan.bootstrap() as runtime:
        switch_cmd = runtime.get_own_command("switch_state")
        assert switch_cmd is not None
        result = await switch_cmd("work")
        assert "started current state" in result
        assert runtime.get_command("work_cmd") is not None


# ============================================================
# state 的 bootstrap 方法 (IoC 集成)
# ============================================================


@pytest.mark.asyncio
async def test_state_bootstrap_called_for_main_state():
    """main_state.bootstrap() 在容器准备阶段被调用."""
    chan = PyChannel(name="main")
    bootstrapped = []

    # 使用 with_binding 触发 bootstrap 路径
    class Foo:
        def __init__(self, val: str):
            self.val = val

    chan.build.with_binding(Foo, Foo("bar"))

    async with chan.bootstrap() as runtime:
        foo = runtime.container.get(Foo)
        assert foo is not None
        assert foo.val == "bar"


# ============================================================
# ChannelModule Protocol & with_module 测试
# ============================================================


def test_py_channel_builder_satisfies_module_protocol():
    """new_channel_state 自动满足 ChannelModule Protocol."""
    builder = new_channel_state(name="test")
    assert hasattr(builder, 'name')
    assert hasattr(builder, 'own_commands')


@pytest.mark.asyncio
async def test_with_module_registration():
    """with_module 注册 module 到 channel."""
    chan = new_prime_channel(name="main")
    module = new_channel_state(name="speech")

    @module.command()
    async def speak(text: str) -> str:
        return f"said: {text}"

    chan.with_module(module)
    assert "speech" in chan.modules()
    assert chan.modules()["speech"] is module


@pytest.mark.asyncio
async def test_module_commands_appear_on_runtime():
    """module 的命令通过 runtime 可见可执行."""
    chan = new_prime_channel(name="main")
    mod = new_channel_state(name="tts")

    @mod.command()
    async def say(text: str) -> str:
        return f"said: {text}"

    # state as mod? bad smell
    chan.with_module(mod)
    async with chan.bootstrap() as runtime:
        cmd = runtime.get_command("say")
        assert cmd is not None
        result = await cmd("hello")
        assert result == "said: hello"


@pytest.mark.asyncio
async def test_module_commands_coexist_with_main():
    """module 命令和 main_state 命令同时存在."""
    chan = PyChannel(name="main")

    @chan.build.command()
    async def main_cmd() -> str:
        return "main"

    mod = new_channel_state(name="tts")

    @mod.command()
    async def tts_cmd() -> str:
        return "tts"

    chan.with_module(mod)
    async with chan.bootstrap() as runtime:
        assert await runtime.get_command("main_cmd")() == "main"
        assert await runtime.get_command("tts_cmd")() == "tts"


@pytest.mark.asyncio
async def test_multiple_modules_cumulative():
    """多个 module 同时激活，所有命令都可用."""
    chan = PyChannel(name="main")
    mod_a = new_channel_state(name="mod_a")

    @mod_a.command()
    async def cmd_a() -> str:
        return "a"

    mod_b = new_channel_state(name="mod_b")

    @mod_b.command()
    async def cmd_b() -> str:
        return "b"

    chan.with_module(mod_a).with_module(mod_b)
    async with chan.bootstrap() as runtime:
        assert await runtime.get_command("cmd_a")() == "a"
        assert await runtime.get_command("cmd_b")() == "b"


@pytest.mark.asyncio
async def test_module_command_priority_main_wins():
    """module 和 main_state 同名，main_state 优先。"""
    chan = PyChannel(name="main")

    @chan.build.command()
    async def foo() -> str:
        return "main"

    mod = new_channel_state(name="mod")

    @mod.command()
    async def foo() -> str:
        return "module"

    chan.with_module(mod)
    async with chan.bootstrap() as runtime:
        assert await runtime.get_command("foo")() == "main"


@pytest.mark.asyncio
async def test_module_command_priority_over_current_state():
    """module 命令优先于 current_state 同名命令。"""
    chan = PyChannel(name="main")

    mod = new_channel_state(name="base_tts")

    @mod.command()
    async def say() -> str:
        return "module"

    work_st = chan.new_state("work", "working")

    @work_st.command()
    async def say() -> str:
        return "state"

    chan.with_module(mod)
    async with chan.bootstrap() as runtime:
        await runtime.switch_state("work")
        assert await runtime.get_command("say")() == "module"


@pytest.mark.asyncio
async def test_module_lifecycle_on_startup():
    """module.on_startup() 在 channel bootstrap 时被调用。"""
    chan = PyChannel(name="main")
    mod = new_channel_state(name="speech")
    started = []

    @mod.startup
    async def on_start() -> None:
        started.append(1)

    chan.with_module(mod)
    assert len(started) == 0
    async with chan.bootstrap() as runtime:
        # on_startup 已在 bootstrap 时激活
        pass
    assert len(started) == 1


@pytest.mark.asyncio
async def test_module_lifecycle_on_close():
    """module.on_close() 在 channel 关闭时被调用。"""
    chan = PyChannel(name="main")
    mod = new_channel_state(name="speech")
    closed = []

    @mod.close
    async def on_close_fn() -> None:
        closed.append(1)

    chan.with_module(mod)
    assert len(closed) == 0
    async with chan.bootstrap() as runtime:
        pass
    assert len(closed) == 1


@pytest.mark.asyncio
async def test_module_context_messages_merged():
    """module 的 context messages 被合并到 meta 中。"""
    chan = PyChannel(name="main")
    mod = new_channel_state(name="sensor")

    @chan.build.context_messages
    async def main_ctx() -> list[Message]:
        return [Message.new().with_content("main")]

    @mod.context_messages
    async def mod_ctx() -> list[Message]:
        return [Message.new().with_content("sensor-data")]

    chan.with_module(mod)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert len(meta.context) >= 2
        texts = []
        for msg in meta.context:
            for c in msg.contents:
                from ghoshell_moss.message import Text
                texts.append(Text.from_content(c).text)
        assert "main" in texts
        assert "sensor-data" in texts


@pytest.mark.asyncio
async def test_channel_meta_includes_modules():
    """ChannelMeta.modules 记录 module name 列表（for debug）。"""
    chan = PyChannel(name="main")
    chan.with_module(new_channel_state(name="mod_a"))
    chan.with_module(new_channel_state(name="mod_b"))

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert "mod_a" in meta.modules
        assert "mod_b" in meta.modules


@pytest.mark.asyncio
async def test_module_on_base_state_channel():
    """BaseStateChannel 也支持 with_module。"""
    main_st = new_channel_state(name="root")
    chan = new_stateful_channel_from_main(main_st)
    mod = new_channel_state(name="extra")

    @mod.command()
    async def extra_cmd() -> str:
        return "extra"

    chan.with_module(mod)
    async with chan.bootstrap() as runtime:
        assert await runtime.get_command("extra_cmd")() == "extra"


@pytest.mark.asyncio
async def test_module_state_independent():
    """module 不受 state 切换影响（module 始终激活）。"""
    chan = PyChannel(name="main")
    mod = new_channel_state(name="core")

    @mod.command()
    async def core_cmd() -> str:
        return "core"

    chan.with_module(mod)
    work_st = chan.new_state("work", "working")

    @work_st.command()
    async def work_cmd() -> str:
        return "work"

    async with chan.bootstrap() as runtime:
        # module 命令在无 state 时可用
        assert await runtime.get_command("core_cmd")() == "core"

        # 切换到 state 后 module 命令仍然可用
        await runtime.switch_state("work")
        assert await runtime.get_command("core_cmd")() == "core"
        assert await runtime.get_command("work_cmd")() == "work"

        # 停止 state 后 module 命令仍然可用
        await runtime.stop_current_state()
        assert await runtime.get_command("core_cmd")() == "core"
        assert runtime.get_command("work_cmd") is None


@pytest.mark.asyncio
async def test_protocol_duck_typing():
    """任意满足接口的对象都是 ChannelModule，不需要继承。"""
    from ghoshell_moss.core.concepts.command import PyCommand, Command

    class CustomModule:
        def __init__(self):
            self._started = False

        def name(self) -> str:
            return "custom"

        def own_commands(self) -> dict[str, Command]:
            async def foo() -> str:
                return "custom-foo"

            return {"foo": PyCommand(func=foo)}

        async def on_startup(self) -> None:
            self._started = True

    mod = CustomModule()
    # 验证协议方法存在
    assert hasattr(mod, 'name')
    assert hasattr(mod, 'own_commands')
    assert hasattr(mod, 'on_startup')

    chan = PyChannel(name="main")
    chan.with_module(mod)
    async with chan.bootstrap() as runtime:
        cmd = runtime.get_command("foo")
        assert cmd is not None
        assert await cmd() == "custom-foo"
    assert mod._started


@pytest.mark.asyncio
async def test_module_no_commands():
    """module 没有命令时也是有效的。"""
    chan = PyChannel(name="main")
    mod = new_channel_state(name="empty")

    @mod.context_messages
    def sync_context_message_case():
        return "hello"

    @chan.build.context_messages
    async def context_messages():
        return "world"

    chan.with_module(mod)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert "empty" in meta.modules
        assert len(meta.context) == 2
