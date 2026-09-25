"""Runtime 超级 debug — 编译临时 module, 手动 await 其 main(container) | 集成 | alpha

给模型一把"在运行时调试 MOSS 自身"的直通钥匙: 把一段 Python 源码编译成临时
module, 拿到里面的 ``main`` 函数, 直接调用 ``main(container)`` —— 其中 ``container``
是**当前运行现场里的 live IoC 容器**。

模型拿到的是真正活着的运行时对象 (matrix / session / configs / resources ...),
通过 IoC 直达, 而不是磁盘上的源码快照。这是"对自身 debug"的最强形态:
从 Shell 内部看 Shell 自己。

channel 先注入一段**代码上文** (prologue, 含 ``import asyncio``), 模型的输入补齐
下文再整体编译。这样编译源码里天然有 ``asyncio``, 模型可以在 ``async def main``
里用 ``asyncio.to_thread`` 包住阻塞代码, 避免卡死事件循环。

``main`` **必须是** ``async def`` —— 同步函数会阻塞事件循环, 通道直接拒绝。

函数签名约定::

    async def main(container):
        # container 就是现场 IoC
        m = container.force_fetch(Matrix)
        ...
        return {"ok": True}

调用时用 CDATA 包裹源码, 防止 CTML 解析破坏:

    <runtime_debug:debug><![CDATA[
    async def main(container):
        from ghoshell_moss.core.blueprint.matrix import Matrix
        m = container.force_fetch(Matrix)
        print("mode =", m.env.mode_name)
        return {"ok": True}
    ]]></runtime_debug:debug>

每次 debug 调用都编译一个**独立临时 module** (不残留命名空间状态), 模型写什么编
什么, 编完即弃。执行正确性是模型自己的责任: 该 channel 只提供"编译 + 注入
container + 调用 main + 汇报 stdout/返回值/异常"的骨架。

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.runtime_debug_channel import build_runtime_debug_channel

    main = new_shell_main_channel()
    main.import_channels(build_runtime_debug_channel())
"""

from __future__ import annotations

import asyncio
import inspect
import traceback as _traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.codex.compiler import Compiler

__all__ = [
    "new_runtime_debug_channel",
    "build_runtime_debug_channel",
]

_MAIN = "main"
# 代码上文: 让编译源码里含有 asyncio, 模型在函数体里可用 asyncio.to_thread.
_PROLOGUE = "import asyncio\n"


def _compile(text: str):
    """把上文 + 模型补的正文编译成临时 module. 失败抛异常, 由调用方格式化."""
    return Compiler(
        source=_PROLOGUE + text,
        filename="<moss_runtime_debug>",
    ).compiled


def _render_result(out: str, ret: Any) -> str:
    """把 stdout + 返回值渲染成 command 返回值."""
    parts: list[str] = []
    if out.strip():
        parts.append("--- stdout ---\n" + out.rstrip())
    if ret is not None:
        parts.append("--- result ---\n" + repr(ret))
    if not parts:
        return '(main returned None, no stdout)'
    return "\n".join(parts)


def _render_error(title: str, exc: Exception) -> str:
    """渲染运行/编译错误. 过滤掉本 channel 与 compiler 的内部 frame, 只留模型代码的栈."""
    tb = _traceback.extract_tb(exc.__traceback__)
    excluded = set()
    for mod in (Compiler,):
        try:
            excluded.add(inspect.getfile(mod))
        except TypeError:
            pass
    try:
        excluded.add(__file__)
    except NameError:
        pass
    frames = [f for f in tb if f.filename not in excluded]
    body = ''.join(_traceback.format_list(frames))
    last = _traceback.format_exception_only(type(exc), exc)[-1]
    head = f"{title}: {type(exc).__name__}: {exc}"
    return head + "\n" + body + last


def new_runtime_debug_channel(
    *,
    name: str = "runtime_debug",
    description: str | None = None,
) -> MutableChannel:
    """创建 Runtime super-debug channel — 编译临时 module, await/main 其 container.

    :param name: CTML 标签名, 默认 ``runtime_debug``.
    :param description: 覆盖默认描述.
    """
    desc = description or (
        "Runtime super-debug — compile a temp Python module, call its "
        "`main(container)` with the live IoC injected. Debug MOSS from inside itself."
    )

    chan = new_channel(name=name, description=desc)

    @chan.build.instruction
    def instruction() -> str:
        return (
            "## runtime_debug channel\n"
            "Debug the LIVE MOSS runtime you are running on. Each call compiles a "
            "temporary Python module and calls its `main(container)` function — then "
            "reports stdout, the return value, or any error.\n"
            "\n"
            "Write the source as `text__` wrapped in CDATA (so CTML parsing does not "
            "break it). `container` is the live IoC container; call "
            "`container.force_fetch(Matrix)` to reach matrix, session, configs, resources.\n"
            "\n"
            "Signature — `main` MUST be `async def` (a plain def would block the "
            "event loop, so it is rejected):\n"
            "  - write `async def main(container):` — the container is the live IoC.\n"
            "  - `print(...)` inside `main` is captured as stdout.\n"
            "  - `return value` becomes the command result (shown as -- result --).\n"
            "  - `asyncio` is already imported and available in scope. If `main` does "
            "    blocking work, wrap it with `asyncio.to_thread(...)`.\n"
            "\n"
            "Every call is hermetic: a fresh temp module, no namespace state carried "
            "over. A power tool — you are responsible for what a single invocation does."
        )

    @chan.build.command(name="debug", always_observe=True)
    async def run_debug(text__: str) -> str:
        """Compile `text__` (+ prologue) into a temp module and run its `main(container)`.

        `main(container)` receives the live IoC. Captures `print` (stdout) and the
        return value; reports exceptions with a filtered traceback. Every call is a
        fresh temp module — no state persists.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx

        container = ChannelCtx.container()

        # 1) 编译阶段 —— Compiler 在 compile_soon 时直接编译, 语法错误在此抛出.
        try:
            module = _compile(text__)
        except Exception as e:
            return _render_error("COMPILE ERROR", e)

        # 2) 取 main —— 只允许 async def, 同步函数会阻塞事件循环.
        main_fn = module.__dict__.get(_MAIN)
        if main_fn is None:
            return (
                f"COMPILE ERROR: no `{_MAIN}` defined in the source. "
                f"You must define async def {_MAIN}(container)."
            )
        if not inspect.iscoroutinefunction(main_fn):
            return (
                f"COMPILE ERROR: `{_MAIN}` must be `async def`. "
                f"A plain def would block the event loop — mark it `async def` and "
                f"wrap blocking calls with `asyncio.to_thread(...)`."
            )

        # 3) 运行阶段 —— await main(container), 捕获 stdout, 兜住运行时错误.
        buffer = StringIO()
        with redirect_stdout(buffer):
            try:
                ret = await main_fn(container)
            except Exception as e:
                return _render_error("RUN ERROR", e)

        return _render_result(buffer.getvalue(), ret)

    return chan


def build_runtime_debug_channel(
    *,
    name: str = "runtime_debug",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂: 从容器解析并返回 runtime_debug channel 的 ChannelFactory.

    runtime_debug 在 command 内部用 ``ChannelCtx.container()`` 现场取 IoC,
    工厂无需依赖 Matrix; 保留 container 参数以符合 :data:`ChannelFactory` 契约.
    """
    def factory(_container: IoCContainer) -> MutableChannel:
        return new_runtime_debug_channel(name=name, description=description)
    return factory
