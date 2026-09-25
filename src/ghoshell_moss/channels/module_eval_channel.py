"""Module 求值 — 可复用的 module 级别有状态运行时 | 元能力 | alpha

把 Python 模块 .py 文件包装为 live runtime——子进程执行, 模型通过 exec/aexec
直接在持久化命名空间写代码, 副作用持续累积 (有状态, 无 undo)。域对象 (浏览器/
ROS 节点/DB 连接) 在子进程启动时物化, 跨调用存活。

命令面收敛为 ``exec`` / ``aexec`` / ``history`` (无 vars/api——builtins 放开后
模型用 ``dir()``/``inspect`` 自行反射)。builtins 不设白名单, domain 源码的 import
即授权边界, 由 instruction 摆给模型。

两种形态:
- ``new_module_eval_channel`` — 单 module, channel 自管生命周期
- ``new_sandbox_hub_channel`` — 父 channel 治理 N 个 module 子 channel
  (``open``/``close``/``list`` + virtual_children 闭包)

Example:
    from ghoshell_moss.channels.module_eval_channel import new_sandbox_hub_channel
    hub = new_sandbox_hub_channel(subprocesses, root="./domains")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.tools.module_eval import ModuleEval

if TYPE_CHECKING:
    from ghoshell_moss.core.concepts.channel import Channel

__all__ = [
    "new_module_eval_channel",
    "new_sandbox_hub_channel",
    "sandbox_hub_channel_factory",
]

_CHANNEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _is_valid_channel_name(name: str) -> bool:
    return bool(_CHANNEL_NAME_RE.fullmatch(name))


def _instruction(eval: ModuleEval) -> str:
    return (
        "A live, persistent Python runtime (reusable module-level stateful runtime). "
        "Every exec mutates this same environment — no undo; variables and imports "
        "accumulate across calls. The domain objects below are already materialized "
        "and stay alive across your calls.\n\n"
        f"{eval.source}"
    )


def _build_module_commands(chan: MutableChannel, eval: ModuleEval) -> None:
    @chan.build.instruction
    def instruction() -> str:
        return _instruction(eval)

    @chan.build.command(name="exec", always_observe=True)
    async def exec_code(text__: str, timeout: float = 30.0) -> str:
        """Execute Python code in the live runtime.  text__: code to run.

        Use CDATA open-close tags to protect code from CTML parsing:
            <{name}:exec><![CDATA[
            x = 1 + 2
            print(x)
            ]]></{name}:exec>

        Variables persist across calls.  Blocks until the result returns; raises
        on timeout.  :param timeout: seconds before raising TimeoutError.
        """
        return await eval.exec(text__, timeout=timeout)

    @chan.build.command(name="aexec", always_observe=False)
    async def exec_async(text__: str) -> str:
        """Run Python code in the background (fire-and-forget).  text__: code.

        Returns immediately with a queue id.  The result is recorded into history
        when it finishes — read it with the ``history`` command.  Use for long or
        parallel work you don't want to block on.
        """
        rid = await eval.aexec(text__)
        return f"queued: {rid}"

    @chan.build.command(name="history", always_observe=True)
    def history(n: int = 10) -> str:
        """Recent executed commands + result summaries (in-memory, oldest first).

        :param n: how many entries to show.
        """
        return eval.history(n)


def new_module_eval_channel(
    module_path: str,
    *,
    subprocesses: Subprocesses | None = None,
    channel_name: str | None = None,
    description: str | None = None,
    history_size: int = 20,
) -> MutableChannel:
    """创建单 module 的 live runtime channel — 自管子进程生命周期。

    :param module_path: .py 文件路径 (父进程只读源码, 不 import)
    :param subprocesses: 传入则用 subprocesses.execute() 托管子进程
    :param channel_name: CTML 标签名, 默认取文件 stem
    :param description: 覆盖默认 description
    """
    eval = ModuleEval(module_path, subprocesses=subprocesses, history_size=history_size)
    name = channel_name or eval.module_name
    desc = description or (
        f"Live Python runtime — {name}. exec/aexec mutate a persistent "
        f"environment. Source: {module_path}"
    )

    chan = new_channel(name=name, description=desc)
    _build_module_commands(chan, eval)

    @chan.build.startup
    async def on_startup():
        await eval.start()

    @chan.build.close
    async def cleanup():
        await eval.shutdown()

    return chan


# ---------------------------------------------------------------------------
# Hub — 父 channel 治理 N 个 module 子 channel
# ---------------------------------------------------------------------------


class ModuleSandbox:
    """本源对象: 持有 N 个 ``ModuleEval``, open/close 治理其子进程生命周期。

    构建方 / GUI 持有实例直接调用公开方法; 模型通过 hub channel 的
    ``open``/``close``/``list`` 命令操作。每个已打开 module 投影为一个可执行
    虚拟子 channel (exec/aexec/history)。
    """

    def __init__(
        self,
        *,
        subprocesses: Subprocesses | None,
        root: str,
        name: str = "sandbox",
        description: str = "",
        history_size: int = 20,
    ):
        self._subprocesses = subprocesses
        self._root = Path(root)
        self._name = name
        self._description = description or "Sandbox — open/close live module runtimes"
        self._history_size = history_size
        self._modules: dict[str, ModuleEval] = {}
        self._channels: dict[str, Channel] = {}

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def modules(self) -> dict[str, ModuleEval]:
        return dict(self._modules)

    def children(self) -> dict[str, Channel]:
        return dict(self._channels)

    def available_domains(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(p.stem for p in self._root.glob("*.py"))

    async def open(self, name: str) -> str:
        if not _is_valid_channel_name(name):
            return f"[{self._name}:{name}] invalid name"
        if name in self._modules:
            return f"[{self._name}:{name}] already open"
        path = self._root / f"{name}.py"
        if not path.is_file():
            avail = self.available_domains()
            suffix = f" Available: {', '.join(avail)}" if avail else ""
            return f"[{self._name}:{name}] no domain `{name}.py` under {self._root}.{suffix}"

        eval = ModuleEval(str(path), subprocesses=self._subprocesses, history_size=self._history_size)
        try:
            await eval.start()
        except Exception as e:
            await eval.shutdown()
            return f"[{self._name}:{name}] start failed: {e}"

        self._modules[name] = eval
        self._channels[name] = _new_module_child_channel(eval, name)
        return f"[{self._name}:{name}] opened"

    async def close(self, name: str) -> str:
        eval = self._modules.pop(name, None)
        self._channels.pop(name, None)
        if eval is None:
            return f"[{self._name}:{name}] not open"
        await eval.shutdown()
        return f"[{self._name}:{name}] closed"

    async def close_all(self) -> None:
        for name in list(self._modules):
            await self.close(name)

    def list_modules(self) -> str:
        lines = ["### Modules"]
        for name in self._modules:
            lines.append(f"[+] {name}")
        avail = [d for d in self.available_domains() if d not in self._modules]
        if avail:
            lines.append("available: " + ", ".join(avail))
        elif not self._modules:
            lines.append("(none open)")
        return "\n".join(lines)


def _new_module_child_channel(eval: ModuleEval, name: str) -> Channel:
    chan = new_channel(name=name, description=f"Live Python runtime — {name}")
    _build_module_commands(chan, eval)
    return chan


def new_sandbox_hub_channel(
    subprocesses: Subprocesses | None,
    root: str,
    *,
    name: str = "sandbox",
    description: str = "",
    history_size: int = 20,
) -> Channel:
    """构建 Sandbox Hub Channel — 父 channel 治理多个 module 运行时。

    :param subprocesses: Subprocesses 实例 (spawn 子进程); None 用裸 asyncio
    :param root: domain .py 文件目录; ``open(name)`` 解析 ``root/{name}.py``
    :param name: channel 名称 (子 channel 挂在它下面)
    :param description: 覆盖默认 description
    """
    hub = ModuleSandbox(
        subprocesses=subprocesses,
        root=root,
        name=name,
        description=description,
        history_size=history_size,
    )
    chan = new_channel(name=name, description=hub.description())

    @chan.build.virtual_children
    def _children() -> dict[str, Channel]:
        return hub.children()

    @chan.build.close
    async def _close() -> None:
        await hub.close_all()

    @chan.build.notice
    async def _notice() -> str:
        opened = list(hub.modules())
        lines = [f"{len(opened)} module(s) open"] + [f"  + {n}" for n in opened]
        avail = [d for d in hub.available_domains() if d not in hub.modules()]
        if avail:
            lines.append("available: " + ", ".join(avail))
        return "\n".join(lines)

    @chan.build.command(name="list", always_observe=True)
    async def list_modules() -> str:
        """List open module runtimes and the domains available to open."""
        return hub.list_modules()

    @chan.build.command(name="open", always_observe=True)
    async def open(name: str) -> str:
        """Open a domain module as an executable sub-channel (exec/aexec/history).

        :param name: domain file stem under the domains dir (e.g. ``playwright``).
        """
        return await hub.open(name)

    @chan.build.command(name="close", always_observe=True)
    async def close(name: str) -> str:
        """Close an open module runtime and shut down its subprocess."""
        return await hub.close(name)

    return chan


def sandbox_hub_channel_factory(
    root: str,
    *,
    name: str = "sandbox",
    description: str = "",
    history_size: int = 20,
) -> ChannelFactory:
    """返回 ChannelFactory: 容器就绪后从 IoC 取 Subprocesses 构建 hub channel。"""

    def _factory(container) -> Channel:
        subprocesses = container.force_fetch(Subprocesses)
        return new_sandbox_hub_channel(
            subprocesses,
            root,
            name=name,
            description=description,
            history_size=history_size,
        )

    return _factory
