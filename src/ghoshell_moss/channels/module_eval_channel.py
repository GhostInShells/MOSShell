"""Module 求值容器 — subprocess sandbox | 系统管理 | alpha

将 Python 模块 .py 文件包装为 CTML Channel — 子进程 Sandbox 执行，
模型通过 exec 命令直接在持久化命名空间中写代码，源码即 instruction。

依赖 tools.ModuleEval 管理子进程 + JSON-line 协议。

Example:
    from ghoshell_moss.channels.module_eval_channel import new_module_eval_channel
    chan = new_module_eval_channel("./my_domain.py")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ghoshell_moss.core.blueprint.channel_builder import new_channel, MutableChannel
from ghoshell_moss.tools.module_eval import ModuleEval

if TYPE_CHECKING:
    from ghoshell_moss.core.blueprint.matrix import Matrix

__all__ = ["new_module_eval_channel"]


def new_module_eval_channel(
    module_path: str,
    *,
    matrix: Matrix | None = None,
    channel_name: str | None = None,
    description: str | None = None,
) -> MutableChannel:
    """创建 ModuleEvalChannel — 子进程 sandbox eval。

    *module_path* 指向一个 .py 文件。该文件的源码成为 channel instruction
    (Code as Prompt)，子进程编译执行 import 和模块级代码，
    AI 在 SANDBOX_BUILTINS 下 exec 后续代码。

    :param module_path: .py 文件路径（父进程只读源码，不 import）
    :param matrix: 传入则用 matrix.processes.execute()（子进程由 MOSS Subprocesses 托管生命周期）
    :param channel_name: CTML 标签名，默认取文件 stem
    :param description: 覆盖默认 description
    """
    eval = ModuleEval(module_path, matrix=matrix)
    name = channel_name or eval.module_name
    desc = description or (
        f"Module eval — exec code in persistent {name} namespace. "
        f"Source file: {module_path}"
    )

    chan = new_channel(name=name, description=desc)

    @chan.build.instruction
    def instruction() -> str:
        return eval.instruction

    @chan.build.startup
    async def on_startup():
        await eval.start()

    @chan.build.close
    async def cleanup():
        await eval.shutdown()

    @chan.build.command(name="exec", always_observe=True)
    async def exec_code(text__: str) -> str:
        """Execute Python code in persistent namespace.  text__: code string.

        Use CDATA open-close tags to protect code from CTML parsing:
            <{name}:exec><![CDATA[
            x = 1 + 2
            print(x)
            ]]></{name}:exec>

        Variables persist across calls.  print() output captured.
        Assign __result__ to return a value explicitly.
        """
        return await eval.exec(text__)

    @chan.build.command(name="vars", always_observe=True)
    async def list_vars() -> str:
        """List namespace contents — name → type mapping."""
        return await eval.vars()

    @chan.build.command(name="api", always_observe=True)
    async def reflect_api(name: str = "") -> str:
        """Reflect on namespace objects.  *name*: variable name for detail.

        Without name, returns full module interface (source + imported attrs).
        With name, returns detail for that object (source for imports,
        signature + docstring + methods for locals).
        """
        return await eval.api(name or None)

    return chan
