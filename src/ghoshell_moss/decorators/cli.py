"""cli decorator — 把一个真实 CLI 命令包成普通可调用函数 (code-as-prompt).

一句话: ``@cli("echo")`` 把命令行工具变成 ``async (arguments: str) -> tuple[int, str, str]``
的普通函数. 函数签名即工具签名, 反射穿透到模型.

三层职责:
- 被装饰函数本体: **纯声明** — 签名 + docstring 是 code-as-prompt 的全部, 运行时不被调用.
  行为通过注册的参数实现:
  - ``input_filter``: 入参过滤 (argv 列表 → argv 列表, 可抛异常拒绝). 结构性, 第一公民.
  - ``output_processor``: 出参加工 (结果三元组 → 结果三元组, 形状不变). 可选便利,
    展示性格式化留给 seat (channel / 调用方).
- 装饰器本身: 边界约束 — 前缀 argv 展开, 工具名, ``-h``/``--help`` 拦截, 反射穿透.
- 执行层 (decorator 内部闭包): 从 project-level IoC 或注入的 facade 取 SubprocessFacade,
  exec 模式 (无 shell) 运行 ``prefix + shlex.split(arguments)``, 等退出并排空输出,
  返回 (code, stdout, stderr). 支持 ``cwd`` / ``timeout`` (超时优雅停止, 返回 124).

已知约束:
- 前缀必须是 argv 形式, argv[0] = 可执行文件. 传字符串用 shlex 拆成 argv.
- ``python -m moss`` 目前不合法: ghoshell_moss 没有 __main__.py, ``moss`` 是
  console script (ghoshell_moss.cli:main_entry). 前缀用 ``["moss", ...]`` 或
  ``[sys.executable, "-m", "<importable module>"]``.

依赖惰性加载: SubprocessFacade / Project 等只在首次调用工具时才 import 并从
project IoC 取, 模块 import 阶段零副作用 — 沙箱里 import 是授权, 工具的
宿主侧执行依赖不该成为 agent 编译期的 import 面.

per-call 可变配置 (cwd/timeout/facade) 没有参数空间 — 这是定义期绑定, 固定配置是
常态. 若工具需要 request-scoped 状态 + 回调, 它已超出 decorator 承载力, 应升级为
接口/类 (见 FEATURE.md cli-decorator 的 ctx 剪枝决策).
"""

from __future__ import annotations

import asyncio
import functools
import shlex
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

__all__ = ["cli", "CliResult"]

CliResult = tuple[int, str, str]
"""工具返回约定: (exit_code, stdout, stderr)."""

CliCallable = Callable[[str], Awaitable[CliResult]]
"""装饰后工具的签名: 调用方传原始参数字符串, 异步拿到结果三元组."""

Help = str | Callable[[], str]
"""``-h``/``--help`` 拦截的返回内容. str 静态返回; callable 每次调用时求值."""

InputFilter = Callable[[list[str]], list[str]]
"""入参过滤: 收到 shlex 拆好的 argv, 返回真正执行的 argv, 可抛异常拒绝调用."""

OutputProcessor = Callable[[CliResult], CliResult]
"""出参加工: 收到 (code, stdout, stderr), 返回形状不变的三元组 (规范化/注解)."""


def cli(
        prefix: str | Sequence[str],
        *,
        name: str = "",
        help: Help | None = None,
        facade: Any = None,
        cwd: str | Path | Callable[[], str | Path] | None = None,
        timeout: float | None = None,
        input_filter: InputFilter | None = None,
        output_processor: OutputProcessor | None = None,
) -> Callable[[Callable[[str], Any]], CliCallable]:
    """把命令行工具包成普通函数.

    :param prefix: 命令前缀. argv[0] 必须是可执行文件.
        字符串用 shlex 拆成 argv (``"python -m pytest"`` → ``["python", "-m", "pytest"]``).
    :param name: 工具名. 默认取被装饰函数名.
    :param help: ``-h``/``--help`` 的拦截返回. 提供后这两个参数不 spawn 子进程,
        直接返回 ``(0, help 内容, "")``.
    :param facade: 注入的 SubprocessFacade. None 时首次调用惰性取 project IoC 单例.
    :param cwd: spawn 工作目录. None 时用 facade 默认. 传 callable 时每次调用惰性求值
        (返回 str/Path) — 工具可定义期绑定一个惰性解析的 root (如 project root).
    :param timeout: 秒. None 不限制; 超时优雅停止 (SIGINT→SIGKILL), 返回 (124, 已捕获输出, 说明).
    :param input_filter: 入参过滤, 见 InputFilter.
    :param output_processor: 出参加工, 见 OutputProcessor.

    调度顺序: shlex split → help 拦截 (单独成参) → input_filter → exec → output_processor.
    """
    argv = shlex.split(prefix) if isinstance(prefix, str) else list(prefix)
    if not argv:
        raise ValueError("cli: prefix must not be empty")

    resolved_facade = facade

    async def _execute(args: list[str], *, sub_name: str) -> CliResult:
        """exec 模式跑 args, 等退出并排空输出.

        ``wait()`` 之后必须先 ``wait_drained()`` 再读 stdout/stderr — 否则 drain
        协程可能还没读到 EOF, 输出会静默丢失 (短命令尤其明显).
        超时路径必须 ``stop()`` 兜底, 否则子进程成孤儿.
        """
        nonlocal resolved_facade
        if resolved_facade is None:
            from ghoshell_moss.core.blueprint.project import Project
            project = Project.discover()
            project.bootstrap()
            from ghoshell_moss.contracts.subprocesses import SubprocessFacade
            resolved_facade = project.container.force_fetch(SubprocessFacade)
        from ghoshell_moss.contracts.subprocesses import CaptureSpec
        spawn_cwd = cwd() if callable(cwd) else cwd
        proc = await resolved_facade.execute(
            *args, name=sub_name, cwd=spawn_cwd,
            capture=CaptureSpec(buffer_lines=200),
        )
        try:
            if timeout is not None:
                await asyncio.wait_for(proc.process.wait(), timeout=timeout)
            else:
                await proc.process.wait()
        except asyncio.TimeoutError:
            await proc.stop()
            out = proc.output
            if out is not None:
                await out.wait_drained()
            return (124, out.stdout() if out else "", f"[timeout after {timeout}s, stopped]")
        out = proc.output
        if out is not None:
            await out.wait_drained()
        return (
            proc.process.returncode or 0,
            out.stdout() if out else "",
            out.stderr() if out else "",
        )

    def decorator(func: Callable[[str], Any]) -> CliCallable:
        tool_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(arguments: str = "") -> CliResult:
            try:
                args = shlex.split(arguments or "")
            except ValueError as e:
                raise ValueError(
                    f"cli {tool_name}: cannot parse arguments {arguments!r}: {e}"
                ) from e
            if help is not None and args in (["-h"], ["--help"]):
                text = help() if callable(help) else help
                return (0, text, "")
            if input_filter is not None:
                args = input_filter(args)
            result = await _execute([*argv, *args], sub_name=tool_name)
            if output_processor is not None:
                result = output_processor(result)
            return result

        wrapper._cli_name = tool_name
        return wrapper

    return decorator
