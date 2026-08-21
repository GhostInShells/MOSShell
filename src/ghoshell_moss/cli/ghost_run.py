"""moss-ghost — launch a Ghost and inject input signals.

Subcommands:
- ``run``: launch a Ghost — interactive TUI, or headless output/log observation.
- ``send``: inject a text input signal to a running Ghost in the same scope.

Without a subcommand, lists all available Ghosts.
"""

import asyncio
import contextlib
import os
import signal
from collections.abc import Awaitable, Callable

import click
import janus

from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.host import Host


@click.group(invoke_without_command=True)
@click.option("--mode", default="default", help="MOSS runtime mode.")
@click.option("--scope", default="default", help="Network scope for session isolation.")
@click.option("--network", default="local", help="Network driver.")
@click.pass_context
def ghost_run_main(ctx, mode, scope, network):
    """Launch a Ghost and inject input signals.

    Without a subcommand, lists all available Ghosts.
    """
    ctx.ensure_object(dict)
    ctx.obj["mode"] = mode
    ctx.obj["scope"] = scope
    ctx.obj["network"] = network
    if ctx.invoked_subcommand is None:
        _resolve(ctx.obj, None)


@ghost_run_main.command("run")
@click.argument("ghost", required=False, default=None)
@click.option(
    "--surface",
    type=click.Choice(["tui", "output", "log"]),
    default="tui",
    show_default=True,
    help="Interaction surface: interactive TUI, stdout output, or log file.",
)
@click.pass_context
def run_cmd(ctx, ghost, surface):
    """Launch a Ghost — interactive TUI, or headless output/log observation.

    GHOST: Ghost name to launch. Without it, lists all available Ghosts.
    """
    if ghost is None:
        _resolve(ctx.obj, None)
        return
    resolved = _resolve(ctx.obj, ghost)
    if resolved is None:
        return
    host, ghost_name = resolved

    if surface == "tui":
        _run_tui(host, ghost_name, ctx.obj)
    elif surface == "output":
        _run_output(host, ghost_name)
    else:  # log
        _run_log(host, ghost_name, ctx.obj)


@ghost_run_main.command("send")
@click.argument("text")
@click.pass_context
def send_cmd(ctx, text):
    """Inject a text input signal to a running Ghost in the same scope."""
    _send_input(text, ctx.obj["mode"], ctx.obj["scope"], ctx.obj["network"])


# ── 共享解析 / 输出 ──────────────────────────────────


def _resolve(ctx: dict, ghost: str | None) -> tuple[Host, str] | None:
    """Build sealed env + host, validate the requested ghost.

    Returns (host, ghost_name) on success, or None after printing a listing / error.
    """
    env = Environment(
        mode=ctx["mode"],
        ghost=ghost,
        scope=ctx["scope"],
        network=ctx["network"],
    )
    env.seal()
    host = Host(env=env)

    available = {}
    for _path, meta in host.project.ghosts():
        if isinstance(meta, Exception):
            continue
        available[meta.name()] = meta

    if not available:
        click.echo("No ghosts found in workspace.")
        click.echo("Place a GhostMeta instance in MOSS/ghosts/ to register one.")
        return None

    if ghost is None:
        click.echo("Available ghosts:\n")
        for name, meta in available.items():
            click.echo(f"  {click.style(name, fg='green', bold=True)} — {meta.prototype()}")
            click.echo(f"    {meta.description().split(chr(10))[0][:100]}")
        click.echo(f"\nRun: {click.style('moss-ghost run <name>', fg='cyan')}")
        return None

    if ghost not in available:
        click.echo(f"Ghost '{ghost}' not found. Available: {', '.join(available.keys())}")
        return None

    return host, ghost


def _print_env_header(host: Host) -> None:
    """打印环境可暴露信息表头 — 对齐 TUI welcome() 的 cell info + env config."""
    env = host.env
    click.echo("== Current Cell Info ==")
    for k, v in (
        ("address", env.this_cell_address),
        ("mode", env.mode_name),
        ("ghost", env.ghost_name),
        ("network", env.network),
        ("scope", env.network_scope),
        ("project_id", env.project_id),
    ):
        click.echo(f"  {k:<12} {v}")
    click.echo("")
    click.echo("== Environment Configuration ==")
    for k, v in env.dump_cell_env(with_os_env=False).items():
        click.echo(f"  {k:<12} {v}")
    click.echo(f"  {'SELF_PID':<12} {os.getpid()}")
    click.echo("")


def _print_output_item(item: OutputItem) -> None:
    text = item.messages_string()
    log = item.log
    if not text and not log:
        return
    if log:
        click.echo(f"--- [{item.role}] {log}")
    else:
        click.echo(f"--- [{item.role}] ---")
    if text:
        click.echo(text)


async def _output_printer(queue: janus.Queue) -> None:
    while True:
        item = await queue.async_q.get()
        if item is None:
            return
        _print_output_item(item)


def _run_ghost_headless(ghost_runtime, main: Callable[[], Awaitable[None]]) -> None:
    """Headless: asyncio.run(main()) with SIGINT → ghost_runtime.close().

    todo: 优雅退出 bug — 信号 handler 只同步调 ``ghost_runtime.close()`` (只关
    moss_runtime/mindflow), 不 await ghost 的 ``__aexit__``, 而 dsh launcher 挂在
    ghost 的 exit stack 上, 没人关 → 残留孤儿 dsh 进程占端口 (复现: 起 ghost 后
    Ctrl+C, ``lsof -iTCP:3083`` 仍见 node dsh). 且只处理 SIGINT, 未处理 SIGTERM.
    修法: 信号 handler 内 schedule 一个 async 任务走 ``ghost_runtime.__aexit__``.
    """
    prev = signal.signal(signal.SIGINT, lambda s, f: ghost_runtime.close())
    try:
        asyncio.run(main())
    finally:
        signal.signal(signal.SIGINT, prev)


# ── 三个交互面 ──────────────────────────────────────


def _run_tui(host: Host, ghost_name: str, ctx: dict) -> None:
    click.echo(
        f"Starting Ghost TUI for [{ghost_name}] in [{ctx['mode']}] mode, "
        f"scope: [{ctx['scope']}]"
    )
    from ghoshell_moss.host.tui_entries.ghost_ui import GhostTUI

    GhostTUI(host=host).run()


def _run_output(host: Host, ghost_name: str) -> None:
    _print_env_header(host)
    ghost_runtime = host.run_ghost(ghost_name)
    queue: janus.Queue = janus.Queue()

    async def _main() -> None:
        printer = asyncio.create_task(_output_printer(queue))
        try:
            # 启动前注册回调 — GhostRuntime 缓冲, __aenter__ (matrix 就绪后) 优先装线,
            # 从而捕获 ghost __aenter__ (stubs sync / dsh 启动) 发出的 output.
            ghost_runtime.on_output(
                lambda item: queue.sync_q.put_nowait(item)
            )
            async with ghost_runtime:
                await ghost_runtime.moss.wait_close()
        finally:
            queue.sync_q.put_nowait(None)
            printer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await printer

    _run_ghost_headless(ghost_runtime, _main)


def _run_log(host: Host, ghost_name: str, ctx: dict) -> None:
    import logging

    from ghoshell_moss.core.helpers.logger import get_console_logger

    _print_env_header(host)
    get_console_logger(logging.INFO)
    ghost_runtime = host.run_ghost(ghost_name)
    ghost_runtime.moss.matrix.logger.info(
        "MOSS ghost log mode running (ghost=%s mode=%s scope=%s network=%s)",
        ghost_name, ctx["mode"], ctx["scope"], ctx["network"],
    )

    async def _main() -> None:
        async with ghost_runtime:
            await ghost_runtime.moss.wait_close()

    _run_ghost_headless(ghost_runtime, _main)


# ── send: 输入注入 ──────────────────────────────────


def _send_input(text: str, mode: str, scope: str, network: str) -> None:
    """Inject a text input signal to a running Ghost in the same scope.

    Signal 命名空间只含 network_scope (MOSS/matrix/scopes/{scope}/signals),
    所以对齐 scope 即可命中 running ghost 的 mindflow input nucleus —
    ghost 名不进路由, 不传 --ghost。
    """
    env = Environment(mode=mode, scope=scope, network=network)
    env.seal()
    matrix = Matrix.new("ghost-send", persist=False, env=env)

    async def _send() -> None:
        async with matrix:
            matrix.session.add_input_signal(text)

    asyncio.run(_send())


if __name__ == "__main__":
    ghost_run_main()
