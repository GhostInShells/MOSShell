"""moss-ghost — launch a Ghost and inject input signals.

Subcommands:
- ``run``: launch a Ghost — interactive TUI, or headless output/log observation.
- ``send``: inject a text signal (input/notify/interrupt/silent) to a running Ghost.

Without a subcommand, lists all available Ghosts.
"""

import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Awaitable, Callable

import click
import janus

from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.blueprint.session import OutputItem, Session
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.core.mindflow.silent_nucleus import new_silent_signal
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
@click.option(
    "--ghost",
    required=True,
    help="Ghost to observe — logos stream is keyed by session scope (includes ghost).",
)
@click.option(
    "--signal",
    "signal_type",
    type=click.Choice(["input", "notify", "interrupt", "silent"]),
    default="input",
    show_default=True,
    help="Signal type to send, routed to the matching nucleus.",
)
@click.option(
    "--priority",
    "priority_name",
    type=click.Choice(["background", "info", "notice", "warning", "error", "critical", "fatal"]),
    default=None,
    help="Override signal priority (default: the signal's own default). "
         "interrupt ignores this — it is always fatal.",
)
@click.pass_context
def send_cmd(ctx, text, ghost, signal_type, priority_name):
    """Inject a text signal to a running Ghost, then stream its logos response."""
    _send_signal(
        text, ctx.obj["mode"], ctx.obj["scope"], ctx.obj["network"], signal_type, ghost, priority_name,
    )


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


_LOGOS_OBSERVE_TIMEOUT = 30.0
"""观测 logos 的超时上限 — 兜底不发语音的 signal (silent / 纯 interrupt)。"""


def _send_signal(
        text: str,
        mode: str,
        scope: str,
        network: str,
        signal_type: str,
        ghost: str,
        priority_name: str | None,
) -> None:
    """Inject a text signal to a running Ghost, then stream its logos response.

    signal_type 决定发哪种 signal, 一一对应现成的 nucleus:
    input → InputSignalNucleus, notify → NotifyNucleus,
    interrupt → InterruptNucleus, silent → SilentNucleus.

    Signal key 是 scope 级 (MOSS/matrix/scopes/{scope}/signals), 但 logos key 是
    session_scope 级 (含 ghost 名), 所以观测 logos 必须传 ghost 对齐订阅 key。
    """
    env = Environment(mode=mode, ghost=ghost, scope=scope, network=network)
    env.seal()
    matrix = Matrix.new("ghost-send", persist=False, env=env)

    async def _send() -> None:
        async with matrix:
            session = matrix.session
            stream = session.get_stream(f"{Session.LOGOS_KEY}/{session.session_scope}")
            # 先订阅 (async with 进入即 declare_subscriber 就绪), 再发信号,
            # 避免 zenoh 无历史导致漏掉最早期的 logos delta.
            async with stream:
                _emit_signal(session, text, signal_type, priority_name)
                try:
                    async with asyncio.timeout(_LOGOS_OBSERVE_TIMEOUT):
                        async for sample in stream:
                            delta = sample.payload.decode('utf-8')
                            if delta == Session.LOGOS_END:
                                break
                            sys.stdout.write(delta)
                            sys.stdout.flush()
                except TimeoutError:
                    pass
        sys.stdout.write('\n')

    asyncio.run(_send())


def _emit_signal(session: Session, text: str, signal_type: str, priority_name: str | None) -> None:
    priority = Priority[priority_name.upper()] if priority_name else None
    if signal_type == "input":
        session.add_input_signal(text, priority=priority)
    elif signal_type == "notify":
        session.add_signal(new_notify_signal(text, priority=priority))
    elif signal_type == "interrupt":
        session.add_signal(new_interrupt_signal(text))
    else:  # silent
        session.add_signal(new_silent_signal(text, priority=priority))


if __name__ == "__main__":
    ghost_run_main()
