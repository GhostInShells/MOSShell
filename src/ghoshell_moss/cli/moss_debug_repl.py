"""moss-shell — MOSS Shell runtime entry.

Four modes:
- ``tui`` (default): interactive Textual/prompt_toolkit debugger for the Shell
  runtime — test CTML, inspect channels, debug before a Ghost runs.
- ``mcp``: expose the MOSS runtime as an MCP server for AI coding platforms
  (formerly the standalone ``moss-mcp`` binary). Requires the ``[mcp]`` extra —
  gated lazily, so the shell works without it.
- ``log``: headless run, no interaction, logs only. For CI / background use.
- ``fractalize``: enter the Matrix network as a single fractal cell whose only
  channel is this mode's NodeManager — remote hosts that ``mesh:accept`` this
  cell can list/read/run/stop the mode's nodes remotely. "Mode as cell."
"""

import click


@click.group(invoke_without_command=True)
@click.option(
    '--mode',
    default='default',
    help='MOSS runtime mode (e.g. default, dev, robot).',
)
@click.option(
    '--scope',
    default='default',
    help='Network scope for session isolation.',
)
@click.option(
    '--network',
    default='local',
    help='Network driver.',
)
@click.pass_context
def moss_shell_main(ctx, mode: str, scope: str, network: str):
    """MOSS Shell runtime — interactive TUI debug, MCP server, or headless log."""
    ctx.obj = {'mode': mode, 'scope': scope, 'network': network}
    if ctx.invoked_subcommand is None:
        ctx.invoke(tui)


def _build_env(ctx):
    """§UU-1 seal 定案: 入口点显式构造 Environment(**cli_args) + seal, 注册 singleton.
    Host 只消费 sealed env, 不承担参数收集责任."""
    from ghoshell_moss.core.blueprint.environment import Environment
    env = Environment(
        mode=ctx.obj['mode'],
        scope=ctx.obj['scope'],
        network=ctx.obj['network'],
    )
    env.seal()
    return env


@moss_shell_main.command()
@click.pass_context
def tui(ctx):
    """Interactive TUI debugger — test CTML, inspect channels (default)."""
    click.echo(
        f"Starting MOSS Shell debugger in [{ctx.obj['mode']}] mode, scope: [{ctx.obj['scope']}]"
    )
    from ghoshell_moss.host import Host
    from ghoshell_moss.host.tui_entries.moss_runtime_ui import MossRuntimeTUI

    host = Host(env=_build_env(ctx))
    ui = MossRuntimeTUI(host=host)
    ui.run()


@moss_shell_main.command()
@click.option(
    '--transport',
    type=click.Choice(['sse', 'std', 'streamable_http']),
    default='streamable_http',
    help='通信协议',
)
@click.option('--host', default='127.0.0.1', help='MCP 服务地址 (network 传输时生效)')
@click.option('--port', default=20773, help='MCP 服务端口 (network 传输时生效)')
@click.option('--server-name', default='MOSS-Toolset-Server', help='MCP 服务名称')
@click.pass_context
def mcp(ctx, transport, host, port, server_name):
    """Run the MOSS runtime as an MCP server for AI coding platforms (was moss-mcp)."""
    from ghoshell_moss.depends import depend_mcp

    try:
        depend_mcp()
    except ImportError:
        click.echo(
            "mcp extra not installed. run: uv sync --all-extras (or install ghoshell_moss[mcp])",
            err=True,
        )
        raise click.exceptions.Exit(1)
    # 惰性 import: mcp 包只在 mcp 模式被拖入 (moss_as_mcp 顶层 import mcp).
    from ghoshell_moss.cli.moss_as_mcp import main_entry

    main_entry(
        mode=ctx.obj['mode'],
        scope=ctx.obj['scope'],
        network=ctx.obj['network'],
        transport=transport,
        host=host,
        port=port,
        server_name=server_name,
    )


@moss_shell_main.command()
@click.pass_context
def log(ctx):
    """Headless run — no interaction, logs only. For CI / background debugging."""
    import logging

    from ghoshell_moss.core.helpers.logger import get_console_logger
    from ghoshell_moss.host import Host

    host = Host(env=_build_env(ctx))
    runtime = host.run()

    # project.bootstrap 已挂 moss.log file handler; 此时补 console handler.
    get_console_logger(logging.INFO)
    runtime.matrix.logger.info(
        'MOSS shell log mode running (mode=%s scope=%s network=%s)',
        ctx.obj['mode'], ctx.obj['scope'], ctx.obj['network'],
    )

    # MossRuntime.run_until_closed: code-as-prompt 同步阻塞入口, 治理完整
    # MossRuntime 生命周期 (__aenter__ → wait_closed → __aexit__) +
    # KeyboardInterrupt graceful teardown.
    runtime.run_until_closed()


@moss_shell_main.command()
@click.pass_context
def fractalize(ctx):
    """Enter the Matrix network as a fractal cell exposing this mode's nodes.

    Mode as Cell — the running process is a single node cell whose only
    providing channel is this mode's NodeManager (list/read/run/stop/...).
    Remote hosts that mesh:accept this cell can govern the mode's nodes
    remotely. Blocks until closed. See workstream: mode-as-cell.
    """
    import logging

    from ghoshell_moss.core.blueprint.cell import normalize
    from ghoshell_moss.core.blueprint.matrix import Matrix
    from ghoshell_moss.core.helpers.logger import get_console_logger
    from ghoshell_moss.channels.matrix_channel import new_nodes_channel

    env = _build_env(ctx)
    node_name = f'{normalize(env.project_name)}_{normalize(env.mode_name)}'

    matrix = Matrix.new(
        node_name,
        description=f'Fractal cell for mode {env.mode_name!r} of project {env.project_name!r}.',
        category='fractal',
        env=env,
        persist=True,
        singleton=True,
    )

    # project.bootstrap 已挂 moss.log file handler; 此时补 console handler,
    # 让 boot 期间的 identity/network 摘要打到 stderr, 方便复制排障.
    get_console_logger(logging.INFO)

    cell = matrix.this
    net = matrix.network_info
    matrix.logger.info(
        'MOSS fractalize running: cell.address=%s cell.role=%s cell.category=%s '
        'project_name=%s project_id=%s mode=%s network=%s scope=%s driver=%s',
        cell.address, cell.role, cell.category,
        env.project_name, env.project_id, env.mode_name,
        ctx.obj['network'], net.scope, net.driver,
    )
    matrix.logger.info(
        'MOSS fractalize providing channel = mode NodeManager. '
        'Remote host: matrix.mesh:accept(<address>) to gain fractal.<short>:list/run/... '
        'Ctrl+C to stop.'
    )

    async def _serve(m: Matrix) -> None:
        channel = new_nodes_channel(m)
        # provide_channel 返回 Future, await 会阻塞到膜被外部关闭.
        # 这是 cell 唯一的入网动作 — Matrix.new + provide_channel + await.
        await m.provide_channel(channel)

    matrix.run(_serve)


if __name__ == '__main__':
    moss_shell_main()
