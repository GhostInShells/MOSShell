"""moss-run-ghost — 启动 Ghost TUI 交互终端."""

import click
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.host import Host
from ghoshell_moss.host.tui_entries.ghost_ui import GhostTUI


@click.command()
@click.argument("ghost", required=False, default=None)
@click.option("--mode", default="default", help="MOSS 运行模式.")
@click.option("--scope", default="default", help="会话范围 (session scope).")
def ghost_run_main(ghost: str | None, mode: str, scope: str):
    """启动 Ghost TUI 交互终端 — 与 Ghost 实时对话。

    GHOST: 要启动的 Ghost 名称。不提供时列出所有可用的 Ghost。
    """
    env = Environment(mode=mode, ghost=ghost, scope=scope)
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
        return

    if ghost is None:
        click.echo("Available ghosts:\n")
        for name, meta in available.items():
            click.echo(f"  {click.style(name, fg='green', bold=True)} — {meta.prototype()}")
            click.echo(f"    {meta.description().split(chr(10))[0][:100]}")
        click.echo(f"\nRun: {click.style('moss-run-ghost <name>', fg='cyan')}")
        return

    if ghost not in available:
        click.echo(f"Ghost '{ghost}' not found. Available: {', '.join(available.keys())}")
        return

    click.echo(f"Starting Ghost TUI for [{ghost}] in [{mode}] mode, scope: [{scope}]")
    tui = GhostTUI(host=host)
    tui.run()


if __name__ == "__main__":
    ghost_run_main()
