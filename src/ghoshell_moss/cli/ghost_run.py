"""moss-ghost — launch a Ghost interactive terminal."""

import click
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.host import Host
from ghoshell_moss.host.tui_entries.ghost_ui import GhostTUI


@click.command()
@click.argument("ghost", required=False, default=None)
@click.option("--mode", default="default", help="MOSS runtime mode.")
@click.option("--scope", default="default", help="Network scope for session isolation.")
@click.option("--network", default="local", help="Network driver.")
def ghost_run_main(ghost: str | None, mode: str, scope: str, network: str):
    """Launch a Ghost interactive terminal — stream logos, inspect output,
    operate the SafeMode approval gate. Meta-control surface for Ghost
    development; real Ghost interaction lives in the nodes system.

    GHOST: Ghost name to launch. Without it, lists all available Ghosts.
    """
    env = Environment(mode=mode, ghost=ghost, scope=scope, network=network)
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
        click.echo(f"\nRun: {click.style('moss-ghost <name>', fg='cyan')}")
        return

    if ghost not in available:
        click.echo(f"Ghost '{ghost}' not found. Available: {', '.join(available.keys())}")
        return

    click.echo(f"Starting Ghost TUI for [{ghost}] in [{mode}] mode, scope: [{scope}]")
    tui = GhostTUI(host=host)
    tui.run()


if __name__ == "__main__":
    ghost_run_main()
