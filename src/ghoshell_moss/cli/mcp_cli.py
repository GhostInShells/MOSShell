"""MCP CLI — ``moss mcp`` group for MCP client/server management."""

import typer

from ghoshell_moss.cli.utils import echo
from ghoshell_moss.depends import depend_mcp

mcp_app = typer.Typer(
    name="mcp",
    help="MCP client/server management",
    no_args_is_help=True,
)


@mcp_app.command('serve-ghost-bridge', short_help='Serve the ghost bridge for external agent↔ghost communication.')
def serve_ghost_bridge(
    host: str = typer.Option('127.0.0.1', '--host', help='MCP server host'),
    port: int = typer.Option(20774, '--port', help='MCP server port'),
    server_name: str = typer.Option('ghost-bridge', '--server-name', help='MCP server name'),
    ttl: float = typer.Option(300.0, '--ttl', help='Bridge envelope TTL in seconds'),
):
    """Serve the ghost bridge — bidirectional MCP channel between external agents and ghost.

    Creates a lightweight cell node that joins the existing mesh and serves
    the bridge.  External agents connect via MCP tools (send/pull/wait_reply).
    The ghost replies via CTML: ghost_bridge:reply(task_id, text__).
    """
    from ghoshell_moss.core.blueprint.environment import Environment
    from ghoshell_moss.core.blueprint.matrix import Matrix
    from ghoshell_moss.mcp import GhostBridge, serve_ghost_bridge as _serve

    try:
        env = Environment.discover()
    except Exception:
        env = Environment()
        env.seal()

    matrix = Matrix.new(
        "ghost_bridge",
        description="MCP ghost bridge — bidirectional agent↔ghost communication",
        env=env,
    )

    # uvicorn hijacks the root logger; stop moss logs from leaking to console.
    import logging
    logging.getLogger('moss').propagate = False

    bridge = GhostBridge(ttl=ttl)

    matrix.run(lambda m: _serve(
        m, bridge,
        server_name=server_name, host=host, port=port,
    ))
