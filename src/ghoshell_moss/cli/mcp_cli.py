"""MCP CLI — ``moss mcp`` group for MCP client/server management."""

import asyncio
import json
import re

import typer

from ghoshell_moss.cli.utils import echo
from ghoshell_moss.depends import depend_mcp

_NAME_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

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
        persist=True,
    )

    # uvicorn hijacks the root logger; stop moss logs from leaking to console.
    import logging
    logging.getLogger('moss').propagate = False

    bridge = GhostBridge(ttl=ttl)

    matrix.run(lambda m: _serve(
        m, bridge,
        server_name=server_name, host=host, port=port,
    ))


# ---------------------------------------------------------------------------
# config management — 一次性 node: 写 config + 通知模型, 不自动 open/close
# ---------------------------------------------------------------------------

def _discover_env():
    from ghoshell_moss.core.blueprint.environment import Environment

    try:
        return Environment.discover()
    except Exception:
        env = Environment()
        env.seal()
        return env


def _run_config_change(change: str, *, name: str = '', server_json: str = '') -> str:
    """一次性 node：入网 → 改 config → 发 notify → 退出。"""
    from ghoshell_moss.channels.mcp_channel import resolve_mcp_config_store
    from ghoshell_moss.core.blueprint.matrix import Matrix
    from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
    from ghoshell_moss.mcp.config import MCPHubConfig, MCPServerConfig

    depend_mcp()
    matrix = Matrix.new("mcp-cli", persist=False, env=_discover_env())

    async def _run():
        async with matrix:
            store = resolve_mcp_config_store(matrix)
            config = store.get_or_create(MCPHubConfig(servers={}))
            if change == 'add':
                try:
                    data = json.loads(server_json) if server_json else {}
                except json.JSONDecodeError as e:
                    return f"[mcp] invalid JSON: {e}"
                try:
                    server = MCPServerConfig.model_validate(data)
                except Exception as e:
                    return f"[mcp] invalid server config: {e}"
                if not _NAME_RE.fullmatch(server.name):
                    return f"[mcp] invalid server name `{server.name}` (must match `[a-zA-Z_][a-zA-Z0-9_]*`)"
                config.servers[server.name] = server
                store.save(config)
                matrix.session.add_signal(new_notify_signal(
                    f"[mcp] config changed: '{server.name}' added (not open).",
                ))
                return f"[mcp] added {server.name}"
            if change == 'remove':
                if name not in config.servers:
                    return f"[mcp] {name} not found in config"
                del config.servers[name]
                store.save(config)
                matrix.session.add_signal(new_notify_signal(
                    f"[mcp] config changed: '{name}' removed.",
                ))
                return f"[mcp] removed {name}"
            raise ValueError(f"unknown change: {change}")

    return asyncio.run(_run())


@mcp_app.command('list', short_help='List configured MCP servers.')
def list_servers():
    """List MCP servers in config (name, transport, description)."""
    from ghoshell_moss.channels.mcp_channel import resolve_mcp_config_store
    from ghoshell_moss.core.blueprint.matrix import Matrix
    from ghoshell_moss.mcp.config import MCPHubConfig

    depend_mcp()
    matrix = Matrix.new("mcp-cli", persist=False, env=_discover_env())

    async def _run():
        async with matrix:
            store = resolve_mcp_config_store(matrix)
            config = store.get_or_create(MCPHubConfig(servers={}))
            return config

    config = asyncio.run(_run())
    if not config.servers:
        echo("No MCP servers configured.")
        return
    for name, server in sorted(config.servers.items()):
        target = server.url if server.transport != 'stdio' else (server.command or '')
        echo(f"{name}\t{server.transport}\t{target}\t{server.description}")


@mcp_app.command('add', short_help='Add an MCP server config (writes config + notifies model, not open).')
def add_server(
    server_json: str = typer.Option(
        ...,
        '--json',
        help='MCPServerConfig JSON, e.g. {"name":"github","transport":"stdio","command":"npx","args":["-y","@modelcontextprotocol/server-github"]}',
    ),
):
    """Add an MCP server to config.  Does NOT open it — the model decides via mcp:open."""
    result = _run_config_change('add', server_json=server_json)
    echo(result)


@mcp_app.command('remove', short_help='Remove an MCP server config (writes config + notifies model).')
def remove_server(name: str = typer.Argument(..., help='MCP server name')):
    """Remove an MCP server from config."""
    result = _run_config_change('remove', name=name)
    echo(result)


@mcp_app.command('auth', short_help='Run interactive OAuth flow for an MCP server.')
def auth_server(name: str = typer.Argument(..., help='MCP server name')):
    """Complete OAuth authorization for an already-added HTTP MCP server.

    Prints the authorization URL — the user decides whether to open it. Tokens
    are stored via ConfigTokenStorage (plaintext under .moss/configs/, see warning).
    """
    from ghoshell_moss.channels.mcp_channel import MCPServerSession, resolve_mcp_config_store
    from ghoshell_moss.core.blueprint.matrix import Matrix
    from ghoshell_moss.mcp.auth import ConfigTokenStorage, LoopbackCallback, build_oauth_provider
    from ghoshell_moss.mcp.config import MCPHubConfig

    depend_mcp()
    echo(
        "[WARN] MOSS has not implemented a credential store yet.\n"
        "       OAuth tokens are stored in PLAINTEXT under .moss/configs/ (gitignored, NOT encrypted).\n"
        "       This is a known risk — credential hardening is tracked in mcp-fusion-point FEATURE.md."
    )

    matrix = Matrix.new("mcp-cli", persist=False, env=_discover_env())

    async def _run():
        async with matrix:
            store = resolve_mcp_config_store(matrix)
            config = store.get_or_create(MCPHubConfig(servers={}))
            server = config.servers.get(name)
            if server is None:
                return f"[mcp] {name} not found in config"
            if server.auth.kind != 'oauth':
                return f"[mcp] {name} auth kind is '{server.auth.kind}', not 'oauth'"
            if server.transport not in ('sse', 'streamable_http'):
                return f"[mcp] {name} transport '{server.transport}' does not support OAuth (needs sse/streamable_http)"

            storage = ConfigTokenStorage(store, name)
            loopback = LoopbackCallback()
            await loopback.start()

            async def _redirect(url: str) -> None:
                echo(f"\nOpen this URL to authorize (or copy it into a browser):\n\n  {url}\n")

            provider = build_oauth_provider(
                server_url=server.url,
                storage=storage,
                client_name=f"moss-{name}",
                redirect_uri=loopback.redirect_uri,
                redirect_handler=_redirect,
                callback_handler=loopback.wait_result,
            )
            session = MCPServerSession(config=server, oauth_provider=provider)
            try:
                await session.connect()
            finally:
                await loopback.close()
            if session.state == 'connected':
                return f"[mcp] {name} authorized and connected"
            return f"[mcp] {name} auth failed: {session.error}"

    echo(asyncio.run(_run()))
