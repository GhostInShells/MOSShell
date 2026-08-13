"""MCP config models — the canonical definition of MCP server connection config.

These are pure data models (no I/O, no transport logic).  Storage and
resolution are handled by :class:`ConfigStore` / :class:`ConfigType`.

Config is shared between:
- CLI (``moss mcp connect`` — writes)
- Channel surface (``mcp`` channel — reads via ConfigStore)
- GUI surface (human reviews / edits / authorizes)
"""

from typing import Literal
from pydantic import BaseModel, Field

from ghoshell_moss.contracts.configs import ConfigType

__all__ = [
    'MCPServerConfig',
    'MCPHubConfig',
    'AuthKind',
    'AuthConfig',
]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

AuthKind = Literal['none', 'env', 'oauth', 'header']


class AuthConfig(BaseModel):
    """Describes what kind of auth a server needs.

    Not the secret itself — just enough info for the GUI to render the right
    controls and for the runtime to select the correct transport auth path.

    - **env**: secret in an env var (stdio or HTTP transport)
    - **oauth**: OAuth 2.1 Authorization Code + PKCE.  The authorization server
      URL is auto-discovered by the MCP SDK from Protected Resource Metadata
      (``.well-known/oauth-protected-resource``).  The redirect URI is a
      loopback on a random port, served by the interactive face (CLI/GUI).
    - **header**: static header with a ``$VAR`` reference to the value
    - **none**: no auth needed
    """

    kind: AuthKind = Field(
        default='none',
        description="none | env | oauth | header",
    )

    # env
    env_var: str = Field(
        default='',
        description="env: name of the env var holding the secret (e.g. GITHUB_TOKEN)",
    )

    # oauth
    scopes: list[str] = Field(
        default_factory=list,
        description="oauth: requested scopes (e.g. ['tools:read', 'tools:write'])",
    )

    # header
    header_name: str = Field(
        default='',
        description="header: static header name (e.g. X-API-Key, Authorization)",
    )
    header_value_ref: str = Field(
        default='',
        description="header: $VAR reference to the header value (e.g. $GITHUB_TOKEN)",
    )


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------

class MCPServerConfig(BaseModel):
    """Connection config for a single MCP server.

    One config = one server that can be connected via ``mcp:connect`` or CLI.
    Secrets are never stored inline — use ``$VAR`` references resolved by
    :meth:`ConfigType.resolve` at read time.
    """

    name: str = Field(description="server identifier (channel path segment, CLI name)")
    transport: Literal['stdio', 'sse', 'streamable_http'] = Field(
        default='stdio',
        description="transport protocol",
    )
    description: str = Field(
        default='',
        description="human-readable summary, shown in help and GUI cards",
    )

    # stdio ---------------------------------------------------------------

    command: str = Field(
        default='',
        description="stdio: executable path or command name",
    )
    args: list[str] = Field(
        default_factory=list,
        description="stdio: cli arguments",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="stdio: env vars (use $VAR for secrets, resolved at read time)",
    )

    # sse / streamable_http -----------------------------------------------

    url: str = Field(
        default='',
        description="sse/streamable_http: server URL",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="sse/streamable_http: request headers (use $VAR for secrets)",
    )

    # auth ----------------------------------------------------------------

    auth: AuthConfig = Field(
        default_factory=AuthConfig,
        description="auth metadata — GUI uses this to render the right controls",
    )

    # misc ----------------------------------------------------------------

    auto_connect: bool = Field(
        default=True,
        description="connect automatically on hub startup",
    )


# ---------------------------------------------------------------------------
# Hub config (collection)
# ---------------------------------------------------------------------------

class MCPHubConfig(ConfigType):
    """Persisted collection of known MCP server configs.

    Stored via ConfigStore (workspace or scoped).  The ``servers`` dict is the
    source of truth for "what can be connected".
    """

    servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="server_name → config",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "mcp_hub"
