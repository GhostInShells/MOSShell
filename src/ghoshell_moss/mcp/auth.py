"""MCP OAuth — token 存储 + loopback 授权回调.

SDK 的 ``mcp.client.auth.OAuthClientProvider`` 已经实现完整的 MCP OAuth 2.1
authorization code + PKCE + 发现 + token exchange + 自动 refresh。本模块补它
缺的两块，让 ``moss mcp auth <name>`` 能把交互式授权走完并让 token 跨启动复用：

- :class:`ConfigTokenStorage` — ``TokenStorage`` 协议的 ConfigStore 实现，把
  ``OAuthToken``（access + refresh）和动态注册的 ``OAuthClientInformationFull``
  落到 ``MCPOAuthConfig`` ConfigType（mode-local，gitignored）。
- :class:`LoopbackCallback` — 在 ``127.0.0.1:0`` 起一个最小 HTTP server，收
  authorization server 的 redirect（``code`` / ``state`` / ``iss``）。

OAuth 只对 HTTP transport（sse / streamable_http）有意义，stdio 走 env 鉴权。

> 安全边界（未硬化，见 mcp-fusion-point FEATURE.md §未做）：token 目前明文
> 落 config。MOSS 尚未实现 credential Contract（未来应从 Project IoC 拿到
> 加密/keychain 背书的 TokenStorage），现阶段做法有风险。
"""

import asyncio
from typing import Awaitable, Callable, ClassVar
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field

from ghoshell_moss.contracts.configs import ConfigStore, ConfigType

try:
    from mcp.client.auth import OAuthClientProvider, TokenStorage
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
except ImportError:
    raise ImportError("mcp not installed. run: uv sync --all-extras")

__all__ = [
    'ConfigTokenStorage',
    'LoopbackCallback',
    'MCPOAuthConfig',
    'build_oauth_provider',
]


# ---------------------------------------------------------------------------
# token 存储 — ConfigType 背书
# ---------------------------------------------------------------------------

class MCPOAuthServerData(BaseModel):
    """单 server 的 OAuth 凭据，以 JSON 字符串存（opaque secret，不做 $VAR 解析）。"""

    tokens: str | None = None        # OAuthToken.model_dump_json()
    client_info: str | None = None   # OAuthClientInformationFull.model_dump_json()


class MCPOAuthConfig(ConfigType):
    """所有 MCP server 的 OAuth 凭据，经 ConfigStore 读写（mode-local，gitignored）。"""

    RESOLVE_ENV_KEY: ClassVar[bool] = False  # token 是 opaque secret，禁止 $VAR 解析

    servers: dict[str, MCPOAuthServerData] = Field(default_factory=dict)

    @classmethod
    def conf_name(cls) -> str:
        return "mcp_oauth"


class ConfigTokenStorage(TokenStorage):
    """``TokenStorage`` 的 ConfigStore 实现。

    现阶段 credential 未硬化：token 明文落 config（`.moss/configs/mcp_oauth.{mode}.yml`）。
    未来 credential Contract 落地后，从 Project IoC 拿加密/keychain 版替换本类。
    """

    def __init__(self, config_store: ConfigStore, name: str):
        self._config_store = config_store
        self._name = name

    def _load(self) -> MCPOAuthConfig:
        self._config_store.invalidate(MCPOAuthConfig)
        return self._config_store.get_or_create(MCPOAuthConfig(servers={}))

    def _server(self) -> MCPOAuthServerData | None:
        return self._load().servers.get(self._name)

    async def get_tokens(self) -> OAuthToken | None:
        data = self._server()
        if data and data.tokens:
            return OAuthToken.model_validate_json(data.tokens)
        return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        config = self._load()
        data = config.servers.setdefault(self._name, MCPOAuthServerData())
        data.tokens = tokens.model_dump_json(exclude_none=True)
        self._config_store.save(config)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._server()
        if data and data.client_info:
            return OAuthClientInformationFull.model_validate_json(data.client_info)
        return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        config = self._load()
        data = config.servers.setdefault(self._name, MCPOAuthServerData())
        data.client_info = client_info.model_dump_json(exclude_none=True)
        self._config_store.save(config)


# ---------------------------------------------------------------------------
# loopback 回调
# ---------------------------------------------------------------------------

class LoopbackCallback:
    """在 ``127.0.0.1:0`` 起一个最小 loopback HTTP server，等 authorization redirect。"""

    def __init__(self):
        self._server: asyncio.Server | None = None
        self._port: int | None = None
        self._result: asyncio.Future | None = None

    @property
    def redirect_uri(self) -> str:
        assert self._port is not None, "start() must be called first"
        return f"http://127.0.0.1:{self._port}/callback"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, '127.0.0.1', 0)
        self._port = self._server.sockets[0].getsockname()[1]
        self._result = asyncio.get_running_loop().create_future()

    async def wait_result(self) -> AuthorizationCodeResult:
        """``callback_handler`` — 等浏览器 redirect 返回授权码。"""
        assert self._result is not None, "start() must be called first"
        return await self._result

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            parts = request_line.decode('utf-8', 'replace').split()
            path = parts[1] if len(parts) > 1 else '/'
            parsed = urlparse(path)
            if parsed.path == '/callback':
                query = parse_qs(parsed.query)
                if query.get('error'):
                    error = query['error'][0]
                    desc = query.get('error_description', [''])[0]
                    self._result.set_exception(
                        RuntimeError(f"Authorization denied: {error}" + (f" — {desc}" if desc else ''))
                    )
                    body = b'<html><body><h1>Authorization failed</h1></body></html>'
                else:
                    self._result.set_result(AuthorizationCodeResult(
                        code=query.get('code', [''])[0],
                        state=query.get('state', [None])[0],
                        iss=query.get('iss', [None])[0],
                    ))
                    body = b'<html><body><h1>Authorization complete</h1><p>You can close this tab and return to the terminal.</p></body></html>'
            else:
                body = b'<html><body><h1>Not found</h1></body></html>'
            writer.write(
                b'HTTP/1.1 200 OK\r\n'
                b'Content-Type: text/html\r\n'
                b'Content-Length: ' + str(len(body)).encode() + b'\r\n'
                b'Connection: close\r\n\r\n' + body
            )
        finally:
            await writer.drain()
            writer.close()
            await writer.wait_closed()


# ---------------------------------------------------------------------------
# provider 构造
# ---------------------------------------------------------------------------

def build_oauth_provider(
    server_url: str,
    storage: TokenStorage,
    *,
    client_name: str,
    redirect_uri: str | None = None,
    redirect_handler: Callable[[str], Awaitable[None]] | None = None,
    callback_handler: Callable[[], Awaitable[AuthorizationCodeResult]] | None = None,
) -> OAuthClientProvider:
    """构造一个 ``OAuthClientProvider``。

    - 交互（CLI ``moss mcp auth``）：传 ``redirect_uri`` + ``redirect_handler`` +
      ``callback_handler``，401 时走完整授权。
    - 非交互（hub 运行时）：三者都不传，只用已落盘的 token 自动 refresh；若
      refresh 也不可用，授权流程会抛 ``OAuthFlowError``，引导人类跑 CLI。
    """
    metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri] if redirect_uri else None,
        client_name=client_name,
    )
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
