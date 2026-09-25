"""MCP Hub Channel — 把外部 MCP server 的工具集接入 MOSS | 集成 | beta

``MCPHub`` 是本源对象：持有 N 个 MCP server session，从 ConfigStore 读配置，
由模型通过 CTML 主动 ``open`` / ``close``。每个已打开的 server 投影为一个虚拟子
channel（以 server 名命名），子 channel 的 ``notice`` 渲染该 server 全部 tool 的
JSON schema 接口描述，并提供 ``call``（阻塞）/ ``acall``（非阻塞）两个调用入口。

同一个对象、两个面：构建方 / GUI 持有 ``MCPHub`` 直接调用公开方法，模型通过
``build_mcp_hub_channel`` 构建出的 channel 用 CTML 操作。

Example:
    from ghoshell_moss.core.blueprint.states_channel import new_shell_main_channel
    from ghoshell_moss.channels.mcp_channel import mcp_hub_channel_factory
    main = new_shell_main_channel()
    main.import_channels(mcp_hub_channel_factory(name='mcp'))
"""

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass, field

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.core.blueprint.channel_builder import ChannelFactory, new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.mcp.config import MCPHubConfig, MCPServerConfig
from ghoshell_moss.message import Base64Image, Message, Text, unique_id

try:
    import httpx2
    import mcp
    from mcp import types as mcp_types
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    raise ImportError("mcp not installed. run: uv sync --all-extras")

__all__ = [
    'MCPHub',
    'MCPServerSession',
    'build_mcp_hub_channel',
    'mcp_hub_channel_factory',
    'mcp_result_to_observe',
    'render_input_schema',
    'render_tools_notice',
    'resolve_mcp_config_store',
    'NOT_IMPLEMENTED',
]

# ---------------------------------------------------------------------------
# Developer-facing record of surfaces this hub does NOT yet wire in.
#
# This is for the *coding model* / next developer only.  It is deliberately
# NOT rendered into any ghost-facing surface (notice / instruction /
# description / context_messages): the ghost sees which commands exist and
# does not need to be told what is missing — absence is self-evident to it.
# See FEATURE.md (mcp-fusion-point) §未做.
# ---------------------------------------------------------------------------

NOT_IMPLEMENTED: tuple[str, ...] = ('tasks', 'resources')
"""MCP surfaces not yet integrated natively into the channel.

- ``tasks``     — MCP Tasks (SEP-2663) 异步长任务协议.
- ``resources`` — MCP resources (与 MOSS 网络内 resource 撞名、不同源).
"""

_CHANNEL_NAME_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _is_valid_channel_name(name: str) -> bool:
    """server 名要能作为 channel 路径段 (同 ``ChannelNamePattern``)."""
    return bool(_CHANNEL_NAME_RE.fullmatch(name))


# ---------------------------------------------------------------------------
# MCP result → Observe
# ---------------------------------------------------------------------------

def mcp_result_to_observe(
    result: mcp_types.CallToolResult,
    *,
    server: str,
    tool: str,
) -> Observe:
    """将 MCP CallToolResult 转为 Observe，只保留 text + image 两种 content。"""
    if result.is_error:
        text_parts = []
        for c in result.content:
            if isinstance(c, mcp_types.TextContent):
                text_parts.append(c.text)
        return Observe.new(f"[mcp:{server}/{tool}] error: {' '.join(text_parts)}")

    messages = []
    for c in result.content:
        if isinstance(c, mcp_types.TextContent):
            messages.append(Message.new(name=f"{server}/{tool}").with_content(Text(text=c.text)))
        elif isinstance(c, mcp_types.ImageContent):
            messages.append(
                Message.new(name=f"{server}/{tool}").with_content(
                    Base64Image.from_base64(media_type=c.mime_type, data=c.data)
                )
            )
    return Observe(messages=messages)


# ---------------------------------------------------------------------------
# schema rendering (ghost-facing: notice 内容)
# ---------------------------------------------------------------------------

def render_input_schema(schema: dict) -> str:
    """将 MCP tool inputSchema 渲染为简洁的参数列表。"""
    if not schema or schema.get('type') != 'object':
        return ''
    properties = schema.get('properties', {})
    if not properties:
        return ''
    required = set(schema.get('required', []))
    parts = []
    for param_name, param_schema in properties.items():
        ptype = param_schema.get('type', 'any')
        pdesc = (param_schema.get('description', '') or '').split('\n')[0][:80]
        req = ', required' if param_name in required else ''
        if pdesc:
            parts.append(f"`{param_name}` ({ptype}{req}): {pdesc}")
        else:
            parts.append(f"`{param_name}` ({ptype}{req})")
    return ', '.join(parts)


def render_tools_notice(session: 'MCPServerSession') -> str:
    """渲染单个 server 的 tool schema 摘要，作为子 channel 的 notice。"""
    name = session.config.name
    if session.state != 'connected':
        if session.error:
            return f"{name}: {session.state} — {session.error}"
        return f"{name}: {session.state}"
    tools = session.tools
    if not tools:
        return f"{name}: connected, 0 tools"
    lines = [f"{name} — {len(tools)} tools:"]
    for t in tools:
        desc = (t.description or '').split('\n')[0][:80]
        params = render_input_schema(t.input_schema)
        sig = f"{t.name}({params})" if params else f"{t.name}()"
        lines.append(f"- {sig}" + (f" — {desc}" if desc else ''))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# MCP server session wrapper
# ---------------------------------------------------------------------------

@dataclass
class MCPServerSession:
    """管理单个 MCP server 的连接生命周期。"""

    config: MCPServerConfig
    oauth_provider: 'httpx2.Auth | None' = None
    client: mcp.ClientSession | None = None
    tools: list[mcp_types.Tool] = field(default_factory=list)
    state: str = 'disconnected'  # disconnected | connecting | connected | error
    error: str = ''
    _exit_stack: contextlib.AsyncExitStack | None = None

    async def connect(self) -> None:
        """建立 transport 连接，初始化 session，发现 tools。"""
        if self.state == 'connected':
            return
        self.state = 'connecting'
        self.error = ''
        try:
            self._exit_stack = contextlib.AsyncExitStack()
            read, write = await self._connect_transport()
            session = mcp.ClientSession(read, write)
            await self._exit_stack.enter_async_context(session)
            await session.initialize()
            result = await session.list_tools()
            self.client = session
            self.tools = list(result.tools)
            self.state = 'connected'
        except Exception as e:
            self.state = 'error'
            self.error = str(e)
            if self._exit_stack:
                with contextlib.suppress(Exception):
                    await self._exit_stack.aclose()
                self._exit_stack = None

    async def _connect_transport(self):
        cfg = self.config
        if cfg.transport == 'stdio':
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=cfg.env or None,
            )
            return await self._exit_stack.enter_async_context(stdio_client(params))
        elif cfg.transport == 'sse':
            return await self._exit_stack.enter_async_context(
                sse_client(url=cfg.url, headers=cfg.headers or None, auth=self.oauth_provider)
            )
        elif cfg.transport == 'streamable_http':
            # headers (PAT/API key) 和 OAuth 都要经自定义 http_client 传入.
            if cfg.headers or self.oauth_provider is not None:
                kwargs: dict = {}
                if cfg.headers:
                    kwargs['headers'] = cfg.headers
                if self.oauth_provider is not None:
                    kwargs['auth'] = self.oauth_provider
                http_client = httpx2.AsyncClient(**kwargs)
                await self._exit_stack.enter_async_context(http_client)
                read, write = await self._exit_stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
                return read, write
            read, write = await self._exit_stack.enter_async_context(
                streamable_http_client(cfg.url)
            )
            return read, write
        raise ValueError(f"unsupported transport: {cfg.transport}")

    async def disconnect(self) -> None:
        """断开连接，清理资源。"""
        self.state = 'disconnected'
        self.tools.clear()
        self.client = None
        if self._exit_stack:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = None

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30.0) -> Observe:
        """调用 MCP tool 并返回 Observe。"""
        if not self.client or self.state != 'connected':
            return Observe.new(f"[mcp:{self.config.name}] server not connected")
        try:
            result = await asyncio.wait_for(
                self.client.call_tool(name=name, arguments=arguments),
                timeout=timeout,
            )
            return mcp_result_to_observe(result, server=self.config.name, tool=name)
        except asyncio.TimeoutError:
            return Observe.new(f"[mcp:{self.config.name}/{name}] timeout after {timeout}s")
        except mcp.McpError as e:
            self.state = 'error'
            self.error = str(e)
            return Observe.new(f"[mcp:{self.config.name}/{name}] MCP error: {e}")


# ---------------------------------------------------------------------------
# MCP Hub — 本源对象
# ---------------------------------------------------------------------------

class MCPHub:
    """管理 N 个 MCP client session 的 hub 状态对象。

    构建方 / GUI 持有实例直接调用公开方法（``open`` / ``close`` / ``list_servers`` /
    ``add_server`` / ``remove_server`` / ``sessions``），模型通过 channel 命令操作。
    """

    def __init__(
        self,
        *,
        config_store: ConfigStore,
        name: str = 'mcp',
        description: str = '',
        allow_config_edit: bool = False,
    ):
        self._config_store = config_store
        self._name = name
        self._description = description or 'MCP Hub — 打开外部 MCP server 并使用其工具'
        self._allow_config_edit = allow_config_edit
        self._uid = unique_id()
        self._sessions: dict[str, MCPServerSession] = {}

    # --- 公开方法 (构建方 / GUI 直接调用) ---

    def id(self) -> str:
        return self._uid

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def sessions(self) -> dict[str, MCPServerSession]:
        """当前已打开的 server session（同步快照）。"""
        return dict(self._sessions)

    def allow_config_edit(self) -> bool:
        return self._allow_config_edit

    async def authorize(self, action: str, name: str) -> None:
        """授权 seam —— hub 的 open/close/add/remove 都经过这一处。

        当前直通，不做审批。待 warrant 体系接入 channel 时，此处是唯一需要改的点：
        调用 ``matrix.warrant.require(...)``，把 MCP capability 变更映射成一条
        Permission（纯业务逻辑）+ Warrant（IO 面）。四个调用点只走这一个函数，
        集成时只动这里。见 FEATURE.md (mcp-fusion-point) §未做 — warrant 集成。
        """
        return None

    async def open(self, name: str) -> str:
        """打开（连接）配置中的某个 MCP server。"""
        if not _is_valid_channel_name(name):
            return f"[mcp:{name}] invalid server name (must match `[a-zA-Z_][a-zA-Z0-9_]*`)"
        if name in self._sessions:
            session = self._sessions[name]
            if session.state == 'connected':
                return f"[mcp:{name}] already connected"
        config = self._load_config()
        server_cfg = config.servers.get(name)
        if server_cfg is None:
            available = sorted(config.servers.keys())
            suffix = f" Available: {', '.join(available)}" if available else ''
            return f"[mcp:{name}] not found in config.{suffix}"
        session = MCPServerSession(
            config=server_cfg,
            oauth_provider=self._build_oauth_provider(server_cfg),
        )
        await session.connect()
        self._sessions[name] = session
        result = f"[mcp:{name}] {session.state}" + (f": {session.error}" if session.error else '')
        if session.error and 'OAuth' in session.error:
            result += f" (authorize first: moss mcp auth {name})"
        return result

    async def close(self, name: str) -> str:
        """关闭（断开）已打开的 MCP server；配置保留，可再次 open。"""
        session = self._sessions.pop(name, None)
        if session is None:
            return f"[mcp:{name}] not open"
        await session.disconnect()
        return f"[mcp:{name}] closed"

    async def list_servers(self) -> str:
        """列出已打开 server 的状态，及配置中尚未打开的 server。"""
        lines = ["### MCP Servers\n"]
        for name, session in self._sessions.items():
            icon = {'connected': '+', 'connecting': '~', 'disconnected': '-', 'error': '!'}.get(
                session.state, '?'
            )
            lines.append(f"[{icon}] **{name}** ({session.state})")
            if session.error:
                lines.append(f"    error: {session.error}")
            lines.append("")
        config = self._load_config()
        available = {
            n: c.description or '' for n, c in config.servers.items() if n not in self._sessions
        }
        if available:
            lines.append("Configured (not open):")
            for n in sorted(available):
                desc = f" — {available[n]}" if available[n] else ''
                lines.append(f"- `{n}`{desc}")
        elif not self._sessions:
            lines.append("No MCP servers configured.")
        return '\n'.join(lines)

    async def add_server(self, text__: str) -> str:
        """向配置新增一个 MCP server（只写配置，不连接）。``text__`` 为 JSON。"""
        try:
            data = json.loads(text__) if text__ else {}
        except json.JSONDecodeError as e:
            return f"[mcp] invalid JSON: {e}"
        try:
            server_cfg = MCPServerConfig.model_validate(data)
        except Exception as e:
            return f"[mcp] invalid server config: {e}"
        if not _is_valid_channel_name(server_cfg.name):
            return f"[mcp] invalid server name `{server_cfg.name}` (must match `[a-zA-Z_][a-zA-Z0-9_]*`)"
        config = self._load_config()
        config.servers[server_cfg.name] = server_cfg
        self._config_store.save(config)
        return f"[mcp:{server_cfg.name}] config added (not open)"

    async def remove_server(self, name: str) -> str:
        """从配置删除一个 MCP server；若已打开则先关闭。"""
        config = self._load_config()
        if name not in config.servers:
            return f"[mcp:{name}] not found in config"
        session = self._sessions.pop(name, None)
        if session is not None:
            await session.disconnect()
        del config.servers[name]
        self._config_store.save(config)
        return f"[mcp:{name}] removed from config"

    async def open_auto_connect(self) -> None:
        """启动时打开 ``auto_connect=True`` 的 server。"""
        config = self._load_config()
        for name, server_cfg in config.servers.items():
            if not server_cfg.auto_connect:
                continue
            await self.open(name)

    async def close_all(self) -> None:
        for name in list(self._sessions):
            await self.close(name)

    # --- internal ---

    def _load_config(self) -> MCPHubConfig:
        # CLI 在另一个进程写 config；invalidate 掉缓存才能读到跨进程的改动。
        self._config_store.invalidate(MCPHubConfig)
        return self._config_store.get_or_create(MCPHubConfig(servers={}))

    def _resolve_token_storage(self, name: str):
        """解析 OAuth token 存储 —— 从 IoC get，None 则回退 ConfigStore 版。

        未来 credential Contract：Project IoC 注册加密/keychain 背书的
        ``TokenStorage``，此处 get 到就用；现在 credential 未实现，恒回退到
        :class:`ConfigTokenStorage`（明文落 config，见 mcp/auth.py 安全边界）。
        """
        from ghoshell_moss.mcp.auth import ConfigTokenStorage, TokenStorage

        try:
            from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
            storage = CommandUtil.get_contract(TokenStorage)
        except Exception:
            storage = None
        if storage is not None:
            return storage
        return ConfigTokenStorage(self._config_store, name)

    def _build_oauth_provider(self, server_cfg: MCPServerConfig) -> 'object | None':
        """OAuth server 的非交互 provider：用已落盘 token 自动 refresh，不弹浏览器。

        交互授权在 CLI ``moss mcp auth <name>`` 完成；hub 运行时只消费 token。
        """
        if server_cfg.auth.kind != 'oauth':
            return None
        from ghoshell_moss.mcp.auth import build_oauth_provider

        storage = self._resolve_token_storage(server_cfg.name)
        return build_oauth_provider(
            server_url=server_cfg.url,
            storage=storage,
            client_name=f"moss-{server_cfg.name}",
        )


# ---------------------------------------------------------------------------
# server 子 channel — 每个打开的 server 投影为一个虚拟子 channel
# ---------------------------------------------------------------------------

def _new_server_channel(session: MCPServerSession) -> Channel:
    name = session.config.name
    chan = new_channel(
        name=name,
        description=session.config.description or f"MCP server {name}",
    )

    @chan.build.notice
    async def _notice() -> str:
        return render_tools_notice(session)

    @chan.build.command(blocking=True, always_observe=True)
    async def call(tool: str, text__: str = '', timeout: float = 30.0) -> Observe:
        """阻塞调用本 server 的 MCP 工具，等待返回后才执行同 channel 后续命令。

        :param tool: 工具名
        :param text__: JSON 格式的调用参数
        :param timeout: 超时秒数
        """
        return await _call(session, tool, text__, timeout)

    @chan.build.command(blocking=False, always_observe=True)
    async def acall(tool: str, text__: str = '', timeout: float = 30.0) -> Observe:
        """非阻塞调用本 server 的 MCP 工具，结果在下一关键帧以 Observe 观察。

        :param tool: 工具名
        :param text__: JSON 格式的调用参数
        :param timeout: 超时秒数
        """
        return await _call(session, tool, text__, timeout)

    return chan


async def _call(session: MCPServerSession, tool: str, text__: str, timeout: float) -> Observe:
    if text__:
        try:
            arguments = json.loads(text__)
        except json.JSONDecodeError as e:
            return Observe.new(f"[mcp:{session.config.name}/{tool}] invalid JSON arguments: {e}")
    else:
        arguments = {}
    return await session.call_tool(tool, arguments, timeout=timeout)


def _reconcile_children(hub: MCPHub, chan: Channel) -> None:
    """把 ``hub._sessions`` 与虚拟子 channel 对齐（在 refresh_meta 里调用）。"""
    children = {c.name(): c for c in chan.virtual_children().values()}
    for name in list(children):
        if name not in hub.sessions():
            chan.remove_virtual_channel(name)
    for name, session in hub.sessions().items():
        if name not in children:
            chan.add_virtual_channel(_new_server_channel(session), alias=name)


# ---------------------------------------------------------------------------
# config store resolution + channel factory
# ---------------------------------------------------------------------------

def resolve_mcp_config_store(matrix: Matrix) -> ConfigStore:
    """MCP hub 的配置存储 — 当前直接用 workspace 级 ``matrix.configs``。

    单一真相源：hub 与 ``moss mcp add/remove`` CLI 读写同一处，
    跨进程改动经 hub 侧 ``_load_config`` 的 invalidate 生效。
    """
    return matrix.configs


def build_mcp_hub_channel(
    matrix: Matrix,
    name: str = 'mcp',
    description: str = '',
    allow_config_edit: bool = False,
) -> Channel:
    """构建 MCP Hub Channel：解析 config store → 构造 MCPHub → 投影成 channel。

    :param matrix: Matrix 实例
    :param name: channel 名称
    :param description: channel 描述
    :param allow_config_edit: 是否挂出 add/remove 命令（动态改配置）
    """
    config_store = resolve_mcp_config_store(matrix)
    hub = MCPHub(
        config_store=config_store,
        name=name,
        description=description,
        allow_config_edit=allow_config_edit,
    )
    chan = new_channel(name=name, description=hub.description(), uid=hub.id())

    @chan.build.startup
    async def _startup() -> None:
        await hub.open_auto_connect()

    @chan.build.close
    async def _close() -> None:
        await hub.close_all()

    @chan.build.refresh_meta
    async def _refresh() -> None:
        _reconcile_children(hub, chan)

    @chan.build.notice
    async def _notice() -> str:
        sessions = hub.sessions()
        if not sessions:
            return "No MCP servers opened."
        parts = []
        for name, session in sessions.items():
            icon = {'connected': '+', 'connecting': '~', 'disconnected': '-', 'error': '!'}.get(
                session.state, '?'
            )
            parts.append(f"[{icon}] {name}")
        return f"{len(parts)} MCP server(s) opened:\n" + "\n".join(parts)

    @chan.build.command(always_observe=True)
    async def list() -> str:
        """列出已打开 MCP server 的状态，及配置中尚未打开的 server。"""
        return await hub.list_servers()

    @chan.build.command(always_observe=True)
    async def open(name: str) -> str:
        """打开配置中的某个 MCP server，其工具随后出现在 `mcp.<name>` 子 channel。"""
        await hub.authorize('open', name)
        return await hub.open(name)

    @chan.build.command(always_observe=True)
    async def close(name: str) -> str:
        """关闭已打开的 MCP server；配置保留，可再次 open。"""
        await hub.authorize('close', name)
        return await hub.close(name)

    if hub.allow_config_edit():
        @chan.build.command(always_observe=True)
        async def add(text__: str = '') -> str:
            """向配置新增一个 MCP server（只写配置，不连接）。text__ 为 JSON。

            :param text__: MCPServerConfig JSON，例如
                {"name": "github", "transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "env": {"GITHUB_TOKEN": "$GITHUB_TOKEN"}}
            """
            return await hub.add_server(text__)

        @chan.build.command(always_observe=True)
        async def remove(name: str) -> str:
            """从配置删除一个 MCP server；若已打开则先关闭。"""
            await hub.authorize('remove', name)
            return await hub.remove_server(name)

    return chan


def mcp_hub_channel_factory(
    name: str = 'mcp',
    description: str = '',
    allow_config_edit: bool = False,
) -> ChannelFactory:
    """返回一个 ChannelFactory：容器就绪后从 IoC 取 Matrix 构建 hub channel。"""

    def _factory(container: IoCContainer) -> Channel:
        matrix = container.force_fetch(Matrix)
        return build_mcp_hub_channel(
            matrix,
            name=name,
            description=description,
            allow_config_edit=allow_config_edit,
        )

    return _factory
