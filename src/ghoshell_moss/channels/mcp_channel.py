"""MCP Hub Channel — MCP 工具集以 channel 形态接入 | 集成 | beta

MCPHubState 是本源对象：持有 N 个 MCP server session，每个 server 暴露为一个
信息辐射器子 channel（只读 notice，无 command）。调用入口集中在父 channel：
``call``（阻塞）/ ``acall``（非阻塞）。模型以原生 CTML 思路操作外部工具，
不感知 MCP 协议存在。

同一个对象、两个面：构建方 / GUI 持有 MCPHubState 直接调用公开方法
（connect_server / disconnect_server / list_servers / call_tool），
模型通过 ``new_channel_from_state`` 投影成 channel 用 CTML 操作。

Example:
    from ghoshell_moss.core.blueprint.states_channel import new_shell_main_channel
    from ghoshell_moss.channels.mcp_channel import build_mcp_hub_channel
    main = new_shell_main_channel()
    main.import_channels(build_mcp_hub_channel(matrix, name='mcp', scopes=['ghost', 'mode']))
"""

import asyncio
import contextlib
import json
from dataclasses import dataclass, field

from ghoshell_moss.contracts.configs import ConfigStore, YamlConfigStore
from ghoshell_moss.core.blueprint.matrix import Matrix, RuntimeScopeKey
from ghoshell_moss.core.blueprint.states_channel import new_channel_from_state
from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelState
from ghoshell_moss.core.concepts.command import Command, Observe, PyCommand
from ghoshell_moss.mcp.config import MCPHubConfig, MCPServerConfig
from ghoshell_moss.message import Base64Image, Message, Text, unique_id

try:
    import mcp
    from mcp import types as mcp_types
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    raise ImportError("mcp not installed. run: uv sync --all-extras")

__all__ = [
    'MCPHubState',
    'MCPServerChannelState',
    'MCPServerSession',
    'build_mcp_hub_channel',
    'mcp_result_to_observe',
    'render_input_schema',
    'resolve_mcp_config_store',
]


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
        return Observe.new(f"[MCP:{server}/{tool}] error: {' '.join(text_parts)}")

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


# ---------------------------------------------------------------------------
# MCP server session wrapper
# ---------------------------------------------------------------------------

@dataclass
class MCPServerSession:
    """管理单个 MCP server 的连接生命周期。"""

    config: MCPServerConfig
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
                sse_client(url=cfg.url, headers=cfg.headers or None)
            )
        elif cfg.transport == 'streamable_http':
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
            return Observe.new(f"[MCP:{self.config.name}] server not connected")
        try:
            result = await asyncio.wait_for(
                self.client.call_tool(name=name, arguments=arguments),
                timeout=timeout,
            )
            return mcp_result_to_observe(result, server=self.config.name, tool=name)
        except asyncio.TimeoutError:
            return Observe.new(f"[MCP:{self.config.name}/{name}] timeout after {timeout}s")
        except mcp.McpError as e:
            self.state = 'error'
            self.error = str(e)
            return Observe.new(f"[MCP:{self.config.name}/{name}] MCP error: {e}")


# ---------------------------------------------------------------------------
# Server sub-channel — 信息辐射器
# ---------------------------------------------------------------------------

class MCPServerChannelState(ChannelState):
    """单个 server 的子 channel state，只做信息辐射（notice），无 command。"""

    def __init__(self, session: MCPServerSession):
        self._session = session

    def name(self) -> str:
        return self._session.config.name

    def description(self) -> str:
        return self._session.config.description or f"MCP server: {self._session.config.name}"

    def is_available(self) -> bool:
        return self._session.state == 'connected'

    def is_dynamic(self) -> bool:
        return True

    def own_commands(self) -> dict[str, Command]:
        return {}

    def get_own_command(self, name: str) -> Command | None:
        return None

    async def get_notice(self) -> str:
        """工具摘要 — 模型通过此子 channel 了解该 server 能做什么。"""
        session = self._session
        if session.state != 'connected':
            if session.error:
                return f"{session.config.name}: {session.state} — {session.error}"
            return f"{session.config.name}: {session.state}"
        tool_names = [t.name for t in session.tools]
        summary = ', '.join(tool_names)
        return f"{session.config.name}: {len(tool_names)} tools — {summary}"

    async def get_context_messages(self) -> list:
        return []


# ---------------------------------------------------------------------------
# MCP Hub State — 本源对象
# ---------------------------------------------------------------------------

class MCPHubState(ChannelState):
    """管理 N 个 MCP client session 的 ChannelState（父 hub）。

    本源对象：构建方 / GUI 持有实例直接调用公开方法
    （connect_server / disconnect_server / list_servers / call_tool / sessions），
    模型通过 new_channel_from_state 投影成 channel 用 CTML 操作。
    """

    def __init__(
        self,
        *,
        config_store: ConfigStore,
        name: str = 'mcp',
        description: str = '',
    ):
        self._config_store = config_store
        self._name = name
        self._description = description or 'MCP Hub — 管理外部 MCP 工具调用'
        self._uid = unique_id()
        self._sessions: dict[str, MCPServerSession] = {}
        self._server_channels: dict[str, Channel] = {}
        self._own_commands: dict[str, Command] = self._build_commands()

    # --- 公开方法 (构建方 / GUI 直接调用) ---

    async def connect_server(self, name: str) -> str:
        """连接指定的 MCP server。

        :param name: server 名称
        """
        if name in self._sessions and self._sessions[name].state == 'connected':
            return f"[MCP:{name}] already connected"

        config = self._load_config()
        server_cfg = config.servers.get(name)
        if server_cfg is None:
            available = sorted(config.servers.keys())
            suffix = f". Available: {', '.join(available)}" if available else ''
            return f"[MCP:{name}] not found{suffix}"

        session = MCPServerSession(config=server_cfg)
        await session.connect()
        self._sessions[name] = session
        self._server_channels[name] = new_channel_from_state(
            MCPServerChannelState(session), id=f"mcp-server-{name}"
        )
        return f"[MCP:{name}] {session.state}" + (f": {session.error}" if session.error else '')

    async def disconnect_server(self, name: str) -> str:
        """断开并移除 MCP server 连接。

        :param name: server 名称
        """
        session = self._sessions.pop(name, None)
        self._server_channels.pop(name, None)
        if session is None:
            return f"[MCP:{name}] not connected"
        await session.disconnect()
        return f"[MCP:{name}] disconnected"

    async def list_servers(self) -> str:
        """列出所有 MCP server 的连接状态、工具目录，及配置中尚未连接的 server。"""
        lines = ["### MCP Servers\n"]
        for name, session in self._sessions.items():
            state_icon = {'connected': '+', 'connecting': '~', 'disconnected': '-', 'error': '!'}.get(
                session.state, '?'
            )
            lines.append(f"[{state_icon}] **{name}** ({session.state})")
            if session.error:
                lines.append(f"    error: {session.error}")
            if session.state == 'connected':
                for tool in session.tools:
                    desc = (tool.description or '').split('\n')[0]
                    lines.append(f"    - {tool.name}: {desc}")
                    params = render_input_schema(tool.input_schema)
                    if params:
                        lines.append(f"      params: {params}")
            lines.append("")
        config = self._load_config()
        available = {n: c.description or '' for n, c in config.servers.items()
                     if n not in self._sessions}
        if available:
            lines.append("Available (not connected):")
            for n in sorted(available):
                desc = f" — {available[n]}" if available[n] else ''
                lines.append(f"- `{n}`{desc}")
        elif not self._sessions:
            lines.append("No servers configured.")
        return '\n'.join(lines)

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict | None = None,
        timeout: float = 30.0,
    ) -> Observe:
        """调用 MCP 工具，返回 Observe。

        :param server: MCP server 名称
        :param tool: 工具名称
        :param arguments: 调用参数字典
        :param timeout: 超时秒数
        """
        session = self._sessions.get(server)
        if session is None:
            return Observe.new(f"[MCP:{server}] server not connected. Use list to see available servers.")
        return await session.call_tool(tool, arguments or {}, timeout=timeout)

    def sessions(self) -> dict[str, MCPServerSession]:
        """当前已连接的 server session（同步快照，供 GUI 渲染）。"""
        return dict(self._sessions)

    # --- commands ---

    def _build_commands(self) -> dict[str, Command]:
        async def call_cmd(server: str, tool: str, timeout: float = 30.0, text__: str = '') -> Observe:
            """阻塞调用 MCP 工具，等待返回后才执行同通道后续命令。

            :param server: MCP server 名称
            :param tool: 工具名称
            :param timeout: 超时秒数
            :param text__: JSON 格式的调用参数
            """
            return await self._call_with_json(server, tool, timeout, text__)

        async def acall_cmd(server: str, tool: str, timeout: float = 30.0, text__: str = '') -> Observe:
            """非阻塞调用 MCP 工具，结果在下一关键帧以 Observe 形式观察。

            :param server: MCP server 名称
            :param tool: 工具名称
            :param timeout: 超时秒数
            :param text__: JSON 格式的调用参数
            """
            return await self._call_with_json(server, tool, timeout, text__)

        async def list_cmd() -> str:
            """列出所有 MCP server 的连接状态、工具目录，及配置中尚未连接的 server。"""
            return await self.list_servers()

        async def connect_cmd(name: str) -> str:
            """连接指定的 MCP server。

            :param name: server 名称
            """
            return await self.connect_server(name)

        async def disconnect_cmd(name: str) -> str:
            """断开并移除 MCP server 连接。

            :param name: server 名称
            """
            return await self.disconnect_server(name)

        return {
            'call': PyCommand(call_cmd, blocking=True, always_observe=True),
            'acall': PyCommand(acall_cmd, blocking=False, always_observe=True),
            'list': PyCommand(list_cmd, always_observe=True),
            'connect': PyCommand(connect_cmd, always_observe=True),
            'disconnect': PyCommand(disconnect_cmd, always_observe=True),
        }

    async def _call_with_json(self, server: str, tool: str, timeout: float, text__: str) -> Observe:
        try:
            arguments = json.loads(text__) if text__ else {}
        except json.JSONDecodeError as e:
            return Observe.new(f"[MCP:{server}/{tool}] invalid JSON arguments: {e}")
        return await self.call_tool(server, tool, arguments, timeout=timeout)

    # --- ChannelState interface ---

    def id(self) -> str:
        return self._uid

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def is_available(self) -> bool:
        return True

    def is_dynamic(self) -> bool:
        return True

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands.copy()

    def get_own_command(self, name: str) -> Command | None:
        return self._own_commands.get(name)

    def get_virtual_children(self) -> dict[ChannelName, Channel]:
        return dict(self._server_channels)

    async def get_notice(self) -> str:
        """连接摘要 — warm 数据，server 连/断时刷新。"""
        if not self._sessions:
            return "No MCP servers connected."
        parts = []
        for name, session in self._sessions.items():
            mark = {'connected': '+', 'connecting': '~', 'disconnected': '-', 'error': '!'}.get(
                session.state, '?'
            )
            tool_count = len(session.tools) if session.state == 'connected' else 0
            parts.append(f"[{mark}] {name} ({tool_count} tools)")
        header = f"{len(self._sessions)} MCP servers connected:"
        return header + "\n" + "\n".join(parts)

    async def get_context_messages(self) -> list:
        """无每帧热数据 — 工具目录在 notice（warm）与 list（on-demand）中。"""
        return []

    def _load_config(self) -> MCPHubConfig:
        return self._config_store.get_or_create(MCPHubConfig(servers={}))

    async def on_startup(self) -> None:
        """启动时连接 auto_connect=True 的 server。ConfigStore 自动解析 $VAR。"""
        config = self._load_config()
        for name, server_cfg in config.servers.items():
            if not server_cfg.auto_connect:
                continue
            await self.connect_server(name)

    async def on_close(self) -> None:
        """关闭所有 MCP 连接。"""
        for session in self._sessions.values():
            await session.disconnect()
        self._sessions.clear()
        self._server_channels.clear()


# ---------------------------------------------------------------------------
# config store resolution + channel factory
# ---------------------------------------------------------------------------

def resolve_mcp_config_store(
    matrix: Matrix,
    scopes: list[RuntimeScopeKey] | None = None,
) -> ConfigStore:
    """从 Matrix 解析 MCP hub 的配置存储，支持按 scope 隔离。"""
    if not scopes:
        return matrix.configs
    storage = matrix.get_runtime_scope_storage(*scopes)
    config_store = YamlConfigStore(storage)
    workspace_store = matrix.configs
    try:
        workspace_config = workspace_store.get(MCPHubConfig)
        scoped_config = config_store.get_or_create(
            MCPHubConfig(servers=dict(workspace_config.servers))
        )
        config_store.save(scoped_config)
    except Exception:
        config_store.get_or_create(MCPHubConfig(servers={}))
    return config_store


def build_mcp_hub_channel(
    matrix: Matrix,
    name: str = 'mcp',
    description: str = '',
    scopes: list[RuntimeScopeKey] | None = None,
) -> Channel:
    """构建 MCP Hub Channel：解析 config store → 构造 MCPHubState → 投影成 channel。

    :param matrix: Matrix 实例
    :param name: channel 名称
    :param description: channel 描述
    :param scopes: 配置存储的隔离级别, e.g. ['ghost', 'mode']
    """
    config_store = resolve_mcp_config_store(matrix, scopes)
    state = MCPHubState(config_store=config_store, name=name, description=description)
    return new_channel_from_state(state, id=state.id())
