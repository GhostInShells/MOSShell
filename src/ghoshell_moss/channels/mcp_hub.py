"""MCP Hub — 将 MCP 协议降级为纯 transport，CTML 接管调度。

Hub 作为 stateful channel 管理 N 个 MCP client session。
模型以原生 CTML 思路操作外部工具，不感知 MCP 协议存在。

Example:
    from ghoshell_moss.channels.mcp_hub import MCPHubChannel
    main.import_channels(MCPHubChannel(name='mcp', scopes=['ghost', 'mode']))
"""

import asyncio
import json
import contextlib
from dataclasses import dataclass, field
from typing import Literal, Optional

from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command, Observe
from ghoshell_moss.core.blueprint.states_channel import new_stateful_channel_from_main, ChannelState
from ghoshell_moss.core.blueprint.matrix import Matrix, ScopeKey
from ghoshell_moss.contracts.configs import ConfigType, ConfigStore, YamlConfigStore
from ghoshell_moss.message import Message, Text, Base64Image, unique_id
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field

try:
    import mcp
    from mcp import types as mcp_types
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    raise ImportError("mcp hub requires ghoshell-moss[mcp]. run: uv sync --all-extras")

__all__ = ['MCPHubChannel', 'build_mcp_hub_channel', 'MCPHubState',
           'MCPServerConfig', 'MCPHubConfig', 'MCPServerSession',
           'mcp_result_to_observe', 'render_input_schema']


class MCPServerConfig(BaseModel):
    """单个 MCP server 的连接配置（运行时使用，非模型可见）。"""

    name: str = Field(description="server 名称，作为 exec 的 server 参数")
    transport: Literal['stdio', 'sse', 'streamable_http'] = Field(
        default='stdio',
        description="传输协议",
    )
    description: str = Field(default='', description="server 描述")

    # stdio transport
    command: str = Field(default='', description="stdio: 可执行文件路径")
    args: list[str] = Field(default_factory=list, description="stdio: 命令行参数")
    env: dict[str, str] = Field(default_factory=dict, description="stdio: 环境变量")

    # sse / streamable_http transport
    url: str = Field(default='', description="sse/streamable_http: 服务 URL")
    headers: dict[str, str] = Field(default_factory=dict, description="sse/streamable_http: 请求头")


class MCPHubConfig(ConfigType):
    """MCP Hub 的完整配置，存储在 scoped storage 中。"""

    servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="server_name → config",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "mcp_hub"


# ---------------------------------------------------------------------------
# MCP server session wrapper
# ---------------------------------------------------------------------------

@dataclass
class MCPServerSession:
    """管理单个 MCP server 的连接生命周期。"""

    config: MCPServerConfig
    client: mcp.ClientSession | None = None
    tools: list[mcp_types.Tool] = field(default_factory=list)
    state: str = 'disconnected'  # connected | disconnected | error
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
            transport = await self._exit_stack.enter_async_context(stdio_client(params))
            return transport
        elif cfg.transport == 'sse':
            read, write = await self._exit_stack.enter_async_context(
                sse_client(url=cfg.url, headers=cfg.headers or None)
            )
            return read, write
        elif cfg.transport == 'streamable_http':
            transport = await self._exit_stack.enter_async_context(
                streamable_http_client(cfg.url)
            )
            read, write, _ = transport
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
# MCP result → Observe
# ---------------------------------------------------------------------------

def mcp_result_to_observe(
    result: mcp_types.CallToolResult,
    *,
    server: str,
    tool: str,
) -> Observe:
    """将 MCP CallToolResult 转为 Observe，只保留 text + image 两种 content。"""
    if result.isError:
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
                    Base64Image.from_base64(media_type=c.mimeType, data=c.data)
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
# MCP Hub State
# ---------------------------------------------------------------------------

class MCPHubState(ChannelState):
    """管理 N 个 MCP client session 的 ChannelState。"""

    def __init__(
        self,
        *,
        config_store: ConfigStore,
        allow_model_config: bool = False,
        name: str = 'mcp',
        description: str = '',
    ):
        self._config_store = config_store
        self._allow_model_config = allow_model_config
        self._name = name
        self._description = description or 'MCP Hub — 管理外部 MCP 工具调用'
        self._uid = unique_id()
        self._sessions: dict[str, MCPServerSession] = {}
        self._own_commands: dict[str, Command] = {}
        self._bootstrap()

    def _bootstrap(self) -> None:
        from ghoshell_moss.core.concepts.command import PyCommand

        async def exec_cmd(server: str, tool: str, timeout: float = 30.0, text__: str = '') -> Observe:
            """调用 MCP 工具, 非阻塞。结果在下一关键帧以 Observe 形式观察。

            :param server: MCP server 名称
            :param tool: 工具名称
            :param timeout: 超时秒数
            :param text__: JSON 格式的调用参数: {"key": "val"}
            """
            session = self._sessions.get(server)
            if session is None:
                return Observe.new(f"[MCP:{server}] server not connected. Use list to see available servers.")
            try:
                arguments = json.loads(text__) if text__ else {}
            except json.JSONDecodeError as e:
                return Observe.new(f"[MCP:{server}/{tool}] invalid JSON arguments: {e}")
            return await session.call_tool(tool, arguments, timeout=timeout)

        async def exec_blocking_cmd(server: str, tool: str, timeout: float = 30.0, text__: str = '') -> Observe:
            """阻塞调用 MCP 工具。等待返回后才执行同通道后续命令。

            :param server: MCP server 名称
            :param tool: 工具名称
            :param timeout: 超时秒数
            :param text__: JSON 格式的调用参数
            """
            return await exec_cmd(server, tool, timeout=timeout, text__=text__)

        async def list_servers() -> str:
            """列出所有 MCP server 的连接状态及配置中尚未连接的 server。"""
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
                lines.append("")
            # show config entries not yet connected
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

        async def connect_server(name: str) -> str:
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
            return f"[MCP:{name}] {session.state}" + (f": {session.error}" if session.error else '')

        async def disconnect_server(name: str) -> str:
            """断开并移除 MCP server 连接。

            :param name: server 名称
            """
            session = self._sessions.pop(name, None)
            if session is None:
                return f"[MCP:{name}] not connected"
            await session.disconnect()
            return f"[MCP:{name}] disconnected"

        async def reconnect_server(name: str) -> str:
            """重新连接 MCP server。

            :param name: server 名称
            """
            session = self._sessions.get(name)
            if session is None:
                return f"[MCP:{name}] not connected. Use connect first."
            await session.disconnect()
            await session.connect()
            return f"[MCP:{name}] {session.state}" + (f": {session.error}" if session.error else '')

        self._own_commands = {
            'exec': PyCommand(exec_cmd, blocking=False, always_observe=True),
            'exec_blocking': PyCommand(exec_blocking_cmd, blocking=True, always_observe=True),
            'list': PyCommand(list_servers, always_observe=True),
            'connect': PyCommand(connect_server, always_observe=True),
            'disconnect': PyCommand(disconnect_server, always_observe=True),
            'reconnect': PyCommand(reconnect_server, always_observe=True),
        }

        if self._allow_model_config:
            async def register(text__: str = '') -> str:
                """注册新的 MCP server 配置并持久化，可选自动连接。

                :param text__: JSON 格式的 MCPServerConfig, 可包含 "connect": true 立即连接。
                e.g. {"name": "myserver", "transport": "stdio", "command": "python", "args": ["-m", "mymod"], "env": {"KEY": "$SECRET"}, "connect": true}
                """
                try:
                    data = json.loads(text__) if text__ else {}
                except json.JSONDecodeError as e:
                    return f"[MCP] invalid JSON: {e}"

                connect = data.pop('connect', True)
                try:
                    server_cfg = MCPServerConfig.model_validate(data)
                except Exception as e:
                    return f"[MCP] invalid server config: {e}"

                config = self._load_config()
                config.servers[server_cfg.name] = server_cfg
                self._config_store.save(config)

                if connect:
                    if server_cfg.name in self._sessions:
                        await self._sessions[server_cfg.name].disconnect()
                    session = MCPServerSession(config=server_cfg)
                    await session.connect()
                    self._sessions[server_cfg.name] = session
                    return f"[MCP:{server_cfg.name}] registered and {session.state}" + (f": {session.error}" if session.error else '')
                return f"[MCP:{server_cfg.name}] registered (not connected)"

            async def unregister(name: str) -> str:
                """移除 MCP server 配置并断开连接。

                :param name: server 名称
                """
                config = self._load_config()
                if name not in config.servers:
                    return f"[MCP:{name}] not found in config"

                session = self._sessions.pop(name, None)
                if session:
                    await session.disconnect()

                del config.servers[name]
                self._config_store.save(config)
                return f"[MCP:{name}] unregistered"

            self._own_commands['register'] = PyCommand(register, always_observe=True)
            self._own_commands['unregister'] = PyCommand(unregister, always_observe=True)

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

    def _load_config(self) -> MCPHubConfig:
        return self._config_store.get_or_create(MCPHubConfig(servers={}))

    async def on_startup(self) -> None:
        """启动时加载配置并连接所有 server。ConfigStore 自动解析 $VAR 占位符。"""
        config = self._load_config()
        for name, server_cfg in config.servers.items():
            session = MCPServerSession(config=server_cfg)
            await session.connect()
            self._sessions[name] = session

    async def on_close(self) -> None:
        """关闭所有 MCP 连接。"""
        for session in self._sessions.values():
            await session.disconnect()
        self._sessions.clear()

    async def get_context_messages(self) -> list[str]:
        """动态生成 server 状态、工具目录、参数定义和可用 preset。"""
        lines = ["### MCP Tools"]
        for name, session in self._sessions.items():
            state_mark = {'connected': '+', 'connecting': '~', 'disconnected': '-', 'error': '!'}.get(
                session.state, '?'
            )
            lines.append(f"[{state_mark}] **{name}**")
            if session.state == 'connected':
                for tool in session.tools:
                    desc = (tool.description or '').split('\n')[0][:120]
                    lines.append(f"  - `{tool.name}`: {desc}")
                    params = render_input_schema(tool.inputSchema)
                    if params:
                        lines.append(f"    params: {params}")
            elif session.error:
                lines.append(f"  error: {session.error[:200]}")
            lines.append("")
        config = self._load_config()
        available = {n: c.description or '' for n, c in config.servers.items()
                     if n not in self._sessions}
        if available:
            if self._sessions:
                lines.append("Available:")
            else:
                lines.append("No MCP servers connected. Available:")
            for n in sorted(available):
                desc = f" — {available[n]}" if available[n] else ''
                lines.append(f"- `{n}`{desc}")
        elif not self._sessions:
            lines.append("No MCP servers available.")
        return ['\n'.join(lines)]


# ---------------------------------------------------------------------------
# MCP Hub Channel + factory
# ---------------------------------------------------------------------------

class MCPHubChannel(Channel):
    """MCP Hub Channel — 将 MCP 协议降级为 transport，CTML 接管调度。"""

    def __init__(
        self,
        name: str = 'mcp',
        description: str = '',
        scopes: list[ScopeKey] | None = None,
        allow_model_config: bool = False,
    ):
        self._name = name
        self._description = description or 'MCP Hub — 管理外部 MCP 工具调用'
        self._scopes = scopes or []
        self._allow_model_config = allow_model_config
        self._id = unique_id()

    def name(self) -> ChannelName:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    def materialize(self, container: IoCContainer) -> ChannelRuntime:
        matrix = container.force_fetch(Matrix)

        if self._scopes:
            storage = matrix.get_scoped_storage(*self._scopes)
            config_store = YamlConfigStore(storage)
            # 首次创建时从 workspace 预设合并，确保 scoped 配置包含全局预设的 server
            workspace_store = matrix.configs
            try:
                workspace_config = workspace_store.get(MCPHubConfig)
                scoped_config = config_store.get_or_create(
                    MCPHubConfig(servers=dict(workspace_config.servers))
                )
                config_store.save(scoped_config)
            except Exception:
                config_store.get_or_create(MCPHubConfig(servers={}))
        else:
            config_store = matrix.configs

        state = MCPHubState(
            config_store=config_store,
            allow_model_config=self._allow_model_config,
            name=self._name,
            description=self._description,
        )
        channel = new_stateful_channel_from_main(state, id=self._id)
        return channel.bootstrap(container)


def build_mcp_hub_channel(
    matrix: Matrix,
    name: str = 'mcp',
    description: str = '',
    scopes: list[ScopeKey] | None = None,
    allow_model_config: bool = False,
) -> Channel:
    """构建 MCP Hub Channel 的工厂函数。

    :param matrix: Matrix 实例
    :param name: channel 名称
    :param description: channel 描述
    :param scopes: 配置存储的隔离级别, e.g. ['ghost', 'mode']
    :param allow_model_config: 是否允许模型动态注册/移除 server
    """
    if scopes:
        storage = matrix.get_scoped_storage(*scopes)
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
    else:
        config_store = matrix.configs

    state = MCPHubState(
        config_store=config_store,
        allow_model_config=allow_model_config,
        name=name,
        description=description,
    )
    return new_stateful_channel_from_main(state)
