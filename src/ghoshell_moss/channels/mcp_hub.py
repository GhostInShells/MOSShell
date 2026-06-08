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
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelRuntime, ChannelCtx
from ghoshell_moss.core.concepts.command import Command, Observe
from ghoshell_moss.core.blueprint.states_channel import new_stateful_channel_from_main, ChannelState
from ghoshell_moss.core.blueprint.matrix import Matrix, ScopesKey
from ghoshell_moss.contracts.configs import ConfigType, YamlConfigStore
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
           'mcp_result_to_observe', 'render_input_schema', 'resolve_env_dict']


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

def resolve_env_dict(env: dict[str, str]) -> dict[str, str]:
    """解析 env dict 中以 $ 开头的值，从 os.environ 读取实际值。

    Config 文件中存储 $VAR_NAME 占位符，连接时由此函数解析为真值。
    不以 $ 开头的值原样返回。
    """
    if not env:
        return env
    resolved = {}
    for k, v in env.items():
        if v.startswith('$'):
            resolved[k] = os.environ.get(v[1:], v)
        else:
            resolved[k] = v
    return resolved


class MCPServerConfig(BaseModel):
    """单个 MCP server 的连接配置。"""

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
            resolved_env = resolve_env_dict(cfg.env) if cfg.env else None
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=resolved_env,
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
        matrix: Matrix,
        name: str = 'mcp',
        description: str = '',
        scopes: list[ScopesKey] | None = None,
    ):
        self._matrix = matrix
        self._name = name
        self._description = description or 'MCP Hub — 管理外部 MCP 工具调用'
        self._scopes = scopes or []
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
            :param text__: JSON 格式的调用参数, 放在开放闭合标签内: <mcp:exec server="x" tool="y">{"key":"val"}</mcp:exec>
            """
            session = self._sessions.get(server)
            if session is None:
                return Observe.new(f"[MCP:{server}] server not found. Use list_servers to see available servers.")
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
            """列出所有 MCP server 的连接状态和工具摘要。"""
            lines = ["### MCP Server Status\n"]
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
            if not self._sessions:
                lines.append("No servers configured. Use add_server to add one.")
            return '\n'.join(lines)

        async def add_server(
            name: str,
            transport: str = '',
            command: str = '',
            args: str = '',
            url: str = '',
            env: str = '',
            description: str = '',
        ) -> str:
            """添加并连接 MCP server。可运行时传参或从配置加载。

            :param name: server 名称
            :param transport: 传输协议 (stdio / sse / streamable_http)。传参时必填
            :param command: stdio: 可执行文件路径
            :param args: stdio: 命令行参数，逗号分隔
            :param url: sse/streamable_http: 服务 URL
            :param env: 环境变量，逗号分隔的 KEY=VALUE 对。value 以 $ 开头时从系统环境变量读取（如 BAIDU_MAPS_API_KEY=$BAIDU_MAPS_API_KEY），避免在 CTML 中暴露敏感信息
            :param description: server 描述
            """
            if name in self._sessions and self._sessions[name].state == 'connected':
                return f"[MCP:{name}] already connected"

            config = self._load_config()

            if transport:
                parsed_env = {}
                if env:
                    for pair in env.split(','):
                        pair = pair.strip()
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            k, v = k.strip(), v.strip()
                            if v.startswith('$'):
                                if v[1:] not in os.environ:
                                    return (
                                        f"[MCP:{name}] env var '{v[1:]}' not set. "
                                        f"请在环境变量中配置 {v[1:]} 后重试。"
                                    )
                                # 存储 $VAR 占位符，解析为真值在 _connect_transport 中进行
                            parsed_env[k] = v
                parsed_args = [a.strip() for a in args.split(',') if a.strip()] if args else []
                server_cfg = MCPServerConfig(
                    name=name,
                    transport=transport,  # type: ignore[arg-type]
                    command=command,
                    args=parsed_args,
                    url=url,
                    env=parsed_env,
                    description=description,
                )
                config.servers[name] = server_cfg
                self._save_config(config)
            else:
                server_cfg = config.servers.get(name)
                if server_cfg is None:
                    available = list(config.servers.keys())
                    return f"[MCP] Server '{name}' not in config. Available: {', '.join(available)}"

            session = MCPServerSession(config=server_cfg)
            await session.connect()
            self._sessions[name] = session
            return f"[MCP:{name}] {session.state}" + (f": {session.error}" if session.error else '')

        async def remove_server(name: str) -> str:
            """断开并移除 MCP server。

            :param name: 要移除的 server 名称
            """
            session = self._sessions.pop(name, None)
            if session is None:
                return f"[MCP:{name}] server not found"
            await session.disconnect()
            return f"[MCP:{name}] removed"

        async def restart_server(name: str) -> str:
            """重启 MCP server 连接。

            :param name: server 名称
            """
            session = self._sessions.get(name)
            if session is None:
                return f"[MCP:{name}] server not found. Use add_server to add it first."
            await session.disconnect()
            await session.connect()
            return f"[MCP:{name}] {session.state}" + (f": {session.error}" if session.error else '')

        self._own_commands = {
            'exec': PyCommand(exec_cmd, blocking=False, always_observe=True),
            'exec_blocking': PyCommand(exec_blocking_cmd, blocking=True, always_observe=True),
            'list_servers': PyCommand(list_servers, always_observe=True),
            'add_server': PyCommand(add_server, always_observe=True),
            'remove_server': PyCommand(remove_server, always_observe=True),
            'restart_server': PyCommand(restart_server, always_observe=True),
        }

    # --- ChannelState interface ---

    def id(self) -> str:
        return self._uid

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def is_available(self) -> bool:
        return self._matrix.is_running()

    def is_dynamic(self) -> bool:
        return True

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands.copy()

    def get_own_command(self, name: str) -> Command | None:
        return self._own_commands.get(name)

    def _load_config(self) -> MCPHubConfig:
        """加载 MCP Hub 配置。有 scopes 走 scoped storage YAML，无 scopes 走全局 ConfigStore。
        首次调用自动初始化空配置并持久化。
        """
        if self._scopes:
            storage = self._matrix.get_scoped_storage(*self._scopes)
            config = storage.read_yaml("mcp_hub", MCPHubConfig)
            if config is None:
                config = MCPHubConfig(servers={})
                storage.write_yaml("mcp_hub", config)
            return config
        else:
            try:
                from ghoshell_moss.contracts.configs import get_conf
                return get_conf(ChannelCtx.container(), MCPHubConfig)
            except Exception:
                config = MCPHubConfig(servers={})
                from ghoshell_moss.contracts.configs import save_conf
                save_conf(ChannelCtx.container(), config)
                return config

    def _save_config(self, config: MCPHubConfig) -> None:
        """持久化 MCP Hub 配置。"""
        if self._scopes:
            storage = self._matrix.get_scoped_storage(*self._scopes)
            storage.write_yaml("mcp_hub", config)
        else:
            from ghoshell_moss.contracts.configs import save_conf
            save_conf(ChannelCtx.container(), config)

    async def on_startup(self) -> None:
        """启动时加载配置并连接所有 server。"""
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
        """动态生成 server 状态、工具目录和参数定义。"""
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
        if not self._sessions:
            lines.append("No MCP servers connected.")
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
        scopes: list[ScopesKey] | None = None,
    ):
        self._name = name
        self._description = description or 'MCP Hub — 管理外部 MCP 工具调用'
        self._scopes = scopes or []
        self._id = unique_id()

    def name(self) -> ChannelName:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    def materialize(self, container: IoCContainer) -> ChannelRuntime:
        matrix = container.force_fetch(Matrix)
        state = MCPHubState(
            matrix=matrix,
            name=self._name,
            description=self._description,
            scopes=self._scopes,
        )
        channel = new_stateful_channel_from_main(state, id=self._id)
        return channel.bootstrap(container)


def build_mcp_hub_channel(
    matrix: Matrix,
    name: str = 'mcp',
    description: str = '',
    scopes: list[ScopesKey] | None = None,
) -> Channel:
    """构建 MCP Hub Channel 的工厂函数。

    :param matrix: Matrix 实例
    :param name: channel 名称
    :param description: channel 描述
    :param scopes: 配置存储的隔离级别, e.g. ['ghost', 'mode']
    """
    state = MCPHubState(
        matrix=matrix,
        name=name,
        description=description,
        scopes=scopes,
    )
    return new_stateful_channel_from_main(state)
