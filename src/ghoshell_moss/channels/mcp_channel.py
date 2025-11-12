from contextvars import copy_context
import json
import logging
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar

from ghoshell_moss.concepts.states import StateStore
from ghoshell_moss.depends import check_mcp

if check_mcp():
    import mcp
    import mcp.types as types
from ghoshell_container import IoCContainer
from ghoshell_moss.channels.py_channel import PyChannel
from ghoshell_moss.concepts.channel import Builder, Channel, ChannelBroker, ChannelMeta
from ghoshell_moss.concepts.command import (
    Command,
    CommandMeta,
    CommandTask,
    CommandWrapper,
)
from ghoshell_common.helpers import uuid

R = TypeVar("R")  # 泛型结果类型


class MCPChannelBroker(ChannelBroker, Generic[R]):
    """MCPChannel的运行时客户端，负责对接MCP服务"""

    def __init__(
            self,
            *,
            name: str,
            container: IoCContainer,
            local_channel: Channel,
            mcp_client: mcp.ClientSession,
    ):
        self._name = name
        self._local_channel = local_channel
        self._mcp_client: Optional[mcp.ClientSession] = mcp_client  # MCP客户端实例
        self._commands: Dict[str, Command] = {}  # 映射后的Mosshell Command
        self._meta: Optional[ChannelMeta] = None  # Channel元信息
        self._running = False  # 运行状态标记
        self._logger: logging.Logger | None = None
        super().__init__(
            id=uuid(),
            container=container
        )

    def name(self) -> str:
        return self._name

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    # --- ChannelClient 核心方法实现 --- #
    async def start(self) -> None:
        """启动MCP客户端并同步工具元信息"""
        if self._running:
            return

        if self._local_channel is not None:
            await self._local_channel.bootstrap(self.container).start()

        # 同步远端工具元信息
        try:
            initialize_result = await self._mcp_client.initialize()  # 初始化MCP连接
            tools = await self._mcp_client.list_tools()

            # 转换为Mosshell Command和ChannelMeta
            self._meta = self._build_channel_meta(initialize_result, tools)
            self._running = True
        except Exception as e:
            raise RuntimeError(f"MCP tool discovery failed: {str(e)}") from e

    @property
    def states(self) -> StateStore:
        return self._local_channel.broker.states

    async def close(self) -> None:
        if not self._running:
            return

        # if self._mcp_client:
        #     await self._mcp_client.disconnect()
        # self._commands.clear()
        # self._meta = None
        # self._running = False

    def is_running(self) -> bool:
        return self._running

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        if not self.is_running():
            raise RuntimeError(f'Channel client {self._name} is not running')
        return self._meta.model_copy()

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        meta = self.meta(no_cache=False)
        result = {}
        for command_meta in meta.commands:
            if not available_only or command_meta.available:
                func = self._get_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def get_command(self, name: str) -> Optional[Command]:
        meta = self.meta()
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_command_func(command_meta)
                return CommandWrapper(meta=command_meta, func=func)
        return None

    def _get_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        name = meta.name
        # 优先尝试从 local channel 中返回.
        if self._local_channel is not None:
            command = self._local_channel.broker.get_command(name)
            if command is not None:
                self.logger.info(f"Channel {self._name} found command `{meta.name}` from local")
                return command.__call__

        # 回调服务端.
        async def _server_caller_as_command(*args, **kwargs):
            # 调用MCP客户端执行工具
            try:
                return await self._mcp_client.call_tool(
                    name=meta.name,
                    arguments=kwargs,
                )
            except mcp.McpError as e:
                raise ConnectionError(f"MCP call failed: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"MCP tool execution failed: {str(e)}") from e

        return _server_caller_as_command

    async def execute(self, task: CommandTask[R]) -> R:
        if not self.is_running():
            raise RuntimeError("MCPChannel is not running")
        func = self._get_command_func(task.meta)
        if func is None:
            raise LookupError(f'Channel {self._name} can find command {task.meta.name}')
        return await func(*task.args, **task.kwargs)

        # --- 工具转Command的核心逻辑 --- #

    def _convert_tools_to_command_metas(self, tools: List[types.Tool]) -> List[CommandMeta]:
        """将MCP工具转换为Mosshell的CommandMeta"""
        metas = []
        for tool in tools:
            tool_name = tool.name

            # 生成符合Code as Prompt的interface（模型可见的函数签名）
            interface = self._generate_code_as_prompt(tool)

            metas.append(CommandMeta(
                name=tool_name,
                description=tool.description or "",
                chan=self._name,
                interface=interface,
                doc=tool.description,
                available=True,
            ))
        return metas

    def _generate_code_as_prompt(self, tool: types.Tool) -> str:
        """生成模型可见的Command接口（Code as Prompt）"""

        def _parse_input_schema(input_schema: Dict[str, Any], error_prefix=""):
            """解析inputSchema并提取参数信息和参数文档"""
            # todo: 考虑直接将 json schema 作为 text__ 参数.
            params = []
            param_docs = []
            if input_schema:
                try:
                    # 解析inputSchema
                    schema = input_schema
                    if isinstance(schema, str):
                        schema = json.loads(schema)

                    if 'properties' in schema:
                        required_params = []
                        optional_params = []
                        required_param_docs = []
                        optional_param_docs = []

                        for param_name, param_info in schema['properties'].items():
                            # 确定参数类型
                            param_type = 'Any'
                            if 'type' in param_info:
                                if param_info['type'] == 'string':
                                    param_type = 'str'
                                elif param_info['type'] == 'integer':
                                    param_type = 'int'
                                elif param_info['type'] == 'number':
                                    param_type = 'float'
                                elif param_info['type'] == 'boolean':
                                    param_type = 'bool'
                                elif param_info['type'] == 'array':
                                    param_type = 'list'
                                elif param_info['type'] == 'object':
                                    param_type = 'dict'

                            # 确定默认值
                            param_str = f"{param_name}: {param_type}"
                            if param_name not in schema.get('required', []):
                                default_value = 'None' if param_type != 'bool' else 'False'
                                param_str += f"={default_value}"

                            # 根据是否必需参数，添加到不同的列表
                            if param_name in schema.get('required', []):
                                required_params.append(param_str)
                                # 添加参数文档
                                if 'description' in param_info:
                                    required_param_docs.append(f"    :param {param_name}: {param_info['description']}")
                            else:
                                optional_params.append(param_str)
                                # 添加参数文档
                                if 'description' in param_info:
                                    optional_param_docs.append(f"    :param {param_name}: {param_info['description']}")

                        # 合并列表，必需参数在前，可选参数在后
                        params = required_params + optional_params
                        param_docs = required_param_docs + optional_param_docs
                except Exception as e:
                    print(f"{error_prefix}解析inputSchema出错: {e}")
            return params, param_docs

        # 提取函数名（将连字符替换为下划线）
        function_name = tool.name.replace('-', '_')

        # 提取参数信息
        params, param_docs = _parse_input_schema(tool.inputSchema, "")

        # 生成Async函数签名（符合Python语法）
        return (
            f"async def {function_name}({', '.join(params)}) -> Any:\n"
            f"    '''\n"
            f"    {tool.description}\n"
            f"    {param_docs}\n"
            f"    '''\n"
            f"    pass"
        )

    def _build_channel_meta(self, initialize_result: types.InitializeResult,
                            tool_result: types.ListToolsResult) -> ChannelMeta:
        """构建Channel元信息（包含所有工具的CommandMeta）"""
        return ChannelMeta(
            name=self._name,
            channel_id=self._name,
            available=True,
            description=initialize_result.instructions or "",
            commands=self._convert_tools_to_command_metas(tools=tool_result.tools),
            children=[],
        )

    # --- 未使用的生命周期方法（默认空实现） --- #
    async def policy_run(self) -> None:
        pass

    async def policy_pause(self) -> None:
        pass

    async def clear(self) -> None:
        pass

    def is_available(self) -> bool:
        return True


class MCPChannel(Channel):
    """对接MCP服务的Channel"""

    def __init__(
            self,
            *,
            name: str,
            description: str,
            block: bool = True,
            mcp_client: mcp.ClientSession,
    ):
        self._name = name
        self._local_channel = PyChannel(name=name, description=description, block=block)

        self._mcp_client = mcp_client
        self._client: Optional[MCPChannelBroker] = None

    # --- Channel 核心方法实现 --- #
    def name(self) -> str:
        return self._name

    @property
    def broker(self) -> ChannelBroker:
        if not self._client or not self._client.is_running():
            raise RuntimeError("MCPChannel not bootstrapped")
        return self._client

    @property
    def build(self) -> Builder:
        return self._local_channel.build

    def bootstrap(self, container: Optional[IoCContainer] = None) -> ChannelBroker:
        if self._client is not None and self._client.is_running():
            raise RuntimeError(f'Channel {self} has already been started.')

        self._client = MCPChannelBroker(
            name=self._name,
            container=container,
            local_channel=self._local_channel,
            mcp_client=self._mcp_client,
        )

        return self._client

    # --- 未使用的Channel方法（默认空实现） --- #
    def import_channels(self, *children: Channel) -> Channel:
        raise NotImplementedError("MCPChannel does not support children")

    def new_child(self, name: str) -> Channel:
        raise NotImplementedError("MCPChannel does not support children")

    def children(self) -> Dict[str, Channel]:
        return {}

    def is_running(self) -> bool:
        return self._client is not None and self._client.is_running()
