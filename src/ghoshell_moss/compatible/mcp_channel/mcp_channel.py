import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Generic, Optional, TypeVar

from ghoshell_moss import CommandError, CommandErrorCode
from ghoshell_moss.compatible.mcp_channel.utils import mcp_call_tool_result_to_message
from ghoshell_moss.core.concepts.states import BaseStateStore, StateStore

try:
    import mcp
    from mcp import types
except ImportError:
    raise ImportError("Could not import mcp. Please install ghoshell-moss[mcp].")

import asyncio

from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer

from ghoshell_moss.core.concepts.channel import Builder, Channel, ChannelRuntime, ChannelMeta
from ghoshell_moss.core.concepts.command import (
    Command,
    CommandDeltaType,
    CommandMeta,
    CommandTask,
    CommandWrapper,
)

R = TypeVar("R")  # 泛型结果类型


class MCPChannelRuntime(ChannelRuntime, Generic[R]):
    """MCPChannel的运行时客户端，负责对接MCP服务"""

    MCP_CONTAINER_TYPES: list[str] = ["array", "object"]

    MCP_PY_TYPES_TRANS_TABLE: dict[str, str] = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }

    COMMAND_DELTA_PARAMETER: str = f"{CommandDeltaType.TEXT.value}:str"

    def __init__(
        self,
        *,
        name: str,
        mcp_client: mcp.ClientSession,
        container: Optional[IoCContainer] = None,
        blocking: bool = False,
    ):
        self._name = name
        self._mcp_client: Optional[mcp.ClientSession] = mcp_client  # MCP客户端实例
        self._commands: dict[str, Command] = {}  # 映射后的Mosshell Command
        self._meta: Optional[ChannelMeta] = None  # Channel元信息
        self._running = False  # 运行状态标记
        self._logger: logging.Logger | None = None
        self._id = uuid()
        self._container = Container(parent=container, name="mcp_channel:" + self._name)
        self._states: Optional[StateStore] = None
        self._blocking = blocking

    def sub_channels(self) -> dict[str, "Channel"]:
        return {}

    @property
    def container(self) -> IoCContainer:
        return self._container

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    # --- ChannelRuntime 核心方法实现 --- #
    async def start(self) -> None:
        """启动MCP客户端并同步工具元信息"""
        if self._running:
            return

        # 同步远端工具元信息
        try:
            await asyncio.to_thread(self._container.bootstrap)
            initialize_result = await self._mcp_client.initialize()  # 初始化MCP连接
            tools = await self._mcp_client.list_tools()

            # 转换为Mosshell Command和ChannelMeta
            self._meta = self._build_channel_meta(initialize_result, tools)
            self._running = True
        except Exception as e:
            raise RuntimeError(f"MCP tool discovery failed: {str(e)}") from e

    @property
    def states(self) -> StateStore:
        if self._states is None:
            _states = self._container.get(StateStore)
            if _states is None:
                _states = BaseStateStore(self._name)
                self._container.set(StateStore, _states)
            self._states = _states
        return self._states

    async def close(self) -> None:
        if not self._running:
            return
        await asyncio.to_thread(self._container.shutdown)

    def is_running(self) -> bool:
        return self._running

    def own_meta(self) -> ChannelMeta:
        # todo: 还没有实现动态更新, 主要是更新 command
        if not self.is_running():
            raise RuntimeError(f"Channel client {self._name} is not running")
        return self._meta.model_copy()

    async def refresh_all_metas(self) -> None:
        # todo: shall refresh command metas
        return None

    def is_connected(self) -> bool:
        # todo: 检查状态.
        return self.is_running()

    async def wait_connected(self) -> None:
        # todo: 检查状态.
        return

    def own_commands(self, available_only: bool = True) -> dict[str, Command]:
        # todo: 这里每次更新, 和上面好像冲突.
        meta = self.own_meta()
        result = {}
        for command_meta in meta.commands:
            if not available_only or command_meta.available:
                func = self._get_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def get_self_command(self, name: str) -> Optional[Command]:
        meta = self.own_meta()
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_command_func(command_meta)
                return CommandWrapper(meta=command_meta, func=func)
        return None

    def _get_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        name = meta.name

        args_schema_properties = meta.args_schema.get("properties", {})
        required_args_list = meta.args_schema.get("required", [])
        schema_param_count = len(args_schema_properties)
        required_schema_param_count = len(required_args_list)

        # 回调服务端.
        async def _server_caller_as_command(*args, **kwargs):
            # 调用MCP客户端执行工具
            try:
                if required_schema_param_count > schema_param_count:
                    raise CommandError(
                        code=CommandErrorCode.VALUE_ERROR.value,
                        message=(
                            "MCP tool: invalid parameter count, required parameter: "
                            f"{required_schema_param_count}, schema parameter: {schema_param_count}"
                        ),
                    )

                param_count = len(args) + len(kwargs)
                final_kwargs = {}
                if schema_param_count == 0:  # do nothing
                    if not param_count == 0:
                        raise CommandError(
                            code=CommandErrorCode.VALUE_ERROR.value,
                            message=f"MCP tool: no parameter, invalid, args={args}, kwargs={kwargs}",
                        )
                else:  # schema_param_count > 1
                    if not (param_count == 1 or required_schema_param_count <= param_count <= schema_param_count):
                        raise CommandError(
                            code=CommandErrorCode.VALUE_ERROR.value,
                            message=f"MCP tool: invalid parameters, invalid, args={args}, kwargs={kwargs}",
                        )
                    if param_count == 1:
                        if len(args) == 1:
                            if required_schema_param_count == 1:
                                if type(args[0]) is not str:
                                    [param_name, param_info], *_ = args_schema_properties.items()
                                    if param_type := param_info.get("type", None):
                                        if type(args[0]).__name__ == self._mcp_type_2_py_type(param_type):
                                            final_kwargs[param_name] = args[0]

                            if not len(final_kwargs):
                                try:
                                    final_kwargs = json.loads(args[0])
                                except TypeError as e:
                                    raise CommandError(
                                        code=CommandErrorCode.VALUE_ERROR.value,
                                        message=f'MCP tool: invalid "text__" type, {str(e)}',
                                    )
                                except json.JSONDecodeError as e:
                                    raise CommandError(
                                        code=CommandErrorCode.VALUE_ERROR.value,
                                        message=(
                                            f"MCP tool: invalid `text__` parameter format, INVALID JSON schema, {e}"
                                        ),
                                    )
                        else:
                            if "text__" in kwargs:
                                final_kwargs = json.loads(kwargs["text__"])
                            elif required_schema_param_count == 1:
                                param_name = required_args_list[0]
                                if param_name not in kwargs:
                                    raise CommandError(
                                        code=CommandErrorCode.VALUE_ERROR.value,
                                        message=f'MCP tool: unknown parameter "{param_name}" parameter format.',
                                    )
                                final_kwargs.update(kwargs)
                            else:
                                raise CommandError(
                                    code=CommandErrorCode.VALUE_ERROR.value,
                                    message=f'MCP tool: missing "text__" parameters, kwargs={kwargs}',
                                )
                    else:
                        for arg_name, arg in zip(args_schema_properties.keys(), args):
                            final_kwargs[arg_name] = arg
                        final_kwargs.update(kwargs)

                mcp_result = await self._mcp_client.call_tool(
                    name=meta.name,
                    arguments=final_kwargs,
                )
                # convert to moss Message
                return mcp_call_tool_result_to_message(mcp_result, name=self.name())
            except mcp.McpError as e:
                raise CommandError(code=CommandErrorCode.FAILED.value, message=f"MCP call failed: {str(e)}") from e
            except Exception as e:
                raise CommandError(
                    code=CommandErrorCode.FAILED.value, message=f"MCP tool execution failed: {str(e)}"
                ) from e

        return _server_caller_as_command

    async def execute(self, task: CommandTask[R]) -> R:
        if not self.is_running():
            raise RuntimeError("MCPChannel is not running")
        func = self._get_command_func(task.meta)
        if func is None:
            raise LookupError(f"Channel {self._name} can find command {task.meta.name}")
        return await func(*task.args, **task.kwargs)

        # --- 工具转Command的核心逻辑 --- #

    def _convert_tools_to_command_metas(self, tools: list[types.Tool]) -> list[CommandMeta]:
        """将MCP工具转换为Mosshell的CommandMeta"""
        metas = []
        for tool in tools:
            tool_name = tool.name

            # 生成符合Code as Prompt的interface（模型可见的函数签名）
            interface, description = self._generate_code_as_prompt(tool)

            metas.append(
                CommandMeta(
                    name=tool_name,
                    description=description or "",
                    chan=self._name,
                    interface=interface,
                    available=True,
                    args_schema=tool.inputSchema,
                    delta_arg=CommandDeltaType.TEXT,
                    # mcp channel 默认不是阻塞的?
                    blocking=self._blocking,
                )
            )
        return metas

    @staticmethod
    def _mcp_type_2_py_type(param_info_type: str) -> str:
        param_type = MCPChannelRuntime.MCP_PY_TYPES_TRANS_TABLE.get(param_info_type.lower(), "Any")
        return param_type

    def _parse_schema(self, schema: dict) -> tuple[list, list]:
        required_params = []
        optional_params = []
        required_param_docs = []
        optional_param_docs = []

        for param_name, param_info in schema.get("properties", {}).items():
            # 确定参数类型
            param_type = self._mcp_type_2_py_type(param_info.get("type", ""))

            # 确定默认值
            param_str = f"{param_name}: {param_type}"
            if param_name not in schema.get("required", []):
                default_value = "None" if param_type != "bool" else "False"
                param_str += f"={default_value}"

            # 根据是否必需参数，添加到不同的列表
            if param_name in schema.get("required", []):
                required_params.append(param_str)
                # 添加参数文档
                if "description" in param_info:
                    required_param_docs.append(f"    :param {param_name}: {param_info['description']}")
            else:
                optional_params.append(param_str)
                # 添加参数文档
                if "description" in param_info:
                    optional_param_docs.append(f"    :param {param_name}: {param_info['description']}")

        return required_params + optional_params, required_param_docs + optional_param_docs

    def _parse_schema_container(self, schema: dict) -> tuple[list, list]:
        params = [self.COMMAND_DELTA_PARAMETER]
        try:
            required_param_docs = [
                "param text__: 用 JSON 描述参数，它的 JSON Schema 如右:",
                json.dumps(schema),
            ]
        except Exception as e:
            raise e

        return params, required_param_docs

    def _parse_input_schema(self, input_schema: dict[str, Any], error_prefix="") -> tuple[list[str], list[str]]:
        """解析inputSchema并提取参数信息和参数文档"""
        # todo: 考虑直接将 json schema 作为 text__ 参数.
        if not input_schema:
            return [], []

        params = []
        param_docs = []
        try:
            # 解析inputSchema
            schema = input_schema
            if isinstance(schema, str):
                schema = json.loads(schema)

            if "properties" not in schema:
                return params, param_docs

            # 合并列表，必需参数在前，可选参数在后
            # mcp_types = [i.get('type', '') for i in list(schema['properties'].values())]
            # if any([x in self.MCP_CONTAINER_TYPES for x in mcp_types]):
            #     params, param_docs = self._parse_schema_container(schema)
            # else:
            #     params, param_docs = self._parse_schema(schema)
            params, param_docs = self._parse_schema_container(schema)

        except Exception as e:
            print(f"{error_prefix}解析inputSchema出错: {e}")
        return params, param_docs

    def _adjust_description(self, description: str, param_doc: str) -> str:
        return f"{description}\n{param_doc}\n"

    def _generate_code_as_prompt(self, tool: types.Tool) -> tuple[str, str]:
        """生成模型可见的Command接口（Code as Prompt）"""

        # 提取函数名（将连字符替换为下划线）
        function_name = tool.name.replace("-", "_")

        # 提取参数信息
        params, param_docs = self._parse_input_schema(tool.inputSchema, "")

        description = tool.description or ""
        if len(params) == 1 and params[0] == self.COMMAND_DELTA_PARAMETER:
            description = self._adjust_description(description, "".join(param_docs))

        # 生成Async函数签名（符合Python语法）
        interface = (
            f"async def {function_name}({', '.join(params)}) -> Any:\n"
            f"    '''\n"
            f"    {description}\n"
            # f"    {''.join(param_docs)}\n"
            f"    '''\n"
            f"    pass"
        )
        return interface, description

    def _build_channel_meta(
        self, initialize_result: types.InitializeResult, tool_result: types.ListToolsResult
    ) -> ChannelMeta:
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
    async def idle(self) -> None:
        pass

    async def policy_pause(self) -> None:
        pass

    async def clear_all(self) -> None:
        pass

    def is_available(self) -> bool:
        return True


class MCPChannel(Channel):
    """对接MCP服务的Channel"""

    def __init__(self, *, name: str, description: str, mcp_client: mcp.ClientSession, blocking: bool = False):
        self._name = name
        self._desc = description
        self._mcp_client = mcp_client
        self._runtime: Optional[MCPChannelRuntime] = None
        self._blocking = blocking

    # --- Channel 核心方法实现 --- #
    def name(self) -> str:
        return self._name

    @property
    def runtime(self) -> ChannelRuntime:
        if not self._runtime or not self._runtime.is_running():
            raise RuntimeError("MCPChannel not bootstrapped")
        return self._runtime

    @property
    def build(self) -> Builder:
        raise NotImplementedError("MCPChannel does not implement `build`")

    def bootstrap(self, container: Optional[IoCContainer] = None) -> ChannelRuntime:
        if self._runtime is not None and self._runtime.is_running():
            raise RuntimeError(f"Channel {self} has already been started.")

        self._runtime = MCPChannelRuntime(
            name=self._name,
            container=container,
            mcp_client=self._mcp_client,
            blocking=self._blocking,
        )

        return self._runtime

    # --- 未使用的Channel方法（默认空实现） --- #

    def children(self) -> dict[str, Channel]:
        return {}

    def is_running(self) -> bool:
        return self._runtime is not None and self._runtime.is_running()
