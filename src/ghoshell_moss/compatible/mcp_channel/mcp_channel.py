import json
from collections.abc import Callable, Coroutine
from typing import Any, Optional, TypeVar

from jsonschema import Draft202012Validator, Draft201909Validator, Draft7Validator, Draft6Validator

from ghoshell_moss import CommandError, CommandErrorCode
from ghoshell_moss.compatible.mcp_channel.utils import mcp_call_tool_result_to_message

try:
    import mcp
    from mcp import types
except ImportError:
    raise ImportError("Could not import mcp. Please install ghoshell-moss[mcp].")

from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer

from ghoshell_moss.core.concepts.channel import Channel, ChannelMeta, ChannelRuntime
from ghoshell_moss.core.concepts.command import (
    Command,
    CommandDeltaType,
    CommandMeta,
    CommandTask,
    CommandWrapper, CommandUniqueName,
)
from ghoshell_moss.core.runtime import AbsChannelRuntime

R = TypeVar("R")  # 泛型结果类型


class MCPChannel(Channel):
    """对接MCP服务的Channel"""

    def __init__(self, *, name: str, description: str, mcp_client: mcp.ClientSession, blocking: bool = False):
        self._name = name
        self._desc = description
        self._id = uuid()
        self._mcp_client = mcp_client
        self._runtime: Optional[MCPChannelRuntime] = None
        self._blocking = blocking

    # --- Channel 核心方法实现 --- #
    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._desc

    @property
    def runtime(self) -> ChannelRuntime:
        if not self._runtime or not self._runtime.is_running():
            raise RuntimeError("MCPChannel not bootstrapped")
        return self._runtime

    def bootstrap(self, container: Optional[IoCContainer] = None) -> ChannelRuntime:
        if self._runtime is not None and self._runtime.is_running():
            raise RuntimeError(f"Channel {self} has already been started.")

        self._runtime = MCPChannelRuntime(
            channel=self,
            container=container,
            mcp_client=self._mcp_client,
            blocking=self._blocking,
        )

        return self._runtime


class MCPChannelRuntime(AbsChannelRuntime[MCPChannel]):
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

    DIALECT_DRAFT_TABLE: dict[str, Any] = {
        # "": Draft202012Validator,
        "draft-07": Draft7Validator,
        "draft-06": Draft6Validator,
        "draft/2019-09": Draft201909Validator,
    }

    COMMAND_DELTA_PARAMETER: str = f"{CommandDeltaType.TEXT.value}:str"

    def __init__(
            self,
            *,
            channel: "MCPChannel",
            mcp_client: mcp.ClientSession,
            container: Optional[IoCContainer] = None,
            blocking: bool = False,
    ):
        super().__init__(channel=channel, container=container)
        self._mcp_client: Optional[mcp.ClientSession] = mcp_client  # MCP客户端实例
        self._meta: Optional[ChannelMeta] = None  # Channel元信息
        self._blocking = blocking

    def sub_channels(self) -> dict[str, "Channel"]:
        return {}

    async def on_startup(self) -> None:
        if self._mcp_client is None:
            raise RuntimeError("MCP client is not set")

        # 同步远端工具元信息（session 初始化由调用方管理，这里只拉取 tools）
        tools = await self._mcp_client.list_tools()
        self._meta = self._build_channel_meta(tool_result=tools)

    async def on_close(self) -> None:
        # mcp session 生命周期由外部管理；这里不主动 close
        return

    async def on_running(self) -> None:
        # 保持运行直到 close() 触发。
        await self._closing_event.wait()

    async def _main_loop(self) -> None:
        # 该 runtime 不依赖内部任务队列；仅等待退出。
        await self._closing_event.wait()

    async def _consume_task_with_paths(self, paths: list[str], task: CommandTask) -> None:
        # 兼容 ChannelRuntime 的任务调度：直接执行并 resolve/fail。
        if len(paths) > 0:
            task.fail(CommandErrorCode.NOT_FOUND.error(f"MCPChannel has no sub channel: {'.'.join(paths)}"))
            return
        if task.func is None:
            task.fail(CommandErrorCode.NOT_FOUND.error(f"command {task.meta.name} not found"))
            return
        self.create_asyncio_task(self.execute_task(task))

    async def wait_connected(self) -> None:
        return

    def is_connected(self) -> bool:
        # 注意：AbsChannelRuntime.start() 会在 `_started` 置位之前调用 is_connected()
        # 来决定是否需要 refresh metas；这里不能依赖 is_running()。
        return self._mcp_client is not None and not self._closing_event.is_set()

    def _is_available(self) -> bool:
        return True

    def is_idle(self) -> bool:
        return True

    async def wait_idle(self) -> None:
        return

    async def _clear_own(self) -> None:
        return

    async def _generate_own_metas(self) -> dict[str, ChannelMeta]:
        if self._meta is None:
            if self._mcp_client is None:
                return {"": ChannelMeta.new_empty(self.id, self.channel)}
            tools = await self._mcp_client.list_tools()
            self._meta = self._build_channel_meta(tool_result=tools)
        return {"": self._meta.model_copy()}

    def get_own_command(self, name: CommandUniqueName) -> Optional[Command]:
        path, name = Command.split_unique_name(name)
        if path:
            return None
        meta = self._meta
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                return command
        return None

    def has_own_command(self, name: CommandUniqueName) -> bool:
        path, name = Command.split_unique_name(name)
        if path:
            return False
        for command_meta in self._meta.commands:
            if command_meta.name == name:
                return True
        return False

    def own_commands(self, available_only: bool = True) -> dict[str, Command]:
        meta = self._meta
        if meta is None:
            raise RuntimeError(f"Channel client {self.name} is not running")
        result = {}
        for command_meta in meta.commands:
            if not available_only or command_meta.available:
                func = self._get_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def _get_validator(self, args_schema: dict):
        dialect = args_schema.get("$schema", "")
        if type(dialect) is not str:
            dialect = ""
        dialect = dialect.lower()
        Validator = Draft202012Validator
        for dialect_key, _Validator in self.DIALECT_DRAFT_TABLE.items():
            if dialect_key in dialect:
                Validator = _Validator
        return Validator(args_schema)

    def _get_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        args_schema_properties = meta.json_schema.get("properties", {})
        required_args_list = meta.json_schema.get("required", [])
        # schema_param_count = len(args_schema_properties)
        required_schema_param_count = len(required_args_list)

        def _assemble_params(*args, **kwargs):
            final_kwargs = {}
            param_count = len(args) + len(kwargs)

            if param_count != 1:
                for arg_name, arg in zip(args_schema_properties.keys(), args):
                    final_kwargs[arg_name] = arg
                final_kwargs.update(kwargs)
                return final_kwargs

            # param_count == 1:
            if len(args) == 1:
                if required_schema_param_count == 1:
                    if type(args[0]) is not str:
                        param_name = required_args_list[0]
                        final_kwargs[param_name] = args[0]
                        return final_kwargs

                text__ = args[0]

            else:  # len(kwargs) == 1:
                # Prioritize parsing "text__"
                if "text__" in kwargs:
                    text__ = kwargs["text__"]

                elif required_schema_param_count == 1:
                    return kwargs

                # if "text__" not in kwargs:
                else:
                    raise CommandError(
                        code=CommandErrorCode.VALUE_ERROR.value,
                        message=f'MCP tool: missing "text__" parameters, kwargs={kwargs}',
                    )

            try:
                final_kwargs = json.loads(text__)
            except TypeError as e:
                raise CommandError(
                    code=CommandErrorCode.VALUE_ERROR.value,
                    message=f'MCP tool: invalid "text__" type, {str(e)}',
                )
            except json.JSONDecodeError as e:
                raise CommandError(
                    code=CommandErrorCode.VALUE_ERROR.value,
                    message=f"MCP tool: invalid `text__` parameter format, INVALID JSON schema, {e}",
                )
            return final_kwargs

        # 回调服务端.
        async def _server_caller_as_command(*args, **kwargs):
            # 调用MCP客户端执行工具
            try:
                final_kwargs = _assemble_params(*args, **kwargs)

                # 使用 jsonschema 验证参数是否符合 schema
                if meta.json_schema:
                    # http://modelcontextprotocol.io/specification/draft/basic
                    # Schema Dialect
                    validator = self._get_validator(meta.json_schema)
                    if errs := validator.iter_errors(final_kwargs):
                        msgs = []
                        for e in errs:
                            msg = e.message
                            if e.json_path and e.json_path[0] != "$":
                                msg += f" at {e.json_path}"
                            msgs.append(msg)
                        if msgs:
                            message = f"MCP tool '{meta.name}': {';'.join(msgs)}"
                            raise CommandError(
                                code=CommandErrorCode.VALUE_ERROR.value,
                                message=message,
                            )

                mcp_result = await self._mcp_client.call_tool(
                    name=meta.name,
                    arguments=final_kwargs,
                )
                # convert to moss Message
                return mcp_call_tool_result_to_message(mcp_result, name=self.name)
            except CommandError as e:
                raise e
            except mcp.McpError as e:
                raise CommandError(code=CommandErrorCode.FAILED.value, message=f"MCP call failed: {str(e)}") from e
            except Exception as e:
                raise CommandError(
                    code=CommandErrorCode.FAILED.value, message=f"MCP tool execution failed: {str(e)}"
                ) from e

        return _server_caller_as_command

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
                    json_schema=tool.inputSchema,
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

    def _build_channel_meta(self, *, tool_result: types.ListToolsResult) -> ChannelMeta:
        """构建Channel元信息（包含所有工具的CommandMeta）"""
        return ChannelMeta(
            name=self.name,
            channel_id=self.channel.id(),
            available=True,
            description=self.channel.description(),
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
