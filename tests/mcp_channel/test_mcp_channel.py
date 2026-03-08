import json
import sys
from contextlib import AsyncExitStack
from os.path import dirname, join

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ghoshell_moss import CommandError
from ghoshell_moss.compatible.mcp_channel.mcp_channel import MCPChannel
from ghoshell_moss.compatible.mcp_channel.types import MCPCallToolResultAddition
from ghoshell_moss.core.concepts.command import CommandTaskResult, CommandErrorCode, BaseCommandTask
from ghoshell_moss.message import Message


def get_mcp_call_tool_result(message: Message) -> MCPCallToolResultAddition:
    return MCPCallToolResultAddition.read(message)


@pytest.mark.asyncio
async def test_mcp_channel_baseline():
    exit_stack = AsyncExitStack()
    async with exit_stack:
        read_stream, write_stream = await exit_stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=sys.executable, args=[join(dirname(__file__), "helper/mcp_server_demo.py")], env=None
                )
            )
        )
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            tool_res = await session.list_tools()
            assert tool_res is not None

            mcp_channel = MCPChannel(
                name="mcp",
                description="MCP channel",
                mcp_client=session,
            )

            async with mcp_channel.bootstrap() as runtime:
                commands = list(runtime.own_commands().values())
                assert len(commands) == 4

                available_test_cmd = runtime.get_command("add")
                assert available_test_cmd is not None

                task_result: CommandTaskResult = await available_test_cmd(1, 2)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert task_result.caller is None
                assert mcp_call_tool_result.structuredContent["result"] == 3

                task_result: CommandTaskResult = await available_test_cmd(x=1, y=2)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                task_result: CommandTaskResult = await available_test_cmd(1, y=2)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                task_result: CommandTaskResult = await available_test_cmd(1)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                task_result: CommandTaskResult = await available_test_cmd(x=1)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                text__: str = json.dumps({"x": 1, "y": 2})
                task_result: CommandTaskResult = await available_test_cmd(text__=text__)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                task_result: CommandTaskResult = await available_test_cmd(text__)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                text__: str = json.dumps({"x": 1})
                task_result: CommandTaskResult = await available_test_cmd(text__=text__)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                available_test_cmd = runtime.get_command("foo")
                assert available_test_cmd is not None

                text__: str = json.dumps({"a": 1, "b": {"i": 2}})
                task_result: CommandTaskResult = await available_test_cmd(text__=text__)
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                available_test_cmd = runtime.get_command("bar")
                assert available_test_cmd is not None

                task_result: CommandTaskResult = await available_test_cmd(s="aaa")
                assert task_result.result is not None
                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3


@pytest.mark.asyncio
async def test_mcp_channel_exception():
    exit_stack = AsyncExitStack()
    async with exit_stack:
        read_stream, write_stream = await exit_stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=sys.executable, args=[join(dirname(__file__), "helper/mcp_server_demo.py")], env=None
                )
            )
        )
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            tool_res = await session.list_tools()
            assert tool_res is not None

            mcp_channel = MCPChannel(
                name="mcp",
                description="MCP channel",
                mcp_client=session,
            )

            async with mcp_channel.bootstrap() as runtime:
                available_test_cmd = runtime.get_command("bar")
                assert available_test_cmd is not None
                with pytest.raises(CommandError) as exc_info:
                    await available_test_cmd("aaa")
                assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                # only 1 arg, default cast to 'text__'
                assert "invalid `text__` parameter format" in exc_info.value.message
                assert "INVALID JSON schema" in exc_info.value.message

                available_test_cmd = runtime.get_command("multi")
                assert available_test_cmd is not None
                with pytest.raises(CommandError) as exc_info:
                    # missing arg "d"
                    await available_test_cmd(1, 2, a=2, c=3)
                assert exc_info.value.code == CommandErrorCode.FAILED.value
                assert "MCP tool: call failed" in exc_info.value.message
                # mcp.ClientSession call_tool
                assert (
                    "Field required [type=missing, input_value={'a': 2, 'b': 2, 'c': 3}, input_type=dict]"
                    in exc_info.value.message
                )

                available_test_cmd = runtime.get_command("add")
                assert available_test_cmd is not None
                with pytest.raises(CommandError) as exc_info:
                    await available_test_cmd("invalid_json")
                assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                assert "invalid `text__` parameter format" in exc_info.value.message
                assert "INVALID JSON schema" in exc_info.value.message

                available_test_cmd = runtime.get_command("foo")
                assert available_test_cmd is not None
                with pytest.raises(CommandError) as exc_info:
                    await available_test_cmd(12345)
                assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                assert 'invalid "text__" type' in exc_info.value.message
                # json.loads() -> TypeError
                assert "the JSON object must be str, bytes or bytearray, not int" in exc_info.value.message

                available_test_cmd = runtime.get_command("bar")
                assert available_test_cmd is not None
                with pytest.raises(CommandError) as exc_info:
                    await available_test_cmd(s="aaa", extra_param="extra")
                assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                assert "invalid parameters" in exc_info.value.message.lower()
                assert "too many parameters passed" in exc_info.value.message

                available_test_cmd = runtime.get_command("multi")
                assert available_test_cmd is not None
                with pytest.raises(CommandError) as exc_info:
                    await available_test_cmd(a=1, b=2)
                assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                assert "invalid parameters" in exc_info.value.message.lower()
                assert "too few parameters passed" in exc_info.value.message


@pytest.mark.asyncio
async def test_mcp_channel_execute():
    exit_stack = AsyncExitStack()
    async with exit_stack:
        read_stream, write_stream = await exit_stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=sys.executable, args=[join(dirname(__file__), "helper/mcp_server_demo.py")], env=None
                )
            )
        )
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            tool_res = await session.list_tools()
            assert tool_res is not None

            mcp_channel = MCPChannel(
                name="mcp",
                description="MCP channel",
                mcp_client=session,
            )

            async with mcp_channel.bootstrap() as runtime:
                add_cmd = runtime.get_command("add")
                assert add_cmd is not None

                task = BaseCommandTask.from_command(
                    add_cmd,
                    chan_="mcp",
                    args=(1, 2),
                )

                task_result: CommandTaskResult = await runtime.execute(task)
                assert task_result is not None
                assert task_result.result is not None
                assert task_result.caller == task.caller_name()
                assert len(task_result.messages) == 1

                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                bar_cmd = runtime.get_command("bar")
                assert bar_cmd is not None

                task = BaseCommandTask.from_command(
                    bar_cmd,
                    chan_="mcp",
                    kwargs={"s": "hello"},
                )

                task_result: CommandTaskResult = await runtime.execute(task)
                assert task_result is not None
                assert task_result.result is not None
                assert task_result.caller == task.caller_name()
                assert len(task_result.messages) == 1

                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 5

                foo_cmd = runtime.get_command("foo")
                assert foo_cmd is not None

                task = BaseCommandTask.from_command(
                    foo_cmd,
                    chan_="mcp",
                    kwargs={"text__": json.dumps({"a": 10, "b": {"i": 20}})},
                )

                task_result: CommandTaskResult = await runtime.execute(task)
                assert task_result is not None
                assert task_result.result is not None
                assert task_result.caller == task.caller_name()

                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 30
