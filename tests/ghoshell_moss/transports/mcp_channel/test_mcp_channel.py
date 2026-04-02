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
from ghoshell_moss.core.concepts.command import CommandErrorCode
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

                message: Message = await available_test_cmd(1, 2)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                message: Message = await available_test_cmd(x=1, y=2)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                message: Message = await available_test_cmd(1, y=2)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                message: Message = await available_test_cmd(1)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                message: Message = await available_test_cmd(x=1)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                text__: str = json.dumps({"x": 1, "y": 2})
                message: Message = await available_test_cmd(text__=text__)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                message: Message = await available_test_cmd(text__)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                text__: str = json.dumps({"x": 1})
                message: Message = await available_test_cmd(text__=text__)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                available_test_cmd = runtime.get_command("foo")
                assert available_test_cmd is not None

                text__: str = json.dumps({"a": 1, "b": {"i": 2}})
                message: Message = await available_test_cmd(text__=text__)
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                available_test_cmd = runtime.get_command("bar")
                assert available_test_cmd is not None

                message: Message = await available_test_cmd(s="aaa")
                assert message is not None
                mcp_call_tool_result = get_mcp_call_tool_result(message)
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
                assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                assert "MCP tool 'multi':" in exc_info.value.message
                # mcp.ClientSession call_tool
                assert "'d' is a required property" in exc_info.value.message

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

                # available_test_cmd = runtime.get_command("bar")
                # assert available_test_cmd is not None
                # with pytest.raises(CommandError) as exc_info:
                #     await available_test_cmd(s="aaa", extra_param="extra")
                # assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                # assert "invalid parameters" in exc_info.value.message.lower()
                # assert "too many parameters passed" in exc_info.value.message

                available_test_cmd = runtime.get_command("multi")
                assert available_test_cmd is not None
                with pytest.raises(CommandError) as exc_info:
                    await available_test_cmd(a=1, b=2)
                assert exc_info.value.code == CommandErrorCode.VALUE_ERROR.value
                assert "MCP tool 'multi'" in exc_info.value.message
                assert "'c' is a required property" in exc_info.value.message
                assert "'d' is a required property" in exc_info.value.message


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
                # task = runtime.create_command_task("add", args=(1, 2))
                # await runtime.push_task(task)
                message = await runtime.execute_command("add", args=(1, 2))
                assert message is not None

                mcp_call_tool_result = get_mcp_call_tool_result(message)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                bar_cmd = runtime.get_command("bar")
                assert bar_cmd is not None
                task = runtime.create_command_task("bar", kwargs={"s": "hello"})

                await runtime.push_task(task)
                await task
                task_result = task.task_result()
                assert task_result is not None
                assert task_result.result is not None

                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 5

                foo_cmd = runtime.get_command("foo")
                assert foo_cmd is not None
                task = runtime.create_command_task(
                    "foo",
                    kwargs={"text__": json.dumps({"a": 10, "b": {"i": 20}})},
                )

                await runtime.push_task(task)
                await task
                task_result = task.task_result()
                assert task_result is not None
                assert task_result.result is not None

                mcp_call_tool_result = get_mcp_call_tool_result(task_result.result)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 30


@pytest.mark.asyncio
async def test_mcp_channel_execute_exception():
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
                # Test 0: execute command
                with pytest.raises(CommandError) as e:
                    _ = await runtime.execute_command(
                        "bar",
                        args=("aaa",),
                    )

                # Test 1: bar command with invalid JSON (single arg "aaa")
                assert runtime.get_command("bar") is not None
                task = runtime.create_command_task(
                    name="bar",
                    args=("aaa",),  # invalid JSON
                )

                await runtime.push_task(task)
                await task.wait(throw=False)
                e = task.exception()
                assert isinstance(e, CommandError)
                assert e.code == CommandErrorCode.VALUE_ERROR.value
                msg = e.args[0]
                assert "invalid `text__` parameter format" in msg
                assert "INVALID JSON schema" in msg

                # Test 2: multi command with missing required arg "d"
                assert runtime.get_command("multi") is not None
                task = runtime.create_command_task(
                    name="multi",
                    args=(1, 2),
                    kwargs={"a": 2, "c": 3},  # missing "d"
                )

                await runtime.push_task(task)
                await task.wait(throw=False)
                e = task.exception()
                assert isinstance(e, CommandError)
                assert e.code == CommandErrorCode.VALUE_ERROR.value
                msg = e.args[0]
                assert "MCP tool 'multi'" in msg
                assert "'d' is a required property" in msg

                # Test 3: add command with invalid JSON string
                assert runtime.get_command("add") is not None
                task = runtime.create_command_task(
                    name="add",
                    args=("invalid_json",),
                )

                await runtime.push_task(task)
                await task.wait(throw=False)
                e = task.exception()
                assert isinstance(e, CommandError)
                assert e.code == CommandErrorCode.VALUE_ERROR.value
                msg = e.args[0]
                assert "invalid `text__` parameter format" in msg
                assert "INVALID JSON schema" in msg

                # Test 4: foo command with non-string arg (int)
                assert runtime.get_command("foo") is not None
                task = runtime.create_command_task(
                    name="foo",
                    args=(12345,),  # should be string for JSON parsing
                )

                await runtime.push_task(task)
                await task.wait(throw=False)
                e = task.exception()
                assert isinstance(e, CommandError)
                assert e.code == CommandErrorCode.VALUE_ERROR.value
                msg = e.args[0]
                assert 'invalid "text__" type' in msg
                assert "the JSON object must be str, bytes or bytearray, not int" in msg

                # Test 5: bar command with too many parameters
                task = runtime.create_command_task(
                    name="bar",
                    kwargs={"s": "aaa", "extra_param": "extra"},
                )

                await runtime.push_task(task)
                await task
                e = task.exception()
                assert e is None
                # assert isinstance(e, CommandError)
                # assert e.code == CommandErrorCode.VALUE_ERROR.value
                # msg = e.args[0]
                # assert "invalid parameters" in msg.lower()
                # assert "too many parameters passed" in msg

                # Test 6: multi command with too few parameters
                task = runtime.create_command_task(
                    name="multi",
                    kwargs={"a": 1, "b": 2},  # missing required params
                )

                await runtime.push_task(task)
                await task.wait(throw=False)
                e = task.exception()
                assert isinstance(e, CommandError)
                assert e.code == CommandErrorCode.VALUE_ERROR.value
                msg = e.args[0]
                assert "MCP tool 'multi'" in msg
                assert "'c' is a required property" in msg
                assert "'d' is a required property" in msg
