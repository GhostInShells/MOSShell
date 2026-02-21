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
from ghoshell_moss.message import Message


def get_mcp_call_tool_result(message: Message) -> MCPCallToolResultAddition:
    """
    测试用例里应该只有一个 MCPStructuredContent
    """

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

            async with mcp_channel.bootstrap() as client:
                commands = list(client.own_commands().values())
                assert len(commands) > 0

                # print('')
                # for i, cmd in enumerate(commands):
                #     print(f"{i}: {cmd.name()} {cmd.meta().model_dump_json()}")

                available_test_cmd = client.get_command("add")
                assert available_test_cmd is not None

                # args
                res: Message = await available_test_cmd(1, 2)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # kwargs
                res: Message = await available_test_cmd(x=1, y=2)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # args + kwargs
                res: Message = await available_test_cmd(1, y=2)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # args, default
                # 无法区分第一个参数是原始函数还是text__
                res: Message = await available_test_cmd(1)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # kwargs, default
                res: Message = await available_test_cmd(x=1)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # text__
                text__: str = json.dumps({"x": 1, "y": 2})
                res: Message = await available_test_cmd(text__=text__)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # args: text__
                res: Message = await available_test_cmd(text__)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # text__, default
                text__: str = json.dumps({"x": 1})
                res: Message = await available_test_cmd(text__=text__)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # foo
                available_test_cmd = client.get_command("foo")
                assert available_test_cmd is not None

                # text__, default
                text__: str = json.dumps({"a": 1, "b": {"i": 2}})
                res: Message = await available_test_cmd(text__=text__)
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                available_test_cmd = client.get_command("bar")
                assert available_test_cmd is not None

                # kwargs
                res: Message = await available_test_cmd(s="aaa")
                mcp_call_tool_result = get_mcp_call_tool_result(res)
                assert mcp_call_tool_result.isError is False
                assert mcp_call_tool_result.structuredContent["result"] == 3

                # args,
                with pytest.raises(CommandError):
                    await available_test_cmd("aaa")

                available_test_cmd = client.get_command("multi")
                assert available_test_cmd is not None

                with pytest.raises(CommandError):
                    await available_test_cmd(1, 2, a=2, c=3)
