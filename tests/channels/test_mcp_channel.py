import pytest
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from contextlib import AsyncExitStack
from ghoshell_moss.channels.mcp_channel import MCPChannel
from os.path import dirname, join
import mcp.types as types


@pytest.mark.asyncio
async def test_mcp_channel_baseline():
    exit_stack = AsyncExitStack()
    async with exit_stack:
        read_stream, write_stream = await exit_stack.enter_async_context(
            stdio_client(StdioServerParameters(
                command="python",
                args=[join(dirname(__file__), "helper/mcp_server_demo.py")],
                env=None
            ))
        )
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        tool_res = await session.list_tools()
        assert tool_res is not None

        mcp_channel = MCPChannel(
            name="mcp",
            description="MCP channel",
            mcp_client=session,
        )

        async with mcp_channel.bootstrap() as client:
            commands = list(client.commands().values())
            assert len(commands) > 0

            available_test_cmd = client.get_command("add")
            assert available_test_cmd is not None
            res: types.CallToolResult = await available_test_cmd(x=1, y=2)
            assert res.isError is False
            assert res.structuredContent['result'] == 3
