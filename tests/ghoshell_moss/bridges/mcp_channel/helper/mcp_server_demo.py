"""Minimal MCP stdio server for MCP Hub integration testing."""
import asyncio
from mcp.server.mcpserver import MCPServer

server = MCPServer("mcp-hub-test-demo")


@server.tool()
async def add(x: int, y: int) -> str:
    """add two numbers"""
    return f"{x} + {y} = {x + y}"


@server.tool()
async def foo() -> str:
    """foo helper"""
    return "foo ok"


@server.tool()
async def bar() -> str:
    """bar helper"""
    return "bar ok"


@server.tool()
async def multi() -> str:
    """multi helper"""
    return "multi ok"


async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
