"""Minimal MCP stdio server for MCP Hub integration testing."""
import json
import sys
import asyncio
from mcp.server.stdio import stdio_server
from mcp.server import Server
from mcp import types as mcp_types

server = Server("mcp-hub-test-demo")


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="add",
            description="add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "first number"},
                    "y": {"type": "number", "description": "second number"},
                },
                "required": ["x", "y"],
            },
        ),
        mcp_types.Tool(
            name="foo",
            description="foo helper",
            inputSchema={"type": "object", "properties": {}},
        ),
        mcp_types.Tool(
            name="bar",
            description="bar helper",
            inputSchema={"type": "object", "properties": {}},
        ),
        mcp_types.Tool(
            name="multi",
            description="multi helper",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    if name == "add":
        x = float(arguments.get("x", 0))
        y = float(arguments.get("y", 0))
        return [mcp_types.TextContent(type="text", text=f"{x} + {y} = {x + y}")]
    elif name == "foo":
        return [mcp_types.TextContent(type="text", text="foo ok")]
    elif name == "bar":
        return [mcp_types.TextContent(type="text", text="bar ok")]
    elif name == "multi":
        return [mcp_types.TextContent(type="text", text="multi ok")]
    return [mcp_types.TextContent(type="text", text=f"unknown: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
