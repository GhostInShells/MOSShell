import asyncio
import contextlib
import json
import logging

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession, McpError
from mcp import types as mcp_types

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.message import Message, Text

logger = logging.getLogger("MapsBaidu")

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    import os
    return os.environ.get("BAIDU_MAPS_API_KEY", "")

# ---------------------------------------------------------------------------
# MCP session
# ---------------------------------------------------------------------------

class _Session:
    """Manage a single Baidu Maps MCP server connection."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: ClientSession | None = None
        self._tools: list[mcp_types.Tool] = []
        self._exit_stack: contextlib.AsyncExitStack | None = None

    @property
    def tools(self) -> list[mcp_types.Tool]:
        return self._tools

    async def connect(self):
        params = StdioServerParameters(
            command="mcp-server-baidu-maps",
            args=[],
            env={"BAIDU_MAPS_API_KEY": self._api_key},
        )
        self._exit_stack = contextlib.AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        session = ClientSession(read, write)
        await self._exit_stack.enter_async_context(session)
        await session.initialize()
        result = await session.list_tools()
        self._client = session
        self._tools = list(result.tools)

    async def call_tool(self, name: str, arguments: dict, timeout: float) -> Observe:
        if self._client is None:
            return Observe.new("[BaiduMaps] not connected")
        try:
            result = await asyncio.wait_for(
                self._client.call_tool(name=name, arguments=arguments),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return Observe.new(f"[BaiduMaps/{name}] timeout after {timeout}s")
        except McpError as e:
            return Observe.new(f"[BaiduMaps/{name}] MCP error: {e}")

        if result.isError:
            texts = [
                c.text for c in result.content
                if isinstance(c, mcp_types.TextContent)
            ]
            return Observe.new(f"[BaiduMaps/{name}] error: {' '.join(texts)}")

        messages = []
        for c in result.content:
            if isinstance(c, mcp_types.TextContent):
                messages.append(
                    Message.new(name=f"baidu/{name}").with_content(Text(text=c.text))
                )
        return Observe(messages=messages)

    async def disconnect(self):
        if self._exit_stack:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = None

# ---------------------------------------------------------------------------
# Schema rendering
# ---------------------------------------------------------------------------

def _render_params(schema: dict) -> str:
    if not schema or schema.get("type") != "object":
        return ""
    properties = schema.get("properties", {})
    if not properties:
        return ""
    required = set(schema.get("required", []))
    parts = []
    for pname, pschema in properties.items():
        ptype = pschema.get("type", "any")
        pdesc = (pschema.get("description", "") or "").split("\n")[0][:80]
        req = ", required" if pname in required else ""
        if pdesc:
            parts.append(f"`{pname}` ({ptype}{req}): {pdesc}")
        else:
            parts.append(f"`{pname}` ({ptype}{req})")
    return "; ".join(parts)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(matrix: Matrix):
    api_key = _get_api_key()
    session = _Session(api_key)
    await session.connect()

    channel = new_channel(
        name="mcp_baidu_map",
        description="Baidu Maps — location search, geocoding, directions, weather, "
                    "traffic, and POI extraction via Baidu Maps API",
    )

    @channel.build.context_messages
    async def context() -> list:
        lines = ["### Baidu Maps Tools"]
        for tool in session.tools:
            desc = (tool.description or "").split("\n")[0][:120]
            lines.append(f"- `{tool.name}`: {desc}")
            params = _render_params(tool.inputSchema)
            if params:
                lines.append(f"  params: {params}")
        return ["\n".join(lines)]

    @channel.build.command(always_observe=True)
    async def call(tool: str, timeout: float = 30.0, text__: str = "") -> Observe:
        """Call a Baidu Maps MCP tool. Pass JSON arguments inside the open/close tag.

        :param tool: tool name (e.g. map_search_places, map_geocode, map_weather)
        :param timeout: timeout in seconds
        :param text__: JSON arguments, placed inside the CTML tag body
        """
        try:
            arguments = json.loads(text__) if text__ else {}
        except json.JSONDecodeError as e:
            return Observe.new(f"[BaiduMaps/{tool}] invalid JSON arguments: {e}")

        return await session.call_tool(tool, arguments, timeout=timeout)

    @channel.build.command(always_observe=True)
    async def list_tools() -> str:
        """List all available Baidu Maps tools with their parameter schemas."""
        if not session.tools:
            return "No tools discovered. The MCP server may not be connected."
        lines = ["### Baidu Maps Tools"]
        for tool in session.tools:
            desc = (tool.description or "").split("\n")[0][:150]
            lines.append(f"\n**{tool.name}**: {desc}")
            params = _render_params(tool.inputSchema)
            if params:
                lines.append(f"  {params}")
        return "\n".join(lines)

    await matrix.provide_channel(channel)
    logger.info("Baidu Maps app started — %d tools", len(session.tools))


if __name__ == "__main__":
    Matrix.discover().run(main)
