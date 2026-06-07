import json
import logging
import os

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.channels.mcp_hub import (
    MCPServerConfig,
    MCPServerSession,
    render_input_schema,
)

logger = logging.getLogger("MCPBaiduMap")

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if k not in os.environ:
                        os.environ[k] = v


def _get_api_key() -> str:
    return os.environ.get("BAIDU_MAPS_API_KEY", "")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(matrix: Matrix):
    _load_dotenv()
    api_key = _get_api_key()

    config = MCPServerConfig(
        name="baidu_map",
        transport="stdio",
        command="mcp-server-baidu-maps",
        env={"BAIDU_MAPS_API_KEY": api_key},
    )
    session = MCPServerSession(config=config)
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
            params = render_input_schema(tool.inputSchema)
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
            params = render_input_schema(tool.inputSchema)
            if params:
                lines.append(f"  {params}")
        return "\n".join(lines)

    await matrix.provide_channel(channel)
    logger.info("Baidu Maps app started — %d tools", len(session.tools))

    # Keep the process alive to serve channel commands through the Zenoh proxy.
    await matrix.wait_closed()


if __name__ == "__main__":
    Matrix.discover().run(main)
