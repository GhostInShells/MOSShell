# Baidu Maps App

Baidu Maps MCP wrapper — self-contained MOSS app that exposes Baidu Maps API tools as a channel.

## Architecture

The app uses MCP Hub's public API (`MCPServerSession`, `mcp_result_to_observe`, `render_input_schema`) for MCP connection management — no duplicated logic.

On startup, it loads `BAIDU_MAPS_API_KEY` from the local `.env` file, connects to the MCP server via stdio transport, discovers all tools, and dynamically generates context messages with tool schemas. Tools are called via a generic `call` command.

## Channel

`mcp_baidu_map` — CTML path: `apps.mcp_baidu_map:<command>`

### Commands

| Command | Signature | Behavior |
|---------|-----------|----------|
| `call` | `(tool: str, timeout: float = 30.0, text__: str = "") -> Observe` | Call any Baidu Maps MCP tool with JSON args |
| `list_tools` | `() -> str` | List all available tools with parameter schemas |

### CTML Examples

```
<apps.mcp_baidu_map:call tool="map_geocode">{"address": "北京市海淀区"}</apps.mcp_baidu_map:call>

<apps.mcp_baidu_map:call tool="map_search_places">{"query": "咖啡", "region": "北京"}</apps.mcp_baidu_map:call>

<apps.mcp_baidu_map:call tool="map_weather">{"district_id": "110108"}</apps.mcp_baidu_map:call>
```

## Configuration

Requires `BAIDU_MAPS_API_KEY` set in the app's local `.env` file (`mcp/baidu_map/.env`).

## Available Tools

The Baidu Maps MCP server exposes 10 tools:

- `map_geocode` — Address → coordinates
- `map_reverse_geocode` — Coordinates → address
- `map_search_places` — Place search by keyword and region
- `map_place_details` — POI detail lookup by UID
- `map_directions` — Route planning (driving, riding, walking, transit)
- `map_directions_matrix` — Batch distance/time matrix
- `map_weather` — Real-time weather and 5-day forecast
- `map_ip_location` — IP geolocation
- `map_road_traffic` — Real-time traffic conditions
- `map_poi_extract` — POI extraction from text (requires advanced API permission)
