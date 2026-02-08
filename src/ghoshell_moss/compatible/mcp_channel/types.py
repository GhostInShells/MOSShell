from mcp import types as mcp_types

from ghoshell_moss.message import Addition


class MCPCallToolResultAddition(Addition, mcp_types.CallToolResult):
    @classmethod
    def keyword(cls) -> str:
        return "mcp_call_tool_result"

    @classmethod
    def from_mcp_types(cls, mcp_result: mcp_types.CallToolResult):
        return cls(
            isError=mcp_result.isError,
            structuredContent=mcp_result.structuredContent,
            content=mcp_result.content,
        )
