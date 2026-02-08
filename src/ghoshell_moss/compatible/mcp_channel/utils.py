from mcp import types

from ghoshell_moss import CommandError, CommandErrorCode
from ghoshell_moss.compatible.mcp_channel.types import MCPCallToolResultAddition
from ghoshell_moss.message import Base64Image, Message, Text


def mcp_call_tool_result_to_message(mcp_result: types.CallToolResult, name: str | None = None) -> Message:
    if mcp_result.isError:
        raise CommandError(
            code=CommandErrorCode.FAILED.value,
            message=f"MCP tool: call failed, {mcp_result.content}",
        )
    result = Message.new(role="assistant", name=name)
    for mcp_content in mcp_result.content:
        if isinstance(mcp_content, types.TextContent):
            result.with_content(Text(text=mcp_content.text))
        if isinstance(mcp_content, types.ImageContent):
            result.with_content(Base64Image(image_type=mcp_content.mimeType, encoded=mcp_content.data))
        if isinstance(mcp_content, types.AudioContent):
            pass
        pass

    result.with_additions(MCPCallToolResultAddition.from_mcp_types(mcp_result))
    return result
