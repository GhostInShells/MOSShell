from typing import Literal, Iterable
import asyncio
from mcp.server.fastmcp import FastMCP
from mcp.types import ContentBlock, TextContent, ImageContent

from ghoshell_moss.message import Message, Text, Base64Image
from ghoshell_moss.host import Host
from ghoshell_moss.host.abcd import MossHost, MossAsToolSet
import click


class FastMCPMessageAdapter:

    @classmethod
    def parse_message_to_blocks(cls, messages: Iterable[Message]) -> Iterable[ContentBlock]:
        for msg in messages:
            for content in msg.as_contents(with_meta=True):
                if text := Text.from_content(content):
                    yield TextContent(
                        type='text',
                        text=text.text,
                    )
                elif base64_image := Base64Image.from_content(content):
                    yield ImageContent(
                        type='image',
                        data=base64_image.source['data'],
                        mimeType=base64_image.source['media_type'],
                    )


# 2. 定义状态容器，用于在 MCP 运行时保存 moss 实例
class ServerState:
    def __init__(self):
        self.host: MossHost | None = None
        self.toolset: MossAsToolSet | None = None


def bootstrap(state: ServerState, mcp: FastMCP):
    @mcp.tool()
    async def moss_instruction() -> str:
        """
        返回 MOSS 架构的系统指令, 需要先调用这个指令了解如何使用 moss.
        """
        if not state.toolset:
            return "Error: MOSS not initialized."
        return state.toolset.moss_instruction(True)

    @mcp.tool()
    async def get_moss_dynamic_info() -> list[ContentBlock]:
        """获取 MOSS 当前的运行状态、动态信息。"""
        if not state.toolset:
            return [TextContent(type='text', text="System not ready.")]
        msgs = await state.toolset.moss_dynamic_messages(refresh=True, max_wait=5.0)
        # 直接返回你的 adapter 生成器
        return list(FastMCPMessageAdapter.parse_message_to_blocks(msgs))

    @mcp.tool()
    async def execute_ctml(logos: str, with_dynamic: bool = False) -> list[ContentBlock]:
        """向 MOSS 执行 CTML 指令。支持多行指令，用于控制系统状态和逻辑流。"""
        if not state.toolset:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]

        # 执行命令并等待观察结果
        executed = await state.toolset.moss_exec(logos, wait_done=True)
        results = list(FastMCPMessageAdapter.parse_message_to_blocks(executed))
        # 将 list[Message] 序列化为可读字符串
        if with_dynamic:
            dynamic_info = await get_moss_dynamic_info()
            results.extend(dynamic_info)

        return results

    @mcp.tool()
    async def interrupt_execution() -> str:
        """强制中断当前所有运行中的逻辑。"""
        await state.toolset.moss_interrupt()
        return "MOSS runtime interrupted."


def main_entry(
        mode: str | None = None,
        session_scope: str | None = None,
        transport: Literal['sse', 'std', 'streamable_http'] = 'sse',
        server_name: str = 'MOSS-Toolset-Server',
        host: str = '127.0.0.1',
        port: int = 20773,
) -> None:
    """启动 MOSS MCP 服务端"""
    mcp = FastMCP(
        server_name,
        host=host,
        port=port,
    )

    moss_host = Host(mode=mode, session_scope=session_scope)
    state = ServerState()
    # 注册对应的工具.
    bootstrap(state, mcp)
    params = dict(
        mode=mode, session_scope=session_scope, transport=transport,
        server_name=server_name, host=host, port=port,
    )

    async def run_server():
        # 启动 MOSS 运行时环境
        async with moss_host.run_as_toolset() as toolset:
            state.host = moss_host
            state.toolset = toolset
            moss_host.matrix().logger.info(
                'Moss MCP toolset started with params: %r',
                params,
            )
            # 启动 MCP Server (FastMCP 内部会处理进程阻塞)
            if transport == 'sse':
                await mcp.run_sse_async()
            elif transport == 'std':
                await mcp.run_stdio_async()
            elif transport == 'streamable_http':
                await mcp.run_streamable_http_async()
            else:
                raise click.BadParameter(f"transport {transport} not supported")

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass


@click.command()
@click.option('--mode', default='default', help='MOSS 运行时模式')
@click.option('--session-scope', default='default', help='Session 作用域')
@click.option('--transport', type=click.Choice(['sse', 'std', 'streamable_http']), default='sse', help='通信协议')
@click.option('--host', default='127.0.0.1', help='SSE 服务地址 (仅在 transport=sse 时生效)')
@click.option('--port', default=20773, help='SSE 服务端口 (仅在 transport=sse 时生效)')
@click.option('--server-name', default='MOSS-Toolset-Server', help='MCP 服务名称')
def main(mode, session_scope, transport, host, port, server_name):
    """MOSS MCP 服务启动程序"""

    # 传递给你的 main_entry
    main_entry(
        mode=mode,
        session_scope=session_scope,
        transport=transport,
        server_name=server_name,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
