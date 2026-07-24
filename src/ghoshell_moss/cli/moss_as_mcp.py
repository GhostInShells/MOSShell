from typing import Literal, Iterable, Optional
import asyncio
from mcp.server.fastmcp import FastMCP
from mcp.types import ContentBlock, TextContent, ImageContent

from ghoshell_moss.message import Message, Text, Base64Image
from ghoshell_moss.host import Host
from ghoshell_moss.host.interleaved_thinking import (
    InterleavedThinkingToolset,
    ShellEvent,
    InterpreterStatus,
    project_events,
)
from ghoshell_moss.core.blueprint.host import MossHost, MossRuntime
from ghoshell_moss.core.blueprint.environment import Environment
import click


class FastMCPMessageAdapter:

    @classmethod
    def parse_message_to_blocks(cls, messages: Iterable[Message]) -> Iterable[ContentBlock]:
        for msg in messages:
            for content in msg.as_contents(with_meta=True, join_text=True):
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
        self.toolset: MossRuntime | None = None
        # server 级 watcher — 跨 MCP 调用持有 shell 观察器, 解决 K10 跨-interpreter 历史丢失.
        # 生命周期与 MCP server 同长, 在 async with moss_host.run() 之内挂载.
        self.watcher: InterleavedThinkingToolset | None = None


def _events_to_messages(events: list[ShellEvent], status: InterpreterStatus) -> list[Message]:
    """MCP 层薄封装: 直接委托到 host.interleaved_thinking.project_events.

    K8/K9 分桶投影: 空成功折叠成 <shell_tally>, observe=True 占位, 其他走 payload.
    """
    return project_events(events, status)


def bootstrap(state: ServerState, mcp: FastMCP):
    # --- 基线组 (A/B 对照): 保留原 4 工具, 一字不改 --- #

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
    async def execute_ctml(
        logos: str,
        with_dynamic: bool = False,
        call_soon: bool = False,
        wait_done: bool = False,
    ) -> list[ContentBlock]:
        """向 MOSS 执行 CTML 指令。支持多行指令，用于控制系统状态和逻辑流。"""
        if not state.toolset:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]

        # 执行命令并等待观察结果
        executed = await state.toolset.moss_exec(logos, call_soon=call_soon, wait_done=wait_done)
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

    # --- interleaved thinking 组 (K1-K10 落地): 4 个新工具 --- #
    # 全部走 state.watcher (server-scoped, 跨 interpreter 观察), 与旧组独立.

    @mcp.tool()
    async def ctml_append(logos: str) -> list[ContentBlock]:
        """铺一段 CTML 到执行轨道并立即返回, 附带累计执行游标图 (K8: append 即 observe).

        语义:
        - 以 append 分支起 interpreter, feed logos, wait_compiled 后返回 —— 不等 wait_stopped.
        - 返回内容 = watcher.drain() 累计事件 (跨 interpreter 一直沉淀的结果) + 当下 status.
        - 编译期异常不 rethrow; 走 InterpreterStopped 事件进 buffer, 在返回的投影里可见.
        - 运行期结果异步沉淀在 watcher, 下一次调用时读回 (K8: 前进即观察).
        """
        if not state.toolset or not state.watcher:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]

        shell = state.toolset.shell
        interpreter = await shell.interpreter(kind='append', clear_after_exit=False)
        async with interpreter:
            interpreter.feed(logos)
            # throw=False: 编译期错误交给 tracer, 由 on_interpreter_stopped 写进 watcher
            await interpreter.wait_compiled(throw=False)

        events = state.watcher.drain()
        status = state.watcher.status()
        messages = _events_to_messages(events, status)
        return list(FastMCPMessageAdapter.parse_message_to_blocks(messages))

    @mcp.tool()
    async def ctml_peek() -> list[ContentBlock]:
        """只读快照: watcher.buffered() + status, **不 drain**, 不阻塞. 用于 debug / UI 观察.

        与 ctml_append/ctml_observe 的区别: peek 不消费 buffer, 事件仍留待下次 drain.
        """
        if not state.watcher:
            return [TextContent(type='text', text="Watcher not initialized.")]
        events = state.watcher.buffered()
        status = state.watcher.status()
        messages = _events_to_messages(events, status)
        return list(FastMCPMessageAdapter.parse_message_to_blocks(messages))

    @mcp.tool()
    async def ctml_observe(budget: Optional[float] = None) -> list[ContentBlock]:
        """等 interpreter 停止 (或超时), drain 累计事件.

        - budget=None: 等到 interpreter idle 才返回.
        - budget>0: 最多等 budget 秒, 到点即返回当下 status + 累计事件.
        - budget=0: 立即返回, 等价 "drain 一次".
        """
        if not state.watcher:
            return [TextContent(type='text', text="Watcher not initialized.")]

        if budget is None:
            status = await state.watcher.wait_interpreter_done()
        elif budget > 0:
            try:
                status = await asyncio.wait_for(
                    state.watcher.wait_interpreter_done(),
                    timeout=budget,
                )
            except asyncio.TimeoutError:
                status = state.watcher.status()
        else:
            # budget<=0: 不等, 直接读一次
            status = state.watcher.status()

        events = state.watcher.drain()
        messages = _events_to_messages(events, status)
        return list(FastMCPMessageAdapter.parse_message_to_blocks(messages))

    @mcp.tool()
    async def ctml_interrupt() -> list[ContentBlock]:
        """掐掉未执行段 (K2: 对读头前方轨道动手), 返回 drain 到的所有累计事件 + 当下 status.

        clear() 会关掉当前 interpreter, tracer 收 on_interpreter_stopped 事件.
        """
        if not state.toolset or not state.watcher:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]

        await state.toolset.shell.clear()
        events = state.watcher.drain()
        status = state.watcher.status()
        messages = _events_to_messages(events, status)
        return list(FastMCPMessageAdapter.parse_message_to_blocks(messages))


def _bootstrap_env(
        mode: str | None,
        scope: str | None,
        network: str | None,
) -> Environment:
    """入口显式构造 + seal (§UU-1). Host 消费 sealed singleton."""
    env = Environment(mode=mode, scope=scope, network=network)
    env.seal()
    return env


def main_entry(
        mode: str | None = None,
        scope: str | None = None,
        network: str | None = None,
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
    moss_host = Host(env=_bootstrap_env(mode, scope, network))
    state = ServerState()
    # 注册对应的工具.
    bootstrap(state, mcp)
    params = dict(
        mode=mode, scope=scope, network=network, transport=transport,
        server_name=server_name, host=host, port=port,
    )

    async def run_server():
        # 启动 MOSS 运行时环境
        async with moss_host.run() as toolset:
            # server-scoped watcher: shell 起来后立即注册, 与 server 同生命周期.
            # 挂在 async with moss_host.run() 之内保证 shell 一定 running.
            async with InterleavedThinkingToolset.new_from_shell(toolset.shell) as watcher:
                state.host = moss_host
                state.toolset = toolset
                state.watcher = watcher
                toolset.matrix.logger.info(
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
@click.option('--scope', default='default', help='网络通讯子空间 (network scope)')
@click.option('--network', default='local', help='网络驱动 (network driver)')
@click.option('--transport', type=click.Choice(['sse', 'std', 'streamable_http']), default='sse', help='通信协议')
@click.option('--host', default='127.0.0.1', help='SSE 服务地址 (仅在 transport=sse 时生效)')
@click.option('--port', default=20773, help='SSE 服务端口 (仅在 transport=sse 时生效)')
@click.option('--server-name', default='MOSS-Toolset-Server', help='MCP 服务名称')
def main(mode, scope, network, transport, host, port, server_name):
    """MOSS MCP 服务启动程序"""

    # 传递给你的 main_entry
    main_entry(
        mode=mode,
        scope=scope,
        network=network,
        transport=transport,
        server_name=server_name,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
