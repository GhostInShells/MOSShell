"""MOSS runtime exposed as an MCP server — internal module for ``moss-shell mcp``.

Formerly the standalone ``moss-mcp`` binary entry. Since 2026-08 the script
entry is gone; ``moss-shell mcp`` lazily imports ``main_entry`` here after the
``depend_mcp()`` gate. The top-level ``mcp`` import is intentional — this module
is only imported inside the mcp mode handler.
"""

from typing import Literal, Iterable, Optional
import asyncio
from mcp.server.mcpserver import MCPServer
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


# 回合制 MCP 面下阻塞动词 (exec/observe) 的最大等待时限. budget=None 映射到它,
# 更大的值 clamp 到它. 只截断等待, 不中断任务 —— 详见 _wait_stopped docstring.
MAX_WAIT_BUDGET = 30.0


class MCPMessageAdapter:

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
        # fire-and-forget interpreter task 引用池, 防 GC. task 自绑 done_callback 自移除.
        self.background_tasks: set[asyncio.Task] = set()


def _events_to_messages(events: list[ShellEvent], status: InterpreterStatus) -> list[Message]:
    """MCP 层薄封装: 直接委托到 host.interleaved_thinking.project_events.

    K8/K9 分桶投影: 空成功折叠成 <shell_tally>, observe=True 占位, 其他走 payload.
    """
    return project_events(events, status)


def bootstrap(state: ServerState, mcp: MCPServer):
    # 会话地基 (2 工具): 协议指令 + 拉模式能力面. 无 A/B, 是任何会话都不能少的.

    @mcp.tool()
    async def moss_instruction() -> str:
        """返回 MOSS 系统指令与静态能力面. 会话开始时调用一次, 了解 CTML 怎么写、有哪些 channel。"""
        if not state.toolset:
            return "Error: MOSS not initialized."
        return state.toolset.moss_instruction(True)

    @mcp.tool()
    async def get_moss_dynamic_info() -> list[ContentBlock]:
        """刷新并返回当前动态能力快照: 已上线 channel、每个 channel 的运行时上下文与状态变化.

        命令的静态签名在 instruction 里, 本工具只带动态增量. channel 有变化时调用一次 (如新 node 加入 mesh、状态切换).
        """
        if not state.toolset:
            return [TextContent(type='text', text="System not ready.")]
        msgs = await state.toolset.moss_dynamic_messages(refresh=True, max_wait=5.0)
        return list(MCPMessageAdapter.parse_message_to_blocks(msgs))

    # --- CTML 交互动词 (5 个) --- #
    # 所有动词共用同一个投影尾段 _drain_and_project: 拉 watcher 累计事件 + 当下 shell status.
    # 带 logos 的动词走 _spawn_interpreter: fire-and-forget task 内跑完整 interpreter 生命周期
    # (feed → wait_compiled → set(compiled) → wait_stopped → async with exit close), MCP 函数
    # 只 await 生命周期节点 (Event), 不阻塞在 async with 内 —— 中断因此是同步动作, 不牵扯执行.

    def _drain_and_project() -> list[ContentBlock]:
        events = state.watcher.drain()
        status = state.watcher.status()
        messages = _events_to_messages(events, status)
        return list(MCPMessageAdapter.parse_message_to_blocks(messages))

    async def _spawn_interpreter(kind: str, logos: str) -> tuple[asyncio.Event, asyncio.Event]:
        """起 interpreter task, 返回 (compiled, stopped) 两个生命周期信号 Event.

        - compiled: 在 wait_compiled 返回后 set. append/replan 只等这个.
        - stopped: 在 wait_stopped 返回后 (即所有 managing_tasks 都 done) set. exec 可选等这个.
        - 任何异常路径 finally 都 set 两者, 防 MCP 函数永久 hang.
        - task 引用挂 state.background_tasks 防 GC, done 时自移除.
        - clear_after_exit 用默认 (True): interpreter 自然 wait_stopped 返回时 managing_tasks
          已全部 done, close cancel 无副作用. 不需要反常识 flag.
        """
        compiled = asyncio.Event()
        stopped = asyncio.Event()
        interpreter = await state.toolset.shell.interpreter(kind=kind)

        async def _run() -> None:
            try:
                async with interpreter:
                    interpreter.feed(logos)
                    await interpreter.wait_compiled(throw=False)
                    compiled.set()
                    await interpreter.wait_stopped()
            finally:
                compiled.set()
                stopped.set()

        task = asyncio.create_task(_run())
        state.background_tasks.add(task)
        task.add_done_callback(state.background_tasks.discard)
        return compiled, stopped

    async def _wait_event(event: asyncio.Event, budget: Optional[float]) -> None:
        # budget 语义: 等待时限, 不是运行时限. 只截断 await, task 内 interpreter 生命周期不动.
        if budget is not None and budget <= 0:
            return
        wait = MAX_WAIT_BUDGET if budget is None else min(budget, MAX_WAIT_BUDGET)
        try:
            await asyncio.wait_for(event.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass

    @mcp.tool()
    async def ctml_append(logos: str) -> list[ContentBlock]:
        """往执行轨道铺一段 CTML, 解析完成即返回, 命令在后台继续跑. 返回同时带回自上次调用以来完成的所有结果.

        选这个动词 = 你不需要等结果就能继续下一步.
        """
        if not state.toolset or not state.watcher:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        compiled, _ = await _spawn_interpreter('append', logos)
        await compiled.wait()
        return _drain_and_project()

    @mcp.tool()
    async def ctml_exec(logos: str, budget: Optional[float] = None) -> list[ContentBlock]:
        """铺一段 CTML 并等到这批命令跑完再返回, 返回带回所有结果.

        选这个动词 = 你必须看到结果才能继续下一步.
        budget 是等待时限 (秒, 默认 30, 上限 30) —— 只截断本次等待, 绝不中断命令.
        超时未完成的命令会继续在后台跑, 结果在下次任意动词调用时带回. 想真的停下用 ctml_interrupt.
        """
        if not state.toolset or not state.watcher:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        compiled, stopped = await _spawn_interpreter('append', logos)
        await compiled.wait()
        await _wait_event(stopped, budget)
        return _drain_and_project()

    @mcp.tool()
    async def ctml_observe(budget: Optional[float] = None) -> list[ContentBlock]:
        """读取执行游标: 返回自上次调用以来完成的所有结果. 如果还有命令在跑, 最多等 budget 秒 (默认 30, 上限 30).

        budget<=0 立即返回一次快照. 本工具不铺轨、不中断、只读.
        """
        if not state.toolset or not state.watcher:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        # 直接从当前 interpreter 拿 wait_stopped 信号 — 上一次 append/exec/replan 的 task
        # 若还在跑, shell.interpreting() 就是活的; 若已自然结束, is_running=False, 快照返回.
        interp = state.toolset.shell.interpreting()
        if interp is not None and interp.is_running():
            if budget is not None and budget <= 0:
                pass
            else:
                wait = MAX_WAIT_BUDGET if budget is None else min(budget, MAX_WAIT_BUDGET)
                try:
                    await asyncio.wait_for(interp.wait_stopped(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
        return _drain_and_project()

    @mcp.tool()
    async def ctml_replan(logos: str) -> list[ContentBlock]:
        """掐掉当前轨道上所有尚未开始的命令, 铺一段新 CTML 顶上, 解析完成即返回.

        已经在执行中的命令继续跑到结束 (说出去的话收不回), 和被取消的命令记录一起在返回里.
        选这个动词 = 计划变了, 要收回未执行段并换新方向.
        """
        if not state.toolset or not state.watcher:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        compiled, _ = await _spawn_interpreter('clear', logos)
        await compiled.wait()
        return _drain_and_project()

    @mcp.tool()
    async def ctml_interrupt() -> list[ContentBlock]:
        """急停: 立即取消所有运行中与待执行的命令, 停止说话. 无参数. 返回带回被中断的记录.

        用于紧急停止, 或在换一个完全新的计划前清空. 与 ctml_replan 的区别: interrupt 只停不铺.
        """
        if not state.toolset or not state.watcher:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        await state.toolset.shell.clear()
        return _drain_and_project()


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
    mcp = MCPServer(server_name)
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
                # 启动 MCP Server transport
                if transport == 'sse':
                    await mcp.run_sse_async(host=host, port=port)
                elif transport == 'std':
                    await mcp.run_stdio_async()
                elif transport == 'streamable_http':
                    await mcp.run_streamable_http_async(host=host, port=port)
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
    """Expose MOSS runtime as an MCP server for AI coding platforms (Claude Code,
    Gemini CLI, etc.). Registers CTML execution and runtime introspection as
    MCP tools."""

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
