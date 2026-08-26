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
from ghoshell_moss.core.blueprint.host import IHost, MOSShellRuntime
from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.core.blueprint.environment import Environment
import click

# 回合制 MCP 面下阻塞动词 (exec/observe) 的最大等待时限. budget=None 映射到它,
# 更大的值 clamp 到它. 只截断等待, 不中断任务 —— 详见 _wait_stopped docstring.
MAX_WAIT_BUDGET = 30.0
# facade 类工具 (moss_instruction full_facade / get_facade) 刷新 channel metas 的
# 防重窗口: 窗口内重复刷新直接跳过, 避免每帧扫 channel 树.
FACADE_REFRESH_STALE = 1.0


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
                        mime_type=base64_image.source['media_type'],
                    )


# 2. 定义状态容器，用于在 MCP 运行时保存 moss 实例
class ServerState:
    def __init__(self, host: IHost, runtime: MOSShellRuntime, trajectory: MShellTrajectory):
        self.host: IHost = host
        self.shell_runtime: MOSShellRuntime = runtime
        # server 级 trajectory — 跨 MCP 调用持有 shell 观察器, 解决 K10 跨-interpreter 历史丢失.
        # 生命周期与 MCP server 同长, 在 async with moss_host.run() 之内挂载.
        self.trajectory: MShellTrajectory = trajectory
        # fire-and-forget interpreter task 引用池, 防 GC. task 自绑 done_callback 自移除.
        self.background_tasks: set[asyncio.Task] = set()


def bootstrap(state: ServerState, mcp: MCPServer):
    # 会话地基 (2 工具): 协议指令 + 能力面. 无 A/B, 是任何会话都不能少的.

    @mcp.tool()
    async def moss_instruction(full_facade: bool = True) -> str:
        """Return the MOSS system instruction (system prompt).

        Combines the Logos grammar, project context, and mode context from the
        runtime system prompter. When full_facade=True (default) it also refreshes
        channel metas and appends the full channel operation surface — call with
        the default on the first turn to bootstrap your world model in one shot.
        Pass full_facade=False to re-read only the system prompt (cheaper,
        prefix-cache friendly).
        """
        if not state.shell_runtime or not state.trajectory:
            return "Error: MOSS not initialized."
        text = state.shell_runtime.system_prompter.base_instruction()
        if full_facade:
            await state.shell_runtime.shell.refresh_metas(
                timeout=5.0, stale_time=FACADE_REFRESH_STALE,
            )
            facade_text = state.trajectory.facade.full_facade()
            if facade_text:
                text += "\n\n# MOSS channels\n\n" + facade_text
        return text

    @mcp.tool()
    async def full_facade() -> list[ContentBlock]:
        """Pull the full channel operation surface — every channel's current facade.

        Channel metas are refreshed (debounced) first, so the surface reflects
        live state. Use get_channel_facade for a single channel's detail.
        """
        if not state.shell_runtime or not state.trajectory:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        await state.shell_runtime.shell.refresh_metas(
            timeout=5.0, stale_time=FACADE_REFRESH_STALE,
        )
        text = state.trajectory.facade.full_facade()
        return [TextContent(type='text', text=text)]

    @mcp.tool()
    async def get_channel_facade(path: str) -> list[ContentBlock]:
        """Pull a single channel's facade by its path (e.g. 'desktop.bash', or '' for __main__).

        Channel metas are refreshed (debounced) first, so the surface reflects
        live state. Returns an error message for an unknown or empty channel.
        """
        if not state.shell_runtime or not state.trajectory:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        await state.shell_runtime.shell.refresh_metas(
            timeout=5.0, stale_time=FACADE_REFRESH_STALE,
        )
        text = state.trajectory.facade.get_channel_full_facade(path)
        if not text:
            return [TextContent(
                type='text',
                text=f"Channel '{path}' not found or empty facade.",
            )]
        return [TextContent(type='text', text=text)]

    # --- CTML 交互动词 (5 个) --- #
    # 所有动词共用同一个投影尾段 _drain_and_project: 拉 trajectory 累计事件 + 当下 shell status.
    # 带 logos 的动词走 _spawn_interpreter: fire-and-forget task 内跑完整 interpreter 生命周期
    # (feed → wait_compiled → set(compiled) → wait_stopped → async with exit close), MCP 函数
    # 只 await 生命周期节点 (Event), 不阻塞在 async with 内 —— 中断因此是同步动作, 不牵扯执行.

    async def _drain_and_project() -> list[ContentBlock]:
        # open/close/pin 会改 channel 树 (virtual children), 投影 facade 前先刷新
        # metas, 否则 facade 反映的是上一轮状态 (如 close 后子 channel 仍残留).
        await state.shell_runtime.shell.refresh_metas(timeout=5.0)
        messages = state.trajectory.pop_frame().project()
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
        interpreter = await state.shell_runtime.shell.interpreter(kind=kind)

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
        """Append a CTML logos to the execution track; return once parsed, commands keep running in background.

        Also returns all results completed since the last call. Choose this verb when
        you do not need to see the result before continuing.
        """
        if not state.shell_runtime or not state.trajectory:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        compiled, _ = await _spawn_interpreter('append', logos)
        await compiled.wait()
        return await _drain_and_project()

    @mcp.tool()
    async def ctml_exec(logos: str, budget: Optional[float] = None) -> list[ContentBlock]:
        """Append a CTML logos and wait until its commands finish (or budget elapses), returning all results.

        budget bounds only this wait (seconds, default 30, max 30) — it never interrupts
        commands. Timed-out commands keep running; their results arrive on the next verb
        call. Use ctml_interrupt to actually stop.
        """
        if not state.shell_runtime or not state.trajectory:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        compiled, stopped = await _spawn_interpreter('append', logos)
        await compiled.wait()
        await _wait_event(stopped, budget)
        return await _drain_and_project()

    @mcp.tool()
    async def moss_observe(budget: Optional[float] = None) -> list[ContentBlock]:
        """Read the observation trajectory: return all results completed since the last call.

        If commands are still running, wait up to budget seconds (default 30, max 30).
        budget<=0 returns an immediate snapshot. Read-only — emits no CTML, interrupts nothing.
        """
        if not state.shell_runtime or not state.trajectory:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        # 直接从当前 interpreter 拿 wait_stopped 信号 — 上一次 append/exec/replan 的 task
        # 若还在跑, shell.interpreting() 就是活的; 若已自然结束, is_running=False, 快照返回.
        interp = state.shell_runtime.shell.interpreting()
        if interp is not None and interp.is_running():
            if budget is not None and budget <= 0:
                pass
            else:
                wait = MAX_WAIT_BUDGET if budget is None else min(budget, MAX_WAIT_BUDGET)
                try:
                    await asyncio.wait_for(interp.wait_stopped(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
        return await _drain_and_project()

    @mcp.tool()
    async def ctml_replan(logos: str) -> list[ContentBlock]:
        """Cancel all not-yet-started commands on the current track and append new CTML; return once parsed.

        Already-running commands continue to completion; their records appear alongside
        the cancelled-command records in the return. Choose this verb when the plan changed.
        """
        if not state.shell_runtime or not state.trajectory:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        compiled, _ = await _spawn_interpreter('clear', logos)
        await compiled.wait()
        return await _drain_and_project()

    @mcp.tool()
    async def ctml_interrupt() -> list[ContentBlock]:
        """Emergency stop: immediately cancel all running and pending commands. No arguments.

        Returns the interrupted records. Difference from ctml_replan: interrupt only
        stops, it does not append new logos.
        """
        if not state.shell_runtime or not state.trajectory:
            return [TextContent(type='text', text="MOSS Runtime not initialized.")]
        await state.shell_runtime.shell.clear()
        return await _drain_and_project()


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
        transport: Literal['sse', 'std', 'streamable_http'] = 'streamable_http',
        server_name: str = 'MOSS-shell_runtime-Server',
        host: str = '127.0.0.1',
        port: int = 20773,
) -> None:
    """启动 MOSS MCP 服务端"""
    mcp = MCPServer(server_name)
    moss_host = Host(env=_bootstrap_env(mode, scope, network))
    # 注册对应的工具.
    params = dict(
        mode=mode, scope=scope, network=network, transport=transport,
        server_name=server_name, host=host, port=port,
    )

    async def run_server():
        # 启动 MOSS 运行时环境
        async with moss_host.run() as runtime:
            # server-scoped trajectory: shell 起来后立即注册, 与 server 同生命周期.
            # 挂在 async with moss_host.run() 之内保证 shell 一定 running.
            async with MShellTrajectory(runtime.shell) as trajectory:
                state = ServerState(host=moss_host, runtime=runtime, trajectory=trajectory)
                bootstrap(state, mcp)
                runtime.matrix.logger.info(
                    'Moss MCP shell_runtime started with params: %r',
                    params,
                )
                # 启动 MCP Server transport
                if transport == 'sse':
                    await mcp.run_sse_async(host=host, port=port)
                elif transport == 'std':
                    await mcp.run_stdio_async()
                elif transport == 'streamable_http':
                    # 走 matrix.aserve_mcp (stateless streamable-http): 关停时由 uvicorn
                    # 干净收尾, 无 SSE 长连接的排干竞态, 并纳入 matrix 生命周期.
                    await runtime.matrix.aserve_mcp(mcp, host=host, port=port)
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
@click.option('--transport', type=click.Choice(['sse', 'std', 'streamable_http']), default='streamable_http', help='通信协议')
@click.option('--host', default='127.0.0.1', help='MCP 服务地址 (network 传输时生效)')
@click.option('--port', default=20773, help='MCP 服务端口 (network 传输时生效)')
@click.option('--server-name', default='MOSS-shell_runtime-Server', help='MCP 服务名称')
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
