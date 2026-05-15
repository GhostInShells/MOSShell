from textwrap import indent

from ghoshell_moss import Command
from ghoshell_moss.core.blueprint.host import MossRuntime
from ghoshell_moss.host.tui import ConsoleOutput
from ghoshell_moss.core.blueprint.session import OutputItem

__all__ = ['MOSSRuntimeInspector']


class MOSSRuntimeInspector:
    """封装对 ToolSet 的操作与观测接口。"""

    def __init__(self, moss_runtime: MossRuntime, output: ConsoleOutput) -> None:
        self._moss_runtime = moss_runtime
        self._output = output

    def instructions(self) -> None:
        """获取当前 MOSS 的指令上下文 (Instruction)。"""
        self._output.syntax(self._moss_runtime.moss_instruction(), 'xml')

    async def dynamic(self, refresh: bool = True) -> None:
        """获取当前 MOSS 的动态上下文讯息. """
        messages = await self._moss_runtime.moss_dynamic_messages(refresh=refresh)
        self._output.output(OutputItem.new("Shell", *messages, log="moss dynamic instructions"))

    async def static(self) -> None:
        """获取当前 MOSS 的静态上下文讯息. """
        await self._moss_runtime.moss_refresh_metas()
        static = self._moss_runtime.moss_static_messages()
        self._output.syntax(static, 'xml')

    async def channel_mets(self, refresh: bool = True) -> None:
        """获取最新的 channel metas 信息."""
        if refresh:
            await self._moss_runtime.shell.refresh_metas()
        metas = self._moss_runtime.shell.channel_metas(available_only=False)
        for key, meta in metas.items():
            self._output.info("channel %s" % key)
            self._output.json(meta.model_dump_json(indent=2, ensure_ascii=False, exclude_none=True, exclude_unset=True))

    async def commands(self) -> None:
        await self._moss_runtime.moss_refresh_metas()
        commands = self._moss_runtime.shell.commands(available_only=True)
        for channel_path, group in commands.items():
            for command_name, command in group.items():
                self._output.info(Command.make_unique_name(channel_path, command_name))
                self._output.syntax(command.meta().interface, 'python')

    async def exec(self, ctml: str, interrupt: bool = True) -> None:
        """
        向运行时注入 CTML 指令。
        :param ctml: CTML 语法指令。
        :param interrupt: 是否打断当前任务并立即执行。
        """
        messages = await self._moss_runtime.moss_exec(ctml, call_soon=interrupt, wait_done=True)
        self._output.rprint(OutputItem.new("Shell", *messages, log="interpreting done"))

    async def observe(self, timeout: float = 5.0) -> None:
        """挂起等待运行状态变更。"""
        messages = await self._moss_runtime.moss_observe(timeout=timeout)
        self._output.rprint(OutputItem.new("Shell", *messages, log="observe done"))

    async def interrupt(self) -> None:
        """立即终止当前执行任务。"""
        messages = await self._moss_runtime.moss_interrupt()
        self._output.rprint(OutputItem.new("Shell", *messages, log="interrupted"))
