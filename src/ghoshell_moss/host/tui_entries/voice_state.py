"""VoiceState — 语音控制面 TUI state.

订阅 ``ClauseTopic`` (听侧 role=user / 说侧 role=ghost 汇成的交错对话轨迹), live 打印;
通过 ``ListenerController`` 提供人类的语音控制面:

- ``/stop`` — 停听 + 上人类锁 (channel 从模型面消失)
- ``/resume`` — 解锁 + 恢复默认礼仪
- ``/etiquette <name>`` — 切换礼仪 (只选, 不设置)
- ``/status`` — 打印当前 listening 状态
- ``/clear`` — 清屏

这是未来跨 node 语音控制面的单进程参考实现: 同一套 controller 面, 换渲染/控制层.
"""
import asyncio

from prompt_toolkit.completion import Completer
from rich.text import Text

from ghoshell_moss.core.blueprint.host import MOSShellRuntime
from ghoshell_moss.core.concepts.topic import TopicClosedError
from ghoshell_moss.host.listener.controller import ListenerController
from ghoshell_moss.host.tui import TUICompleter, TUIState
from ghoshell_moss.types.topics import ClauseTopic

__all__ = ["VoiceState"]

_VOICE_COMMANDS = {
    "stop": "stop listening (human lock — channel hidden from model)",
    "resume": "resume listening (unlock + default etiquette)",
    "etiquette": "switch etiquette: /etiquette <name>",
    "send": "strong send — drain buffer / force commit (empty input + Enter does the same)",
    "status": "show listener status",
    "clear": "clear the voice view",
}


class VoiceState(TUIState):
    """语音观测 + 控制面."""

    def __init__(self, moss: MOSShellRuntime, name: str = "voice"):
        self._moss = moss
        self._name = name
        self._controller: ListenerController | None = None
        self._sub = None
        self._sub_task: asyncio.Task | None = None
        self._signal_disposer = None

    def name(self) -> str:
        return self._name

    def completer(self) -> Completer | None:
        return TUICompleter(_VOICE_COMMANDS, command_mark="/")

    def on_switch(self, alive: bool) -> None:
        if alive:
            self.console.info(
                "Voice control — clause stream (user / ghost) + listener commands.\n"
                "/stop  /resume  /etiquette <name>  /send  /status  /clear\n"
                "empty input + Enter = strong send (drain buffer / force commit)"
            )
            self._print_status()
        else:
            self.console.info("Leave voice control")

    def on_interrupt(self, event) -> None:
        # voice state 无长运行操作可打断; 中断不动作.
        pass

    def key_bindings(self):
        """绑定 enter: 输入区为空时触发 send_now (人类强发送), 有内容时走默认提交."""
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("enter")
        def _enter(event) -> None:
            buffer = event.current_buffer
            if buffer and buffer.text.strip():
                # 有内容 → 正常提交 (命令 / 文本), 不劫持.
                buffer.validate_and_handle()
                return
            # 空输入 → 人类强发送.
            self._do_send_now()

        return kb

    def handle_input(self, console_input: str) -> None:
        text = console_input.strip()
        if not text:
            return
        cmd = text.split(maxsplit=1)
        head = cmd[0]
        arg = cmd[1].strip() if len(cmd) > 1 else ""

        if head == "/stop":
            self._do(lambda c: c.pause(True), "listener stopped — human lock on")
        elif head == "/resume":
            self._do(lambda c: c.pause(False), "listener resumed — default etiquette")
        elif head == "/etiquette":
            if self._controller is None:
                self._print_unavailable()
            elif not arg:
                names = [s.name for s in self._controller.etiquette_config().etiquettes]
                self.console.info(f"available etiquettes: {', '.join(names)}")
            else:
                self.console.info(self._controller.activate_etiquette(arg))
        elif head == "/send":
            self._do_send_now()
        elif head == "/status":
            self._print_status()
        elif head == "/clear":
            self.console.clear()
        else:
            self.console.info(
                f"unknown voice command: {text} — try /stop /resume /etiquette <name> /send /status /clear"
            )

    async def __aenter__(self):
        # controller 是控制面 (可选); clause 订阅是观测面 (无条件) — 说侧/听侧都 pub
        # 到同一 ClauseTopic, 即使 --voice=speak (只说不听) 也应看到 ghost 的 clause.
        try:
            self._controller = self._moss.matrix.container.get(ListenerController)
        except Exception:
            self._controller = None
        if self._controller is not None:
            # signal 发射观测 (debug hint) — 每次 FIRST/TAIL 触发打印一行, 观测
            # "signal 没到" 是 emit=False 还是没接通.
            self._signal_disposer = self._controller.on_signal_emit(self._render_signal)
        topics = self._moss.matrix.session.topics
        self._sub = topics.subscribe_model(ClauseTopic)
        await self._sub.__aenter__()
        self._sub_task = asyncio.get_running_loop().create_task(self._consume_clauses())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._signal_disposer is not None:
            self._signal_disposer()
            self._signal_disposer = None
        if self._sub_task is not None and not self._sub_task.done():
            self._sub_task.cancel()
            try:
                await self._sub_task
            except asyncio.CancelledError:
                pass
            self._sub_task = None
        if self._sub is not None:
            await self._sub.__aexit__(None, None, None)
            self._sub = None

    # ── internals ──

    def _do(self, action, ok_msg: str) -> None:
        if self._controller is None:
            self._print_unavailable()
            return
        action(self._controller)
        self.console.info(ok_msg)

    def _do_send_now(self) -> None:
        if self._controller is None:
            self._print_unavailable()
            return
        result = self._controller.send_now()
        if result == "buffer empty — nothing to send":
            self.console.hint("buffer empty — nothing to send")
        else:
            self.console.notice(result)

    def _print_unavailable(self) -> None:
        self.console.info("listener not available — run with --voice listen (or all)")

    def _print_status(self) -> None:
        if self._controller is None:
            self._print_unavailable()
            return
        snap = self._controller.snapshot()
        wired = "wired" if self._controller.signal_wired() else "NOT wired"
        self.console.info(
            f"etiquette={snap.etiquette}  listening={snap.listening}  "
            f"paused={snap.paused}  signal={wired}"
        )
        buf = self._controller.buffer_status()
        if buf:
            self.console.hint(f"buffer: {buf}")

    async def _consume_clauses(self) -> None:
        try:
            while True:
                try:
                    clause = await self._sub.poll_model()
                except TopicClosedError:
                    break
                if clause is not None:
                    self._render_clause(clause)
        except asyncio.CancelledError:
            pass

    def _render_clause(self, clause: ClauseTopic) -> None:
        role = clause.role or "?"
        style = "bold green" if role == "user" else "bold cyan" if role == "ghost" else "white"
        text = Text()
        text.append(f"[{role}] ", style=style)
        text.append(clause.text or "")
        self.console.rprint(text)

    def _render_signal(self, hint: str) -> None:
        text = Text()
        text.append("[signal] ", style="bold magenta")
        text.append(hint)
        self.console.rprint(text)
