import asyncio
import logging
import os
import threading
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from pynput import keyboard as _keyboard
from rich.console import Console
from rich.panel import Panel

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss_contrib.asr.async_concepts import (
    AsyncListenerService,
    AsyncListenerCallback,
    AsyncListenerStateName,
    Recognition,
)
from ghoshell_moss_contrib.asr.async_listener_service import AsyncListenerServiceImpl
from ghoshell_moss_contrib.asr.configs import ListenerConfig

load_dotenv()

# 可配置的 PTT 触发键，默认 media_play_pause
# 取值参考 pynput.keyboard.Key: media_play_pause, f1-f20, ctrl, shift, etc.
_PTT_KEY_NAME = os.getenv("VOICE_PTT_KEY", "media_play_pause")

# ── ThreadedListenerService (extracted from ConsolePTTChat) ──────────────


class ThreadedListenerService:
    """在独立线程中运行 Listener 服务，与主 event loop 完全解耦。"""

    def __init__(self, inner: AsyncListenerService, main_loop: asyncio.AbstractEventLoop, logger):
        self._inner = inner
        self._main_loop = main_loop
        self._logger = logger
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _schedule(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _await_schedule(self, coro):
        future = self._schedule(coro)
        return await asyncio.wrap_future(future)

    async def _forward_callback(self, callback):
        class _ForwardCallback(AsyncListenerCallback):
            def __init__(self, cb, main_loop, logger):
                self._cb = cb
                self._main_loop = main_loop
                self._logger = logger

            async def on_recognition(self, result):
                asyncio.run_coroutine_threadsafe(self._cb.on_recognition(result), self._main_loop)

            async def on_state_change(self, state):
                asyncio.run_coroutine_threadsafe(self._cb.on_state_change(state), self._main_loop)

            async def on_waken(self):
                asyncio.run_coroutine_threadsafe(self._cb.on_waken(), self._main_loop)

            async def on_error(self, error):
                asyncio.run_coroutine_threadsafe(self._cb.on_error(error), self._main_loop)

            async def save_batch(self, rec, audio):
                asyncio.run_coroutine_threadsafe(self._cb.save_batch(rec, audio), self._main_loop)

        return _ForwardCallback(callback, self._main_loop, self._logger)

    async def bootstrap(self):
        ready = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._inner.bootstrap())
                ready.set()
                self._loop.run_forever()
            finally:
                pending = asyncio.all_tasks(self._loop)
                for t in pending:
                    t.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True, name="voice-listener")
        self._thread.start()
        ready.wait(timeout=10)
        if not ready.is_set():
            raise RuntimeError("Listener service failed to bootstrap within 10s")
        self._logger.info("ThreadedListenerService started")

    async def shutdown(self):
        if self._loop and self._loop.is_running():
            try:
                await self._await_schedule(self._inner.shutdown())
            except Exception as e:
                self._logger.warning(f"Listener shutdown error: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._logger.info("ThreadedListenerService shutdown complete")

    async def set_callback(self, callback):
        forwarded = await self._forward_callback(callback)
        await self._await_schedule(self._inner.set_callback(forwarded))

    async def set_state(self, state: str):
        self._schedule(self._inner.set_state(state))

    async def current_state(self):
        return await self._await_schedule(self._inner.current_state())

    async def commit(self):
        self._schedule(self._inner.commit())

    async def clear_buffer(self):
        self._schedule(self._inner.clear_buffer())


# ── Voice Status Display ──────────────────────────────────────────────────


class VoiceStatusDisplay:
    """最小状态 TUI：区分识别中（流式刷新）和识别完成。"""

    def __init__(self):
        self.console = Console()
        self.state = "idle"
        self._partial_printed = False  # 跟踪是否有未闭合的 partial 行

    def show_header(self):
        self.console.print(Panel("[bold cyan]Voice Input App[/bold cyan] — input/voice", title="MOSS"))

    def show_state(self, state: str):
        """状态切换，相同状态不重复打印"""
        if state == self.state:
            return
        # 如果之前在流式打印 partial，先换行收尾
        if self._partial_printed:
            self.console.print()
            self._partial_printed = False
        icons = {"idle": "⏸", "recording": "🔴", "sending": "📤"}
        labels = {"idle": "Waiting", "recording": "Recording", "sending": "Sending"}
        self.state = state
        self.console.print(f"  {icons.get(state, '?')} [bold]{labels.get(state, state)}[/bold]")

    def show_partial(self, text: str):
        """流式覆盖打印当前识别中的部分文本"""
        # console.print + end="\r" 让 Rich 正确渲染颜色，同时不换行
        self.console.print(f"  [green]→ {text}[/green]", end="\r")
        self._partial_printed = True

    def show_recognized(self, text: str, reason: str = ""):
        """识别完成，清除 partial 行后打印最终结果"""
        reason_tag = f" [{reason}]" if reason else ""
        if self._partial_printed:
            self.console.file.write("\r\033[K")
            self.console.file.flush()
        self.console.print(f"  [cyan]✓ {text}{reason_tag}[/cyan]")
        self._partial_printed = False

    def show_sent(self):
        self.console.print(f"  [dim]📤 sent to ghost[/dim]")

    def show_footer(self):
        self.console.print(f"[dim]Press {_PTT_KEY_NAME} to talk / Ctrl+C to quit[/dim]")


# ── Main Entry ────────────────────────────────────────────────────────────


async def _toggle_recording(threaded: ThreadedListenerService):
    try:
        current_state = await threaded.current_state()
        state_name = current_state.name().value
    except Exception:
        return

    if state_name == AsyncListenerStateName.PDT_LISTENING.value:
        await threaded.commit()
    else:
        await threaded.set_state(AsyncListenerStateName.PDT_LISTENING.value)


async def main(matrix: Matrix) -> None:
    console = Console()
    logger = logging.getLogger("VoiceInput")
    logging.basicConfig(level=logging.WARNING)

    display = VoiceStatusDisplay()
    display.show_header()
    console.print("[green]Initializing voice input pipeline...[/green]")

    # 1. Listener config + service
    config = ListenerConfig()
    inner = AsyncListenerServiceImpl(
        config=config,
        logger=logger,
        audio_input=None,  # use default PyAudio input
    )

    main_loop = asyncio.get_running_loop()
    threaded = ThreadedListenerService(inner, main_loop, logger)
    await threaded.bootstrap()

    # 2. Callback: ASR text → session.add_input_signal()
    import time as _time

    _dedup = {"text": "", "ts": 0.0}  # 防重复 send
    _last_voice_state: str = ""  # 防重复 pub stream

    def _pub_voice_state(state: str) -> None:
        """Publish voice state to stream for other apps (e.g. ai_eye) to react."""
        nonlocal _last_voice_state
        if state != _last_voice_state:
            _last_voice_state = state
            matrix.session.pub_stream_delta("voice/state", state.encode())

    class VoiceCallback(AsyncListenerCallback):
        async def on_recognition(self, result: Recognition):
            if not result.text or not result.text.strip():
                return
            if result.is_last:
                display.show_recognized(result.text, result.commit_reason or "")
                display.show_state("sending")
                # 防止 VAD auto-commit 和手动 commit 重复发送同一句
                now = _time.monotonic()
                if not (result.text == _dedup["text"] and now - _dedup["ts"] < 1.5):
                    matrix.session.add_input_signal(
                        result.text,
                        description=f"voice: {result.text[:50]}",
                    )
                    _dedup["text"] = result.text
                    _dedup["ts"] = now
                display.show_sent()
                display.show_state("idle")
                display.show_footer()
                _pub_voice_state("recording_stopped")
            else:
                display.show_partial(result.text)

        async def on_state_change(self, state: str):
            if "listening" in state.lower():
                display.show_state("recording")
                _pub_voice_state("recording_started")
            # 不处理 waiting → idle，让 on_recognition(is_last=True) 统一收尾

        async def on_error(self, error: str):
            display.console.print(f"  [red]❌ {error}[/red]")

        async def on_waken(self):
            pass

        async def save_batch(self, rec: Recognition, audio: np.ndarray):
            pass

    await threaded.set_callback(VoiceCallback())

    # 3. PTT trigger (pynput keyboard listener)
    quit_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    ptt_key = getattr(_keyboard.Key, _PTT_KEY_NAME, None)

    def on_press(key):
        if ptt_key and key == ptt_key:
            asyncio.run_coroutine_threadsafe(_toggle_recording(threaded), loop)

    kb_listener = _keyboard.Listener(on_press=on_press, on_release=lambda k: True)
    kb_listener.start()
    console.print(f"[green]PTT listener started. Key: {_PTT_KEY_NAME}[/green]")

    display.show_state("idle")
    display.show_footer()

    # 4. Wait for quit
    try:
        await quit_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        console.print("[yellow]Shutting down...[/yellow]")
        if kb_listener:
            kb_listener.stop()
        await threaded.shutdown()
        console.print("[green]Voice input app stopped.[/green]")


if __name__ == "__main__":
    try:
        Matrix.discover().run(main)
    except KeyboardInterrupt:
        print("\nVoice input app stopped.")
