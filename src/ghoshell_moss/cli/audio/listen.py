"""listen command — HostListener multi-mode voice input probe.

Three modes, each an independent async state machine. Each mode owns its own
HostListener (and, for ``enter``, its own prompt-toolkit session) and lifecycle,
so the state machines can be reused as a GUI node sample with a different
interaction surface swapped in:

  - once   listen until the first complete segment (one turn), then exit.
  - always continuous listen, no commit, stop on cancel/timeout.
  - enter  prompt-toolkit loop: Enter commits a segment.

Recognition results are translated into listener signals (first/clause/tail) and
broadcast over the session bus — observable cross-process by the signal_receiver node:

    moss --mode system_test nodes run .moss/system_test_nodes/signal_receiver
    moss --mode system_test audio listen --listen-mode once
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.utils import echo, is_ai_mode, print_error, print_info, print_success, print_warning
from ghoshell_moss.contracts.asr import RecognitionPhase, RecognitionEvent
from ghoshell_moss.contracts.audio import AudioCaptureConfig, AudioCaptureSource
from ghoshell_moss.contracts.configs import get_or_create_conf, ConfigStore
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerPacket, new_listener_signal
from ghoshell_moss.host.listener.listener import HostListener
from ghoshell_moss.host.listener.volcengine_sauc import VolcengineSaucASR, VolcengineSaucConfig

_MODES = ("once", "always", "enter")


@audio_app.command("listen")
def listen_cmd(
    listen_mode: str = typer.Option("once", "--listen-mode", "-m", help="State machine: once | always | enter."),
    timeout: float = typer.Option(60.0, "--timeout", "-t", help="Overall session timeout in seconds."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Capture device name pattern."),
    emit_signals: bool = typer.Option(True, "--signals/--no-signals", help="Broadcast listener signals to the session bus."),
    json_mode: bool = typer.Option(False, "--json", help="Emit recognition records as JSON lines."),
) -> None:
    """Listen through HostListener; each mode is its own async state machine."""
    if listen_mode not in _MODES:
        print_error(f"unknown listen mode '{listen_mode}' — choose from {', '.join(_MODES)}")
        raise typer.Exit(code=2)

    matrix = Matrix.new("audio_listen", category="cli")
    result = matrix.run(lambda m: _async_listen(
        m,
        listen_mode=listen_mode,
        timeout=timeout,
        device=device,
        emit_signals=emit_signals,
        json_mode=json_mode,
    ))
    if result is None:
        return
    duration, turn_count, clause_count, interrupted = result
    if interrupted:
        print_warning("session interrupted")
    else:
        print_success(f"session done: {duration:.1f}s, {turn_count} turns, {clause_count} clauses")


# ── shared helpers ──


@dataclass
class _Ctx:
    """IoI-bound pieces handed to each state machine. The mode builds its own listener."""

    capture: AudioCaptureSource
    asr: VolcengineSaucASR
    session: Any  # matrix Session — signal broadcast target
    logger: Any
    timeout: float
    emit_signals: bool
    json_mode: bool


@dataclass
class _Stats:
    turns: int = 0
    clauses: int = 0


class _PacketTranslator:
    """RecognitionEvent (text axis) -> listener packets (first/clause/tail).

    clause_index counts clauses per segment. FIRST comes from the recognizer's
    own FIRST phase (the first meaningful packet), not synthesized here.
    """

    def __init__(self) -> None:
        self._clause_index = 0

    def translate(self, result: RecognitionEvent) -> list[tuple[ListenerPacket, str, int]]:
        packets: list[tuple[ListenerPacket, str, int]] = []
        if result.phase == RecognitionPhase.FIRST:
            self._clause_index = 0
            packets.append((ListenerPacket.FIRST, result.text, 0))
        elif result.phase == RecognitionPhase.CLAUSE:
            self._clause_index += 1
            clause_text = result.clause.text if result.clause else result.text
            packets.append((ListenerPacket.CLAUSE, clause_text, self._clause_index))
        elif result.phase == RecognitionPhase.TAIL:
            packets.append((ListenerPacket.TAIL, result.text, self._clause_index))
        return packets


def _emit_signal(session, packet: ListenerPacket, result: RecognitionEvent, text: str,
                 clause_index: int) -> None:
    clause = result.clause
    session.add_signal(new_listener_signal(
        packet,
        text,
        turn_id=result.segment_id,
        clause_index=clause_index,
        start_ms=clause.start_ms if clause else 0,
        end_ms=clause.end_ms if clause else 0,
        description=f"listener:{packet.value}",
    ))


def _render(packet: ListenerPacket, result: RecognitionEvent, text: str, clause_index: int,
            json_mode: bool) -> None:
    if json_mode:
        clause = result.clause
        echo(json.dumps({
            "packet": packet.value,
            "phase": result.phase.value,
            "text": text,
            "segment_id": result.segment_id,
            "clause_index": clause_index,
            "start_ms": clause.start_ms if clause else 0,
            "end_ms": clause.end_ms if clause else 0,
            "error": result.error or None,
        }, ensure_ascii=False))
        return

    if is_ai_mode():
        if packet in (ListenerPacket.CLAUSE, ListenerPacket.TAIL):
            echo(text)
            echo("---")
        return

    if packet == ListenerPacket.FIRST:
        return
    if packet == ListenerPacket.CLAUSE:
        _commit_line(text)
        echo("---")
    elif packet == ListenerPacket.TAIL:
        _commit_line(f"[tail] {text}")
        echo("")


def _handle_result(result: RecognitionEvent, *, translator: _PacketTranslator, session: Any,
                   emit_signals: bool, json_mode: bool, stats: _Stats) -> None:
    """Translate + emit + render one recognition result; update stats."""
    for packet, text, clause_index in translator.translate(result):
        if emit_signals:
            _emit_signal(session, packet, result, text, clause_index)
        _render(packet, result, text, clause_index, json_mode)
    if result.phase == RecognitionPhase.CLAUSE:
        stats.clauses += 1
    elif result.phase == RecognitionPhase.TAIL:
        stats.turns += 1


def _banner(ctx: _Ctx, mode: str, hint: str) -> None:
    if ctx.json_mode or is_ai_mode():
        return
    print_info(
        f"device={ctx.capture.device_explain()}  mode={mode}  "
        f"signals={'on' if ctx.emit_signals else 'off'}  timeout={ctx.timeout}s"
    )
    echo(hint)


# ── three state machines ──


async def _run_once(ctx: _Ctx) -> _Stats | None:
    """Listen until the first complete segment, then exit."""
    listener = HostListener(capture=ctx.capture, asr=ctx.asr, logger=ctx.logger)
    stats = _Stats()
    translator = _PacketTranslator()
    handle = partial(
        _handle_result, translator=translator, session=ctx.session,
        emit_signals=ctx.emit_signals, json_mode=ctx.json_mode, stats=stats,
    )
    got_segment = asyncio.Event()

    async with listener:
        if "not started" in ctx.capture.device_explain():
            print_error("capture device not started — may be locked by another process")
            return None
        _banner(ctx, "once", "speak now — listening for one utterance, then exit. Ctrl+C to stop.\n")

        state = await listener.listen()

        committed = False

        def on_result(result: RecognitionEvent) -> None:
            nonlocal committed
            handle(result)
            # 第一句稳定 (VAD 判停) 后主动 commit, 触发 is_last_package → 切段 → 退出.
            if result.phase == RecognitionPhase.CLAUSE and not committed:
                committed = True
                state.commit()

        state.on_recognition_result(on_result)
        state.on_recognition_segment(lambda _segment: got_segment.set())
        async with state:
            try:
                await asyncio.wait_for(got_segment.wait(), timeout=ctx.timeout)
            except asyncio.TimeoutError:
                print_warning("session timeout")
            else:
                # Give the just-produced segment's signals a beat to flush before exit.
                await asyncio.sleep(0.2)
    return stats


async def _run_always(ctx: _Ctx) -> _Stats | None:
    """Continuous listen; no commit, no exit — stop on cancel/timeout."""
    listener = HostListener(capture=ctx.capture, asr=ctx.asr, logger=ctx.logger)
    stats = _Stats()
    translator = _PacketTranslator()
    on_result = partial(
        _handle_result, translator=translator, session=ctx.session,
        emit_signals=ctx.emit_signals, json_mode=ctx.json_mode, stats=stats,
    )

    async with listener:
        if "not started" in ctx.capture.device_explain():
            print_error("capture device not started — may be locked by another process")
            return None
        _banner(ctx, "always", "listening continuously — Ctrl+C to stop.\n")

        state = await listener.listen()
        state.on_recognition_result(on_result)
        async with state:
            await asyncio.sleep(ctx.timeout)
            print_warning("session timeout")
    return stats


async def _run_enter(ctx: _Ctx) -> _Stats | None:
    """prompt-toolkit loop: Enter commits a segment; recognition renders concurrently."""
    listener = HostListener(capture=ctx.capture, asr=ctx.asr, logger=ctx.logger)
    prompt_session = PromptSession()
    stats = _Stats()
    translator = _PacketTranslator()
    on_result = partial(
        _handle_result, translator=translator, session=ctx.session,
        emit_signals=ctx.emit_signals, json_mode=ctx.json_mode, stats=stats,
    )

    async with listener:
        if "not started" in ctx.capture.device_explain():
            print_error("capture device not started — may be locked by another process")
            return None
        _banner(ctx, "enter", "press Enter to commit a segment, Ctrl+C to stop.\n")

        state = await listener.listen()
        state.on_recognition_result(on_result)
        async with state:
            async def _prompt_loop() -> None:
                with patch_stdout(raw=True):
                    while True:
                        try:
                            await prompt_session.prompt_async("press Enter to commit > ")
                        except KeyboardInterrupt:
                            return
                        state.commit()

            try:
                await asyncio.wait_for(_prompt_loop(), timeout=ctx.timeout)
            except asyncio.TimeoutError:
                print_warning("session timeout")
    return stats


_RUNNERS = {
    "once": _run_once,
    "always": _run_always,
    "enter": _run_enter,
}


# ── dispatcher ──


async def _async_listen(matrix, *, listen_mode: str, timeout: float, device: Optional[str],
                        emit_signals: bool, json_mode: bool):
    con = matrix.container
    config = get_or_create_conf(con, VolcengineSaucConfig())

    asr = VolcengineSaucASR(config=config, logger=matrix.logger)

    if device is not None:
        conf = get_or_create_conf(con, AudioCaptureConfig())
        conf.device_pattern = device

    capture = con.get(AudioCaptureSource)
    if capture is None:
        print_error("AudioCaptureSource not registered")
        return None

    ctx = _Ctx(
        capture=capture,
        asr=asr,
        session=matrix.session,
        logger=matrix.logger,
        timeout=timeout,
        emit_signals=emit_signals,
        json_mode=json_mode,
    )

    session_start = time.monotonic()
    interrupted = False
    stats: _Stats | None = None
    try:
        stats = await _RUNNERS[listen_mode](ctx)
    except asyncio.CancelledError:
        interrupted = True

    duration = time.monotonic() - session_start
    if stats is None:
        return duration, 0, 0, interrupted
    return duration, stats.turns, stats.clauses, interrupted


def _live_write(text: str) -> None:
    """Update the current terminal line in-place with partial ASR text."""
    sys.stdout.write(f"\r\033[K  {text}")
    sys.stdout.flush()


def _commit_line(text: str) -> None:
    """Clear the live-update line and print the final text."""
    sys.stdout.write(f"\r\033[K  {text}\n")
    sys.stdout.flush()
