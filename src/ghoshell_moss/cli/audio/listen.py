"""listen command — HostListener multi-mode voice input probe.

The assembly (capture + seedasr + controller) is reused from
``ghoshell_moss.host.nodes.listener_node``; this file only adds the CLI
presentation surface (banner / rendering / stats) and the interactive ``enter`` mode.

Recognition results are translated into listener signals (first/clause/tail) and
broadcast over the session bus — observable cross-process by the signal_receiver node:

    moss --mode system_test nodes run .moss/system_test_nodes/signal_receiver
    moss --mode system_test audio listen --listen-mode once
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.utils import echo, is_ai_mode, print_error, print_info, print_success, print_warning
from ghoshell_moss.contracts.asr import RecognitionPhase, RecognitionEvent
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerPacket
from ghoshell_moss.host.listener.controller import ListenerController, PacketTranslator
from ghoshell_moss.host.nodes.listener_node import assemble_controller

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
    """Assembled controller + the CLI-only surface (capture, flags)."""

    controller: ListenerController
    capture: AudioCaptureSource
    timeout: float
    emit_signals: bool
    json_mode: bool


@dataclass
class _Stats:
    turns: int = 0
    clauses: int = 0


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
            "clause_created": clause.created if clause else None,
            "created": result.created,
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


def _handle_result(result: RecognitionEvent, *, translator: PacketTranslator, json_mode: bool,
                   stats: _Stats) -> None:
    """Translate + render one recognition result; update stats. 信号发射在 ListenerController."""
    for packet, text, clause_index in translator.translate(result):
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
    stats = _Stats()
    translator = PacketTranslator()
    handle = partial(_handle_result, translator=translator, json_mode=ctx.json_mode, stats=stats)

    if "not started" in ctx.capture.device_explain():
        print_error("capture device not started — may be locked by another process")
        return None
    _banner(ctx, "once", "speak now — listening for one utterance, then exit. Ctrl+C to stop.\n")
    ctx.controller.on_recognition_result(handle)
    await ctx.controller.once(timeout=ctx.timeout)
    return stats


async def _run_always(ctx: _Ctx) -> _Stats | None:
    stats = _Stats()
    translator = PacketTranslator()
    on_result = partial(_handle_result, translator=translator, json_mode=ctx.json_mode, stats=stats)

    if "not started" in ctx.capture.device_explain():
        print_error("capture device not started — may be locked by another process")
        return None
    _banner(ctx, "always", "listening continuously — Ctrl+C to stop.\n")
    ctx.controller.on_recognition_result(on_result)
    await ctx.controller.always(timeout=ctx.timeout)
    print_warning("session timeout")
    return stats


async def _run_enter(ctx: _Ctx) -> _Stats | None:
    prompt_session = PromptSession()
    stats = _Stats()
    translator = PacketTranslator()
    on_result = partial(_handle_result, translator=translator, json_mode=ctx.json_mode, stats=stats)

    if "not started" in ctx.capture.device_explain():
        print_error("capture device not started — may be locked by another process")
        return None
    _banner(ctx, "enter", "press Enter to commit a segment, Ctrl+C to stop.\n")

    state = await ctx.controller.listen()
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
    controller = await assemble_controller(matrix, device=device, emit_signals=emit_signals)
    capture = matrix.container.get(AudioCaptureSource)
    if capture is None:
        print_error("AudioCaptureSource not registered")
        return None

    ctx = _Ctx(
        controller=controller,
        capture=capture,
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


def _commit_line(text: str) -> None:
    """Clear the live-update line and print the final text."""
    sys.stdout.write(f"\r\033[K  {text}\n")
    sys.stdout.flush()
