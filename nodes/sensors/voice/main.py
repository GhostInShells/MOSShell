"""Voice input node — thin shell that constructs the voice core from IoC and provides channels.

Usage:
    moss nodes run nodes/sensors/voice                    # headless (default)
    moss nodes run nodes/sensors/voice -- --mode webview  # with GUI (Round 2)

The node:
1. Reads VoiceConfig from voice.json (node home)
2. Constructs VoiceControllerImpl (host/speech/voice)
3. Registers a matrix-aware EventHandler (adapter → topics + signals)
4. Provides the voice control channel to the matrix
"""
from __future__ import annotations

import asyncio
import logging
import pathlib

import click

from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.topic import TopicModel
from ghoshell_moss.core.mindflow.audio_signal import AudioAction, AudioSignal
from ghoshell_moss.host.speech.voice.capture import VoiceCapture
from ghoshell_moss.host.speech.voice.contracts import (
    AsrFinal,
    AsrPartial,
    BufferUpdated,
    EventHandler,
    StreamStateChanged,
    VoiceConfig,
    VoiceController,
    VoiceMode,
    VoiceNodeRuntime,
)
from ghoshell_moss.host.speech.voice.controller import VoiceControllerImpl

_TTS_TOPIC_DEVICE = "speaker"
_NODE_DIR = pathlib.Path(__file__).resolve().parent


# ── adapter: core events → matrix (topics + signals) ──


class VoiceMatrixAdapter:
    """Bridges voice core lifecycle events to matrix Session (topics + signals).

    Implements EventHandler protocol — callback methods are called synchronously
    from the controller's asyncio loop.
    """

    def __init__(self, session, controller: VoiceController, logger: logging.Logger):
        self._session = session
        self._ctrl = controller
        self._log = logger
        self._started_ids: set[str] = set()  # utterance ids that got SPEECH_STARTED

    def on_stream_state_changed(self, e: StreamStateChanged) -> None:
        pass

    def on_asr_partial(self, e: AsrPartial) -> None:
        # First non-empty partial → SPEECH_STARTED (complete=False, occupies attention)
        if e.utterance_id not in self._started_ids:
            self._started_ids.add(e.utterance_id)
            meta = AudioSignal(action=AudioAction.SPEECH_STARTED)
            self._add_signal(e.utterance_id, meta, e.text, complete=False)

    def on_asr_final(self, e: AsrFinal) -> None:
        # Publish SpeechTopic — sentence-level broadcast (shared schema, role=human)
        from ghoshell_moss.topics.audio import SpeechTopic

        self._session.topics.pub(
            SpeechTopic(text=e.text, role="human", speaker_name="User"),
        )
        # Complete the utterance — same id → same-id absorb in mindflow
        meta = AudioSignal(action=AudioAction.SPEECH_FINAL)
        self._add_signal(e.utterance_id, meta, e.text, complete=True)
        self._started_ids.discard(e.utterance_id)

    def on_buffer_updated(self, e: BufferUpdated) -> None:
        pass  # buffer growth — no signal needed

    def _add_signal(self, uid: str, meta: AudioSignal, text: str, *, complete: bool) -> None:
        from ghoshell_moss.core.blueprint.mindflow import Priority, Signal
        from ghoshell_moss.message import Message

        sig = Signal(
            id=uid,
            name=meta.signal_name(),
            priority=Priority.WARNING,
            messages=[Message.new().with_content(text)],
            description=text,
            metadata=meta.model_dump(exclude_defaults=True, exclude_none=True),
            complete=complete,
        )
        self._session.add_signal(sig)


# ── node entry ──


async def _main(matrix: Matrix, config: VoiceConfig, mode: str) -> None:
    logger = matrix.logger or logging.getLogger("moss.voice.node")
    logger.info("Voice node starting (mode=%s)", mode)

    # Gate check: read AudioRuntimeTopic to detect TTS playback
    from ghoshell_moss.topics.audio import AudioRuntimeTopic

    tts_window = matrix.session.topics.create_window_for(AudioRuntimeTopic, max_size=10)

    def _tts_playing() -> bool:
        for t in reversed(tts_window.values()):
            if getattr(t, "device_name", "") == _TTS_TOPIC_DEVICE:
                return getattr(t, "running", False)
        return False

    # Construct the core controller
    ctrl = VoiceControllerImpl(config, gate_check=_tts_playing, logger=logger)

    # Register the matrix adapter
    adapter = VoiceMatrixAdapter(matrix.session, ctrl, logger)
    ctrl.add_handler(adapter)

    # Build the voice channel and provide it
    from ghoshell_moss.host.speech.voice.channel import VoiceChannel

    # Inject VoiceController into channel's IoC scope
    main_chan = new_channel(name="main", description="voice input node")
    main_chan.build.with_binding(VoiceController, ctrl)
    main_chan.import_channels(VoiceChannel.factory)

    await matrix.provide_channel(main_chan)


@click.command()
@click.option(
    "--mode",
    default="headless",
    type=click.Choice(["headless", "webview"]),
    help="Launch mode: headless (pure process) or webview (with GUI)",
)
def cli(mode: str) -> None:
    """Voice input node — thin shell for host/speech/voice core."""
    # Load config from node home
    storage = LocalStorage(_NODE_DIR)

    try:
        config = VoiceConfig.load_json(storage)
    except Exception:
        config = VoiceConfig()
        config.save_json(storage)

    # Replace env-style strings with real values (Volcengine ASR keys)
    config = config.resolve()
    config.save_json(storage)

    async def run(matrix: Matrix) -> None:
        await _main(matrix, config, mode)

    Matrix.discover().run(run)


if __name__ == "__main__":
    cli()
