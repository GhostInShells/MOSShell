from ghoshell_moss.contracts.audio import AudioAction, AudioSignal
from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus


class AudioNucleus(BufferNucleus):
    """Audio signal nucleus with SPEECH_STARTED filtering.

    Current policy: only SPEECH_FINAL triggers impulse. SPEECH_STARTED is
    dropped at the nucleus boundary.

    Why keep the path: when we implement TTS preemption (Step 13), a
    SPEECH_STARTED arriving while Ghost is speaking should immediately call
    Preemptable.attenuate() — without waiting for the full sentence. At that
    point this filter becomes a routing gate, not a drop.
    """

    def add_signal(self, signal: Signal) -> None:
        audio_meta = AudioSignal.from_signal(signal)
        if audio_meta and audio_meta.action == AudioAction.SPEECH_STARTED:
            # Reserved for preemption hook — see class docstring.
            self._logger.debug("Dropping SPEECH_STARTED — preemption hook reserved for future")
            return
        super().add_signal(signal)
