from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.core.concepts.speech import TTS, Speech, SpeechStream, StreamAudioPlayer
from ghoshell_moss.speech.mock import MockSpeech
from ghoshell_moss.speech.stream_tts_speech import TTSSpeech, TTSSpeechStream


def make_baseline_tts_speech(
    player: StreamAudioPlayer | None = None,
    tts: TTS | None = None,
    logger: LoggerItf | None = None,
) -> TTSSpeech:
    """
    基线示例.
    """
    from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer
    from ghoshell_moss.speech.volcengine_tts import VolcengineTTS

    return TTSSpeech(
        player=player or PyAudioStreamPlayer(),
        tts=tts or VolcengineTTS(),
        logger=logger,
    )
