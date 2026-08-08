from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.speech import TTS, StreamAudioPlayer
from ghoshell_moss.core.speech.stream_tts_speech import BaseTTSSpeech


def make_baseline_tts_speech(
        player: StreamAudioPlayer | None = None,
        tts: TTS | None = None,
        logger: LoggerItf | None = None,
) -> BaseTTSSpeech:
    """
    基线示例.
    """
    from ghoshell_moss.host.speech.player.miniaudio_player import MiniAudioStreamPlayer
    from ghoshell_moss.host.speech.volcengine_tts.tts import VolcengineTTS

    return BaseTTSSpeech(
        player=player or MiniAudioStreamPlayer(),
        tts=tts or VolcengineTTS(),
        logger=logger,
    )
