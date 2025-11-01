from ghoshell_moss.speech.mock import MockSpeech

from ghoshell_moss.speech.stream_tts_speech import TTSSpeech, TTSSpeechStream


def make_baseline_tts_speech() -> TTSSpeech:
    """
    基线示例.
    """
    from ghoshell_moss.speech.volcengine_tts import VolcengineTTS
    from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer

    return TTSSpeech(
        player=PyAudioStreamPlayer(),
        tts=VolcengineTTS(),
    )
