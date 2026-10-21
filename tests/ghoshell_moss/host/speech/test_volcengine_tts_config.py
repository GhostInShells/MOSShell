"""The global default speech rate is the floor, not a hard pin.

The engine's own default (0) is noticeably slower than how a human speaks, so the
config carries a global ``speech_rate`` a session falls back to when the speaker
did not set one. A per-speaker value — including an explicit 0 — must win,
otherwise a tone tuned once could not be brought back to normal.
"""
from ghoshell_moss.host.speech.volcengine_tts.config import VolcengineTTSConf


def _rate(conf: VolcengineTTSConf, *rates: int) -> int | None:
    speaker = conf.default_speaker_conf()
    if rates:
        speaker.voice.speech_rate = rates[0]
    return conf.to_session(speaker).req_params.audio_params.speech_rate


def test_speaker_without_a_rate_inherits_the_global_default():
    conf = VolcengineTTSConf(speech_rate=-10)
    assert _rate(conf) == -10


def test_explicit_speaker_rate_wins_over_the_global_default():
    conf = VolcengineTTSConf(speech_rate=20)
    assert _rate(conf, 0) == 0
    assert _rate(conf, 50) == 50
