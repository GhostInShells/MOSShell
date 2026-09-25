"""NullSpeech 表面契约 — 不可用信号经 played_text 浮出, 供 say/content 命令返回."""

from ghoshell_moss.core.speech.null import NULL_SPEECH_PLAYED_TEXT, NullSpeech


def test_null_speech_played_text_is_unavailable():
    stream = NullSpeech().new_segment()
    assert stream.played_text() == NULL_SPEECH_PLAYED_TEXT
    assert NULL_SPEECH_PLAYED_TEXT == "speech 注册不可用"
