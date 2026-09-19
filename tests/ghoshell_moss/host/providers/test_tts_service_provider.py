"""TTSManagerConfig.validate — 缺 env 时 raise, 是 speech 降级 (NullSpeech) 的触发点."""

import pytest

from ghoshell_moss.host.providers.tts_service_provider import TTSManagerConfig


def test_validate_raises_on_missing_env():
    conf = TTSManagerConfig()
    with pytest.raises(ValueError):
        conf.validate()


def test_validate_passes_on_resolved_env(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_STREAM_TTS_APP", "app")
    monkeypatch.setenv("VOLCENGINE_STREAM_TTS_ACCESS_TOKEN", "token")
    conf = TTSManagerConfig().resolve()
    conf.validate()
