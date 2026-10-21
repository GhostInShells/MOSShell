"""TTSManagerConfig.validate_configuration — 无可用凭据时 raise, 是 speech 降级 (NullSpeech) 的触发点."""

import pytest

from ghoshell_moss.host.providers.tts_service_provider import TTSManagerConfig

_CREDENTIAL_ENV = (
    "SEED_API_KEY",
    "VOLCENGINE_STREAM_TTS_APP",
    "VOLCENGINE_STREAM_TTS_ACCESS_TOKEN",
)


def _clear_credentials(monkeypatch):
    for name in _CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)


def test_raises_when_no_credential(monkeypatch):
    _clear_credentials(monkeypatch)
    conf = TTSManagerConfig().resolve()
    with pytest.raises(ValueError):
        conf.validate_configuration()


def test_passes_on_seed_api_key(monkeypatch):
    _clear_credentials(monkeypatch)
    monkeypatch.setenv("SEED_API_KEY", "key")
    conf = TTSManagerConfig().resolve()
    conf.validate_configuration()


def test_passes_on_legacy_credentials(monkeypatch):
    _clear_credentials(monkeypatch)
    monkeypatch.setenv("VOLCENGINE_STREAM_TTS_APP", "app")
    monkeypatch.setenv("VOLCENGINE_STREAM_TTS_ACCESS_TOKEN", "token")
    conf = TTSManagerConfig().resolve()
    conf.validate_configuration()


def test_raises_on_incomplete_legacy_credentials(monkeypatch):
    _clear_credentials(monkeypatch)
    monkeypatch.setenv("VOLCENGINE_STREAM_TTS_APP", "app")
    conf = TTSManagerConfig().resolve()
    with pytest.raises(ValueError):
        conf.validate_configuration()
