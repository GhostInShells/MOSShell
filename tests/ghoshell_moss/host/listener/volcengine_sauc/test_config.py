"""VolcengineSaucConfig.validate — 缺 api_key 时 raise (listener 降级触发点)."""

import pytest

from ghoshell_moss.host.listener.volcengine_sauc import VolcengineSaucConfig


def test_validate_raises_on_missing_api_key():
    conf = VolcengineSaucConfig()
    with pytest.raises(ValueError):
        conf.validate()


def test_validate_passes_on_resolved_api_key(monkeypatch):
    monkeypatch.setenv("SEEDASR_API_KEY", "key")
    conf = VolcengineSaucConfig().resolve()
    conf.validate()
