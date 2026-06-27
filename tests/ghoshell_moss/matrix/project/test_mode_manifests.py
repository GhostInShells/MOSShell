"""Mode manifests 单元测试 — channel + nuclei + ScannedModeManifests 集成."""

from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.blueprint.project import Manifest, ModeManifests
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.project.manifests.channels import (
    ChannelManifest,
    search_main_channel,
)
from ghoshell_moss.project.manifests.mode_impl import ScannedModeManifests

# -- stub 路径
STUB_MODE_ROOT = 'ghoshell_moss.stubs.workspace.modes.default.src.HOST'
STUB_MATRIX_PROVIDERS = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests.providers'


# ==================================================================
# ChannelManifest
# ==================================================================

class TestChannelManifest:
    def test_basic(self, tmp_path):
        from ghoshell_moss import new_default_shell_main_channel
        ch = new_default_shell_main_channel()
        m = ChannelManifest(
            name=ch.name(),
            value=ch,
            found_at=tmp_path,
            import_path='test:main',
            description=ch.description(),
        )
        assert m.name() == '__main__'
        assert isinstance(m.value(), PrimeChannel)

    def test_error_manifest(self, tmp_path):
        m = ChannelManifest(name='broken', error=ValueError('bad'), found_at=tmp_path)
        assert m.is_error()
        with pytest.raises(ValueError, match='bad'):
            m.value()


# ==================================================================
# search_main_channel
# ==================================================================

class TestSearchMainChannel:
    def test_finds_main_channel_from_stub_mode(self):
        """stub mode 的 channels.py 定义了 main = new_default_shell_main_channel()."""
        m = search_main_channel(f'{STUB_MODE_ROOT}.channels')
        assert isinstance(m, ChannelManifest)
        assert not m.is_error()
        assert m.name() == '__main__'
        assert isinstance(m.value(), PrimeChannel)

    def test_returns_error_when_no_main_found(self):
        m = search_main_channel(STUB_MATRIX_PROVIDERS)
        assert m.is_error()

    def test_package_not_exists_does_not_raise(self):
        """包不存在时返回 error manifest，不抛异常."""
        m = search_main_channel('nonexistent.package.channels')
        assert isinstance(m, ChannelManifest)
        assert m.is_error()


# ==================================================================
# ScannedModeManifests — 集成
# ==================================================================

class TestScannedModeManifests:
    def test_is_mode_manifests_instance(self):
        m = ScannedModeManifests(STUB_MODE_ROOT)
        assert isinstance(m, ModeManifests)

    def test_channel_found(self):
        m = ScannedModeManifests(STUB_MODE_ROOT)
        ch = m.channel()
        assert not ch.is_error()
        assert ch.name() == '__main__'

    def test_nuclei_empty_when_no_instances(self):
        """stub mode nuclei.py 只有 import 没有实例，返回空列表."""
        m = ScannedModeManifests(STUB_MODE_ROOT)
        results = list(m.nuclei())
        assert results == []

    def test_providers_inherited_from_matrix(self):
        """继承自 ScannedMatrixManifest 的 providers 和 signals 正常工作."""
        m = ScannedModeManifests(STUB_MODE_ROOT)
        providers = list(m.providers())
        assert len(providers) >= 1
        for p in providers:
            assert not p.is_error()

    def test_package_not_exists_is_tolerant(self):
        """不存在的 mode 包返回合法的 ModeManifests，Iterable 方法返回空列表."""
        m = ScannedModeManifests('nonexistent.mode.pkg')
        assert m.channel().is_error()
        assert list(m.nuclei()) == []
        assert list(m.resources()) == []
        assert list(m.providers()) == []
        assert list(m.configs()) == []
        assert list(m.topics()) == []
        assert list(m.signals()) == []
        assert m.parameters().is_error()
