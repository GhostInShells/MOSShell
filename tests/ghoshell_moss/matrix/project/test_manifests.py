"""Manifests 体系单元测试 — 从 Provider 开始验证 ScannedManifest + 扫描链路."""

from pathlib import Path

import pytest

from ghoshell_container import Provider, IoCContainer
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.project.manifests.base import ScannedManifest
from ghoshell_moss.project.manifests.providers import (
    ProviderManifest,
    search_provider_manifests,
)

# -- 项目中 stub manifests 的约定扫描路径
STUB_MANIFESTS_PROVIDERS = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests.providers'


# ==================================================================
# ScannedManifest — 通用包装器
# ==================================================================

class TestScannedManifest:
    def test_basic_fields(self, tmp_path):
        m = ScannedManifest[str](
            name='test.name',
            value='hello',
            found_at=tmp_path / 'found.py',
            import_path='some.pkg:attr',
            description='A test item',
            source='class X: pass',
            detail='more detail',
        )

        assert m.name() == 'test.name'
        assert m.value() == 'hello'
        assert m.found_at() == tmp_path / 'found.py'
        assert m.import_path() == 'some.pkg:attr'
        assert m.description() == 'A test item'
        assert m.source() == 'class X: pass'
        assert m.detail() == 'more detail'

    def test_error_manifest(self, tmp_path):
        """error= 构造的 manifest: is_error() → True, error() 返回异常, value() raise."""
        err = ValueError('bad')
        m = ScannedManifest[str](name='broken', error=err, found_at=tmp_path)

        assert m.is_error()
        assert m.error() is err
        assert 'bad' in str(m.error())
        with pytest.raises(ValueError, match='bad'):
            m.value()

    def test_update_container_default_is_noop(self, tmp_path):
        from ghoshell_container import Container
        m = ScannedManifest[str](name='x', value='hello', found_at=tmp_path)
        c = Container()
        m.update_container(c)  # 不应抛异常

    def test_is_manifest_instance(self, tmp_path):
        m = ScannedManifest[str](name='x', value='v', found_at=tmp_path)
        assert isinstance(m, Manifest)

    def test_normal_manifest_not_error(self, tmp_path):
        """正常 manifest: is_error() → False, error() → None, value() 正常返回."""
        m = ScannedManifest[str](name='x', value='ok', found_at=tmp_path)
        assert not m.is_error()
        assert m.error() is None
        assert m.value() == 'ok'


# ==================================================================
# ProviderManifest — update_container 行为
# ==================================================================

class TestProviderManifest:
    def test_is_scanned_manifest_and_isinstance_provider(self, tmp_path):
        """ProviderManifest 封装的 value 应该通过 isinstance(obj, Provider) 判定."""

        class PlainProvider(Provider[str]):
            def singleton(self) -> bool:
                return True

            def factory(self, con: IoCContainer) -> str:
                return 'ok'

        p = PlainProvider()
        assert isinstance(p, Provider)

        m = ProviderManifest(name='test', value=p, found_at=tmp_path)
        assert isinstance(m, Manifest)
        assert isinstance(m, ScannedManifest)
        assert isinstance(m.value(), Provider)

    def test_update_container_registers(self, tmp_path):
        from ghoshell_container import Container

        class MyContract:
            pass

        class PlainProvider(Provider[MyContract]):
            def singleton(self) -> bool:
                return True

            def factory(self, con: IoCContainer) -> MyContract:
                return MyContract()

        p = PlainProvider()
        m = ProviderManifest(name='test', value=p, found_at=tmp_path)
        c = Container()
        m.update_container(c)
        assert c.get(MyContract) is not None

    def test_update_container_skips_error_manifest(self, tmp_path):
        """error manifest 的 update_container 应该是 no-op 不抛异常."""
        from ghoshell_container import Container
        m = ProviderManifest(name='test', error=RuntimeError('boom'), found_at=tmp_path)
        c = Container()
        m.update_container(c)  # 不应抛异常


# ==================================================================
# search_provider_manifests — 用 stub 包验证真实扫描链路
# ==================================================================

class TestSearchProviderManifests:
    def test_finds_all_providers(self):
        results = list(search_provider_manifests(STUB_MANIFESTS_PROVIDERS))
        names = {m.name() for m in results}
        assert len(results) == 4
        for m in results:
            assert isinstance(m, ProviderManifest)

    def test_each_manifest_has_required_fields(self):
        for m in search_provider_manifests(STUB_MANIFESTS_PROVIDERS):
            assert m.name()
            assert m.found_at().exists()
            assert m.import_path() and ':' in m.import_path()
            assert isinstance(m.value(), Provider)

    def test_deduplicates(self):
        """同一包扫两遍不应重复."""
        results = list(search_provider_manifests(STUB_MANIFESTS_PROVIDERS))
        assert len(results) == 4  # stub 包 4 个 unique provider

    def test_empty_package(self):
        """不存在的包 → 空结果 (不抛异常)."""
        results = list(search_provider_manifests('ghoshell_moss.stubs.not_a_package'))
        assert results == []

    def test_skips_non_provider_objects(self):
        """stub 包的 configs/topics 等不含 Provider, 扫描应返回空."""
        results = list(search_provider_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.configs'
        ))
        assert results == []

    def test_each_manifest_update_container_works(self):
        from ghoshell_container import Container
        c = Container()
        for m in search_provider_manifests(STUB_MANIFESTS_PROVIDERS):
            m.update_container(c)  # 不应抛异常

    def test_yields_error_manifest_rather_than_silent_skip(self):
        """扫描出错的模块不再是静默跳过，而是返回 is_error() 为 True 的 Manifest."""
        results = list(search_provider_manifests(
            'ghoshell_moss.core.helpers.exception_pkg',
        ))
        assert len(results) == 1
        m = results[0]
        assert isinstance(m, ProviderManifest)
        assert m.is_error()
        err = m.error()
        assert isinstance(err, Exception)
        assert 'hello world' in str(err)
        assert 'exception_pkg' in m.name()
        # value() 应该 raise
        with pytest.raises(Exception):
            m.value()
