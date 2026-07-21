"""Manifests 体系单元测试 — 从 Provider 开始验证 ScannedManifest + 扫描链路."""

import inspect
import sys
from pathlib import Path

import pytest

from ghoshell_container import Provider, IoCContainer
from ghoshell_moss.contracts.configs import ConfigType, ConfigSchema
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.blueprint.mindflow import SignalMeta, SignalSchema
from ghoshell_moss.project.manifests.base import ScannedManifest
from ghoshell_moss.project.manifests.providers import (
    ProviderManifest,
    search_provider_manifests,
)
from ghoshell_moss.project.manifests.configs import (
    ConfigManifest,
    search_config_manifests,
)
from ghoshell_moss.project.manifests.signals import (
    SignalManifest,
    search_signal_manifests,
)
from ghoshell_moss.project.manifests.topics import (
    TopicManifest,
    search_topic_manifests,
)
from ghoshell_moss.project.manifests.parameters import (
    ParameterManifest,
    search_parameter_manifests,
)
from ghoshell_moss.project.manifests.nuclei import (
    NucleusManifest,
    search_nucleus_manifests,
)
from ghoshell_moss.project.manifests.resources import (
    ResourceManifest,
    search_resource_manifests,
)
from ghoshell_moss.project.manifests.impl import ScannedMatrixManifest
from ghoshell_moss.core.concepts.topic import TopicSchema
from ghoshell_moss.core.blueprint.parameter import ParameterSchema
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta, SignalSchema
from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_container import Provider

# -- 项目中 stub manifests 的约定扫描路径
STUB_MANIFESTS_PROVIDERS = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests.providers'
STUB_MANIFESTS_CONFIGS = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests.configs'
STUB_MANIFESTS_SIGNALS = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests.signals'
STUB_MANIFESTS_TOPICS = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests.topics'
STUB_MANIFESTS_PARAMETERS = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests.parameters'
STUB_MANIFESTS_NUCLEI = 'ghoshell_moss.stubs.workspace.modes.default.src.HOST.nuclei'
STUB_MANIFESTS_ROOT = 'ghoshell_moss.stubs.workspace.src.MOSS.manifests'


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


# ==================================================================
# ConfigManifest
# ==================================================================

class TestConfigManifest:
    def test_basic(self, tmp_path):
        class MyConfig(ConfigType):
            key: str = 'default-val'

            @classmethod
            def conf_name(cls) -> str:
                return 'my_config'

        instance = MyConfig(key='custom')
        m = ConfigManifest(
            name=instance.conf_name(),
            value=instance,
            found_at=tmp_path,
            import_path='test:my_instance',
            description='test config',
        )

        assert m.name() == 'my_config'
        assert m.value().key == 'custom'
        s = m.schema()
        assert isinstance(s, ConfigSchema)
        assert s.name == 'my_config'
        assert 'json_schema' in s.model_dump()

    def test_error_manifest(self, tmp_path):
        m = ConfigManifest(
            name='broken',
            error=ValueError('bad config'),
            found_at=tmp_path,
        )
        assert m.is_error()
        assert isinstance(m.error(), ValueError)
        with pytest.raises(ValueError, match='bad config'):
            m.value()


# ==================================================================
# search_config_manifests — 扫描
# ==================================================================

class TestSearchConfigManifests:
    def test_finds_config_instances(self):
        """stub configs.py 里是 LLMConfig() 实例."""
        results = list(search_config_manifests(STUB_MANIFESTS_CONFIGS))
        assert len(results) == 1
        m = results[0]
        assert isinstance(m, ConfigManifest)
        assert m.name() == 'llms'
        assert isinstance(m.value(), ConfigType)
        assert m.schema().name == 'llms'

    def test_yields_error_manifest(self):
        results = list(search_config_manifests(
            'ghoshell_moss.core.helpers.exception_pkg',
        ))
        assert len(results) == 1
        m = results[0]
        assert m.is_error()

    def test_skips_non_config_objects(self):
        """stub topics.py 不含 ConfigType 实例."""
        results = list(search_config_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.topics'
        ))
        assert results == []

    def test_empty_package(self):
        results = list(search_config_manifests('ghoshell_moss.stubs.not_a_package'))
        assert results == []


# ==================================================================
# SignalManifest
# ==================================================================

class TestSignalManifest:
    def test_from_signal_meta_class(self, tmp_path):
        """SignalMeta 子类 → to_signal_schema() → value 是 SignalSchema."""

        class TestSignal(SignalMeta):
            """A test signal."""

            @classmethod
            def signal_name(cls) -> str:
                return 'test.signal'

            @classmethod
            def priority(cls) -> int:
                return 10

        m = SignalManifest(
            name=TestSignal.signal_name(),
            value=TestSignal.to_signal_schema(),
            found_at=tmp_path,
            import_path='test:TestSignal',
            description=TestSignal.__doc__ or '',
            source=inspect.getsource(TestSignal),
        )

        assert m.name() == 'test.signal'
        assert isinstance(m.value(), SignalSchema)
        assert m.value().name == 'test.signal'
        assert m.value().default_priority == 10
        assert 'class TestSignal' in m.source()

    def test_from_signal_schema_instance(self, tmp_path):
        schema = SignalSchema(
            name='custom.signal',
            description='Custom signal',
            default_priority=5,
            metadata_schema={},
        )
        m = SignalManifest(
            name=schema.name,
            value=schema,
            found_at=tmp_path,
            import_path='test:custom_schema',
            description=schema.description,
            source='',
        )
        assert m.name() == 'custom.signal'
        assert m.source() == ''

    def test_error_manifest(self, tmp_path):
        m = SignalManifest(
            name='broken',
            error=ValueError('bad signal'),
            found_at=tmp_path,
        )
        assert m.is_error()
        with pytest.raises(ValueError, match='bad signal'):
            m.value()


# ==================================================================
# search_signal_manifests — 扫描
# ==================================================================

class TestSearchSignalManifests:
    def test_finds_all_signal_classes(self):
        """stub signals.py 有 > 0 个 SignalMeta 子类."""
        results = list(search_signal_manifests(STUB_MANIFESTS_SIGNALS))
        assert len(results) > 0
        for m in results:
            assert isinstance(m, SignalManifest)
            assert isinstance(m.value(), SignalSchema)
            assert m.name()
            assert m.source()  # class-based 有源码

    def test_yields_error_manifest(self):
        results = list(search_signal_manifests(
            'ghoshell_moss.core.helpers.exception_pkg',
        ))
        assert len(results) == 1
        m = results[0]
        assert m.is_error()

    def test_skips_non_signal_objects(self):
        results = list(search_signal_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.topics'
        ))
        assert results == []

    def test_finds_signal_schema_instance(self, tmp_path):
        import sys
        root = tmp_path / 'src'
        root.mkdir()
        pkg = root / 'sig_pkg'
        pkg.mkdir()
        (pkg / '__init__.py').write_text('\n'.join([
            'from ghoshell_moss.core.blueprint.mindflow import SignalSchema',
            'my_signal = SignalSchema(',
            '    name="my.signal",',
            '    description="test",',
            '    default_priority=3,',
            '    metadata_schema={},',
            ')',
        ]))
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
        try:
            results = list(search_signal_manifests('sig_pkg'))
            assert len(results) == 1
            m = results[0]
            assert isinstance(m, SignalManifest)
            assert m.name() == 'my.signal'
            assert m.value().default_priority == 3
            assert m.source() == ''  # instance 无源码
        finally:
            if s in sys.path:
                sys.path.remove(s)

    def test_empty_package(self):
        results = list(search_signal_manifests('ghoshell_moss.stubs.not_a_package'))
        assert results == []


# ==================================================================
# TopicManifest — TopicModel 子类 或 TopicSchema 实例
# ==================================================================

class TestTopicManifest:
    def test_from_topic_schema_instance(self, tmp_path):
        schema = TopicSchema(
            topic_name='test.topic',
            topic_type='test',
            description='Test topic',
        )
        m = TopicManifest(
            name=schema.topic_name,
            value=schema,
            found_at=tmp_path,
            import_path='test:my_topic',
            description=schema.description,
        )
        assert m.name() == 'test.topic'
        assert m.value() is schema

    def test_error_manifest(self, tmp_path):
        m = TopicManifest(name='broken', error=ValueError('bad'), found_at=tmp_path)
        assert m.is_error()
        with pytest.raises(ValueError, match='bad'):
            m.value()


# ==================================================================
# search_topic_manifests — 扫描
# ==================================================================

class TestSearchTopicManifests:
    def test_finds_all_topic_classes(self):
        """stub topics.py 有 3 个 TopicModel 子类."""
        results = list(search_topic_manifests(STUB_MANIFESTS_TOPICS))
        assert len(results) == 3
        for m in results:
            assert isinstance(m, TopicManifest)
            assert isinstance(m.value(), TopicSchema)
            assert m.name()

    def test_yields_error_manifest(self):
        results = list(search_topic_manifests(
            'ghoshell_moss.core.helpers.exception_pkg',
        ))
        assert len(results) == 1
        assert results[0].is_error()

    def test_skips_non_topic_objects(self):
        results = list(search_topic_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.providers'
        ))
        assert results == []

    def test_empty_package(self):
        results = list(search_topic_manifests('ghoshell_moss.stubs.not_a_package'))
        assert results == []


# ==================================================================
# ParameterManifest — ParameterModel 子类 或 ParameterSchema 实例
# ==================================================================

class TestParameterManifest:
    def test_from_parameter_schema_instance(self, tmp_path):
        schema = ParameterSchema(
            name='test.param',
            description='Test parameter',
            json_schema={},
            default={},
        )
        m = ParameterManifest(
            name=schema.name,
            value=schema,
            found_at=tmp_path,
            import_path='test:my_param',
            description=schema.description,
        )
        assert m.name() == 'test.param'
        assert m.value() is schema

    def test_error_manifest(self, tmp_path):
        m = ParameterManifest(name='broken', error=ValueError('bad'), found_at=tmp_path)
        assert m.is_error()
        with pytest.raises(ValueError, match='bad'):
            m.value()


# ==================================================================
# search_parameter_manifests — 扫描 (单值返回)
# ==================================================================

class TestSearchParameterManifests:
    def test_finds_parameter_class(self):
        """stub parameters.py 有 ExampleParameter (ParameterModel 子类)."""
        m = search_parameter_manifests(STUB_MANIFESTS_PARAMETERS)
        assert isinstance(m, ParameterManifest)
        assert not m.is_error()
        assert m.name() == 'example'
        assert isinstance(m.value(), ParameterSchema)

    def test_returns_error_when_none_found(self):
        m = search_parameter_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.providers'
        )
        assert isinstance(m, ParameterManifest)
        assert m.is_error()

    def test_returns_error_for_empty_package(self):
        m = search_parameter_manifests('ghoshell_moss.stubs.not_a_package')
        assert m.is_error()


# ==================================================================
# NucleusManifest — NucleusMeta 实例
# ==================================================================

class TestNucleusManifest:
    def test_from_nucleus_meta_instance(self, tmp_path):
        class StubNucleus(NucleusMeta):
            def name(self) -> str:
                return 'stub.nucleus'

            def description(self) -> str:
                return 'A stub nucleus'

            def signals(self):
                return []

            def factory(self, container):
                raise NotImplementedError

        instance = StubNucleus()
        m = NucleusManifest(
            name=instance.name(),
            value=instance,
            found_at=tmp_path,
            import_path='test:stub',
            description=instance.description(),
        )
        assert m.name() == 'stub.nucleus'
        assert m.value() is instance

    def test_error_manifest(self, tmp_path):
        m = NucleusManifest(name='broken', error=ValueError('bad'), found_at=tmp_path)
        assert m.is_error()


# ==================================================================
# search_nucleus_manifests — 扫描
# ==================================================================

class TestSearchNucleusManifests:
    def test_finds_nucleus_instances(self):
        """host stubs nuclei.py 有 ExampleNucleusMeta 实例."""
        results = list(search_nucleus_manifests(STUB_MANIFESTS_NUCLEI))
        assert len(results) >= 1
        for m in results:
            assert isinstance(m, NucleusManifest)
            assert isinstance(m.value(), NucleusMeta)

    def test_empty_for_package_without_nuclei(self):
        results = list(search_nucleus_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.providers'
        ))
        assert results == []

    def test_empty_for_missing_package(self):
        results = list(search_nucleus_manifests('ghoshell_moss.stubs.not_a_package'))
        assert results == []


# ==================================================================
# search_resource_manifests — 扫描
# ==================================================================

class TestSearchResourceManifests:
    def test_finds_resource_instances(self):
        """stub resources.py 有 LocalImageResourceMeta 实例."""
        results = list(search_resource_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.resources'
        ))
        assert len(results) >= 1
        for m in results:
            assert isinstance(m, ResourceManifest)
            assert not m.is_error()

    def test_empty_for_package_without_resources(self):
        results = list(search_resource_manifests(
            'ghoshell_moss.stubs.workspace.src.MOSS.manifests.providers'
        ))
        assert results == []

    def test_empty_for_missing_package(self):
        results = list(search_resource_manifests('ghoshell_moss.stubs.not_a_package'))
        assert results == []


# ==================================================================
# ScannedMatrixManifest — 集成
# ==================================================================

class TestScannedMatrixManifest:
    def test_all_stub_manifests_have_values(self):
        """stub manifests 根包扫描: 每个类别都有值，无异常."""
        m = ScannedMatrixManifest(STUB_MANIFESTS_ROOT)

        providers = list(m.providers())
        assert len(providers) >= 1
        for p in providers:
            assert not p.is_error()
            assert isinstance(p.value(), Provider)

        configs = list(m.configs())
        assert len(configs) >= 1
        for c in configs:
            assert not c.is_error()
            assert isinstance(c.value(), ConfigType)

        topics = list(m.topics())
        assert len(topics) >= 1
        for t in topics:
            assert not t.is_error()
            assert isinstance(t.value(), TopicSchema)

        signals = list(m.signals())
        assert len(signals) >= 1
        for s in signals:
            assert not s.is_error()
            assert isinstance(s.value(), SignalSchema)

        params = m.parameters()
        assert not params.is_error()
        assert isinstance(params.value(), ParameterSchema)

        resources = list(m.resources())
        assert len(resources) >= 1
        for r in resources:
            assert not r.is_error()
