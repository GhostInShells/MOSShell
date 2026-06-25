"""ZenohNetworkConfig 单元测试 — 配置默认值 / dispatch / 序列化往返."""

import pytest
from typing import ClassVar

from ghoshell_moss.core.blueprint.cell import CellType
from ghoshell_moss.core.blueprint.project import NetworkConfig, NetworkMetadata
from ghoshell_moss.matrix.networks.zenoh_network import (
    ZenohNodeConfig,
    ZenohNetworkConfig,
    ZENOH_DRIVER,
    create_zenoh_session_from_metadata,
)


class TestZenohNetworkDefaults:
    def test_host_defaults(self):
        cfg = ZenohNetworkConfig()
        assert cfg.host.listen == 'tcp/127.0.0.1:20380'
        assert cfg.host.connect == ''
        assert cfg.host.multicast is False

    def test_worker_defaults(self):
        cfg = ZenohNetworkConfig()
        assert cfg.worker.connect == 'tcp/127.0.0.1:20380'
        assert cfg.worker.listen == ''
        assert cfg.worker.multicast is False

    def test_driver_name(self):
        assert ZenohNetworkConfig.driver_name() == ZENOH_DRIVER


class TestForCell:
    def test_host_type(self):
        cfg = ZenohNetworkConfig()
        node = cfg.for_cell(CellType.host)
        assert node == cfg.host

    def test_worker_type(self):
        cfg = ZenohNetworkConfig()
        node = cfg.for_cell(CellType.worker)
        assert node == cfg.worker

    def test_unknown_type_falls_back_to_worker(self):
        cfg = ZenohNetworkConfig()
        assert cfg.for_cell('robot') == cfg.worker

    def test_custom_type_in_configs(self):
        cfg = ZenohNetworkConfig(
            cell_type_configs={
                'robot': ZenohNodeConfig(connect='tcp/10.0.0.5:20380'),
            },
        )
        node = cfg.for_cell('robot')
        assert node.connect == 'tcp/10.0.0.5:20380'

    def test_str_cell_type(self):
        cfg = ZenohNetworkConfig()
        assert cfg.for_cell('host') == cfg.host


class TestNetworkConfigRoundtrip:
    """NetworkConfig ↔ NetworkMetadata 序列化往返."""

    def test_to_metadata(self):
        cfg = ZenohNetworkConfig(
            host=ZenohNodeConfig(listen='tcp/0.0.0.0:20380', multicast=True),
            worker=ZenohNodeConfig(connect='tcp/192.168.1.10:20380'),
        )
        meta = cfg.to_metadata(name='lab', description='lab network')
        assert meta.name == 'lab'
        assert meta.description == 'lab network'
        assert meta.driver == ZENOH_DRIVER
        assert meta.config['host']['listen'] == 'tcp/0.0.0.0:20380'
        assert meta.config['worker']['connect'] == 'tcp/192.168.1.10:20380'

    def test_from_metadata_roundtrip(self):
        original = ZenohNetworkConfig()
        meta = original.to_metadata(name='default')
        restored = ZenohNetworkConfig.from_metadata(meta)
        assert restored is not None
        assert restored.host.listen == original.host.listen
        assert restored.worker.connect == original.worker.connect

    def test_from_metadata_driver_mismatch(self):
        class FakeConfig(NetworkConfig):
            DRIVER: ClassVar[str] = 'fake'

            @classmethod
            def driver_name(cls) -> str:
                return cls.DRIVER
        meta = NetworkMetadata(name='x', driver='nope', scope='s')
        assert FakeConfig.from_metadata(meta) is None
        assert ZenohNetworkConfig.from_metadata(meta) is None


class TestCreateSessionFromMetadata:
    def test_driver_mismatch_returns_none(self):
        meta = NetworkMetadata(name='x', driver='mqtt', scope='s', config={})
        assert create_zenoh_session_from_metadata(meta, CellType.host) is None
