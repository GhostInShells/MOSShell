"""Zenoh transport config — 按 CellType 分发的类型化配置.

host (peer):  监听 + multicast 发现, 网络入口点.
worker (client):  直连 host, 不监听.
"""

from typing import ClassVar, TYPE_CHECKING

from pydantic import BaseModel, Field

from ghoshell_moss.core.blueprint.cell import CellType
from ghoshell_moss.core.blueprint.project import NetworkConfig, NetworkMetadata

if TYPE_CHECKING:
    import zenoh

__all__ = [
    'ZenohNodeConfig', 'ZenohNetworkConfig', 'ZENOH_DRIVER',
    'create_zenoh_session_from_metadata',
]

ZENOH_DRIVER = 'zenoh'


class ZenohNodeConfig(BaseModel):
    """单个 cell type 的 zenoh 连接参数."""

    connect: str = Field(
        default='',
        description="直连 endpoint, 如 tcp/192.168.1.10:20380. 为空时仅 listen",
    )
    listen: str = Field(
        default='',
        description="监听地址. tcp/0.0.0.0:0 为随机端口",
    )
    multicast: bool = Field(
        default=False,
        description="是否开启 multicast scouting",
    )

    # 透传给 zenoh.Config 的剩余字段
    extra: dict = Field(
        default_factory=dict,
        description="透传 zenoh.Config 扩展配置",
    )


class ZenohNetworkConfig(NetworkConfig):
    """zenoh driver 的网络配置 — 按 cell type 组织."""

    DRIVER: ClassVar[str] = ZENOH_DRIVER

    host: ZenohNodeConfig = Field(
        default_factory=lambda: ZenohNodeConfig(listen='tcp/127.0.0.1:20380'),
        description="host 节点: 默认 localhost 监听, LAN 部署改为 tcp/0.0.0.0:20380 + multicast",
    )
    worker: ZenohNodeConfig = Field(
        default_factory=lambda: ZenohNodeConfig(connect='tcp/127.0.0.1:20380'),
        description="worker 节点: 默认直连本地 host",
    )
    cell_type_configs: dict[str, ZenohNodeConfig] = Field(
        default_factory=dict,
        description="根据其它 type 做的映射",
    )

    @classmethod
    def driver_name(cls) -> str:
        return cls.DRIVER

    def for_cell(self, cell_type: CellType | str) -> ZenohNodeConfig:
        """取对应 cell type 的节点配置."""
        cell_type = cell_type.value if isinstance(cell_type, CellType) else str(cell_type)
        if cell_type == CellType.host:
            return self.host
        elif cell_type in self.cell_type_configs:
            return self.cell_type_configs[cell_type]
        else:
            return self.worker


def create_zenoh_session_from_metadata(metadata: NetworkMetadata, cell_type: CellType | str) -> 'zenoh.Session | None':
    config = ZenohNetworkConfig.from_metadata(metadata)
    if config is None:
        return None
    node_config = config.for_cell(cell_type)
    return _create_zenoh_session(node_config)


def _create_zenoh_session(config: ZenohNodeConfig) -> 'zenoh.Session':
    from ghoshell_moss.depends import depend_zenoh
    depend_zenoh()
    import zenoh

    zconfig = zenoh.Config()
    if config.listen:
        zconfig.insert_json5('listen/endpoints', f'["{config.listen}"]')
    if config.connect:
        zconfig.insert_json5('connect/endpoints', f'["{config.connect}"]')
    if config.multicast:
        zconfig.insert_json5('scouting/multicast/enabled', 'true')
    for k, v in config.extra.items():
        zconfig.insert_json5(k, v)
    return zenoh.open(zconfig)
