from typing_extensions import Self
from ghoshell_moss.core.blueprint.matrix import Cell
from ghoshell_common.helpers import uuid

__all__ = [
    'FractalCell',
    'FractalKeyExpressions',
    'FRACTAL_ADDRESS_PREFIX', 'FRACTAL_SESSION_SCOPE',
]

FRACTAL_SESSION_SCOPE = "fractal_session"
FRACTAL_ADDRESS_PREFIX = "moss_fractal"


class FractalCell(Cell):
    """
    type='fractal' 的 Cell。表示通过 fractal 协议连接到当前 Matrix 的外部节点。
    """

    def __init__(self, name: str, description: str = "", where: str = ''):
        self.name = name
        self.type = 'fractal'
        self.description = description or f"Fractal node: {name}"
        self.where = where
        self.accepted: bool = False
        self.connection_keys: dict[str, str] = {}
        self.uid = uuid()

    @property
    def address(self) -> str:
        return Cell.make_address('fractal', self.name)

    def is_alive(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, data: dict) -> None | Self:
        name = data.get('name')
        description = data.get('description', '')
        where = data.get('where', '')
        if not name:
            return None
        return cls(name, description, where)

    def to_detail_info(self) -> dict[str, str]:
        cell_dict = self.to_dict()
        cell_dict['uid'] = self.uid
        cell_dict['accepted'] = self.accepted
        cell_dict['connection_keys'] = self.connection_keys
        return cell_dict


class FractalKeyExpressions:
    """
    约定一个声明的标准实现.
    主要是为 zenoh 体系服务, 但实际上也适合 redis 或其它广播协议.
    """

    def __init__(
            self,
            hub_name: str,
            address_prefix: str | None = None,
    ):
        self.hub_name = hub_name.strip('/')
        self._address_prefix = address_prefix.strip('/') if address_prefix else FRACTAL_ADDRESS_PREFIX

    def manifest_key(self, node_name: str) -> str:
        prefix = self.manifests_prefix()
        return f"{prefix}/{node_name}"

    def manifests_prefix(self) -> str:
        return f"{self._address_prefix}/{self.hub_name}/manifests"

    def manifests_wildcard(self) -> str:
        return f"{self.manifests_prefix()}/**"

    def provider_cell_address_prefix(self):
        # 增加一个 moss mode name, 方便在同一个 IP 的实例上, 通过不同的 identity 监听同一个端口.
        return f"{self._address_prefix}/{self.hub_name}/providers"

    def provider_wildcard(self) -> str:
        return f"{self.provider_cell_address_prefix()}/**"

    def provider_cell_address(self, cell_name: str) -> str:
        cell_name = cell_name.strip('/')
        prefix = self.provider_cell_address_prefix()
        return "/".join([prefix, cell_name])
