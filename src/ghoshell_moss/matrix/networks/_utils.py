"""
matrix/networks/ 层的 key/path 表达 — 一处定义, presence 与 mesh 共用.

镜像 bridges/zenoh_bridge/_utils.py 的模式:
  {域}Keyspace  = 命名空间层键位 (顶层 prefix + 通配 + 前缀剥离)
  {域}KeyExpr   = 单实体键位打包 (per-address)

网络键结构 (在 tools.zenoh_helper.MatrixNamespace.cells_ns 之下):

  {cells_ns}                                顶层 cells 命名空间
  {cells_ns}/{address}                      单 cell 的 presence key
                                            (liveness token + queryable + refetch queryable 共用)
  {cells_ns}/{EVENTS_SEGMENT}               events 子命名空间
  {cells_ns}/{EVENTS_SEGMENT}/{address}     单 cell 的 CellEvent publisher key

通配 (subscriber 用):
  {cells_ns}/**                            liveness 订阅
                                            (会连带匹配到 events/..., 反解时用 address_from_cell_key 排除)
  {cells_ns}/{EVENTS_SEGMENT}/**           CellEvent 订阅

§ZZ-10 副路径 (作废但暂留兼容, M10 清):
  {hosts_ns}/{scope}/cells/liveness/{address}    host 跨域 liveness token
"""

from typing import ClassVar

from ghoshell_moss.core.blueprint.cell import CellAddress
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace

__all__ = ['CellsKeyspace', 'CellKeyExpr']


class CellsKeyspace:
    """cells 命名空间下的 key 打包 (namespace-scope).

    presence 与 mesh 共用一份 keyspace, 避免任一侧手写 f-string.
    adapter 构造一份, 传给 new_presence / new_watcher.
    """

    EVENTS_SEGMENT: ClassVar[str] = 'events'
    """events 子命名空间段. presence 发布 / mesh 订阅共用."""

    _LEGACY_HOST_LIVENESS_SEGMENTS: ClassVar[tuple[str, ...]] = ('cells', 'liveness')
    """§ZZ-10 副路径的段序列. 主路径 {cells_ns}/host/... 落地后作废, M10 前保留兼容."""

    def __init__(self, namespace: MatrixNamespace):
        self.namespace = namespace
        self.cells_ns = namespace.cells_ns

        self.events_ns = '/'.join([self.cells_ns, self.EVENTS_SEGMENT])

        self.cells_ns_prefix = self.cells_ns + '/'
        self.events_ns_prefix = self.events_ns + '/'

        # subscriber 通配.
        # cell_liveness_wildcard 会连带匹配到 events/..., 反解 address 时排除.
        self.cell_liveness_wildcard = self.cells_ns + '/**'
        self.events_wildcard = self.events_ns + '/**'

    # ---- per-cell key builders ---------------------------------------

    def cell_key(self, address: CellAddress) -> str:
        """单 cell 的 presence key — liveness token / queryable / refetch 共用."""
        return self.cells_ns_prefix + address

    def event_key(self, address: CellAddress) -> str:
        """单 cell 的 CellEvent publisher key (对齐 events subscriber 通配)."""
        return self.events_ns_prefix + address

    def per_cell(self, address: CellAddress) -> 'CellKeyExpr':
        """打包某个 cell 一次入网所需的 keys, presence 侧一次构造多次读取."""
        return CellKeyExpr(self, address)

    # ---- key → address 反解 ------------------------------------------

    def address_from_cell_key(self, key: str) -> CellAddress | None:
        """从 {cells_ns}/{address} 剥前缀; events/... 命中返回 None (liveness 通配会连带匹配)."""
        if not key.startswith(self.cells_ns_prefix):
            return None
        rest = key[len(self.cells_ns_prefix):]
        if rest.startswith(self.EVENTS_SEGMENT + '/'):
            return None
        return rest or None

    def address_from_event_key(self, key: str) -> CellAddress | None:
        """从 {cells_ns}/{EVENTS_SEGMENT}/{address} 剥前缀."""
        if not key.startswith(self.events_ns_prefix):
            return None
        rest = key[len(self.events_ns_prefix):]
        return rest or None

    # ---- §ZZ-10 副路径 (deprecated) ----------------------------------

    def legacy_host_liveness_key(self, scope: str, address: CellAddress) -> str:
        """§ZZ-10 副路径 hosts_ns/{scope}/cells/liveness/{address}.

        主路径 (cells_ns 下的 host/{name}/{uid} address + 通配订阅) 承接后本副路径作废,
        保留以免破坏 M10 前的 ZenohLivenessListener 跨域消费者.
        """
        return '/'.join([
            self.namespace.hosts_ns, scope,
            *self._LEGACY_HOST_LIVENESS_SEGMENTS,
            address,
        ])


class CellKeyExpr:
    """
    单 cell 一次入网所需的键位打包 (per-address).

    presence 构造时预算一次, 后续只读 — 不再拼字符串.
    """

    def __init__(self, keyspace: CellsKeyspace, address: CellAddress):
        self.address = address
        self.cell_key: str = keyspace.cell_key(address)
        """liveness token + queryable key."""
        self.event_key: str = keyspace.event_key(address)
        """event publisher key (发送到 events subscriber 通配)."""
