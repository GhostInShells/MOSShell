"""
ZenohCellPresence — 单 cell 入网侧实现 (CellPresence ABC).

一个 Presence 实例治理一个 cell 的入网 zenoh 资源 (key 表见 _utils.py):
  cell_key                                → liveness token + queryable
  event_key                               → CellEvent publisher
  legacy_host_liveness_key (§ZZ-10 作废)   → host 跨域 liveness (仅 host, M10 清)

Presence 的承诺: 让本 cell 在网络上可被发现、可被查询、可提供 channel.
不观察别人 — 观察是 Mesh 的事 (§UU-7 拆分).

announce payload 只包含膜类型标签 (Cell.providing), 不含膜具体内容 —
channel meta 靠 duplex hub.proxy + refresh_metas 同步 (廉价).

生命周期:
  __aenter__ 触发首次 announce (declare queryable / liveness / event publisher).
  __aexit__ 触发 revoke (undeclare 全部资源, 幂等).
  运行时 payload 变化 (如 providing 添加 'channel') 通过 self._cell_presence.update()
  就地修改, queryable 回复函数每次 dump 现值; publish_event 广播 CellEvent 提示
  订阅侧 refetch.
"""
# -- §UU-7: 入网侧成本 O(1) 被动, 每个 cell 永远开.
#    debug 问责单一性: "别人看不见我" → 审讯本对象.
# -- §UU-8: accept/deny 归 Mesh, 不在这里. 本对象只管入网.
# -- Provider 归属: hub.provider(address) 返 ChannelProvider, provide_channel
#    只做副作用 (providing += 'channel' + publish event) + 返 provider.
#    provider.arun_until_closed 的 task 归 Matrix 起 (单根约束在 Matrix 侧).
# -- check_unique 不实现 (TT-2 已作废): 一致性问题让位给单写者原则 (run_node 咽喉
#    domain 档查重, host 档 flock). 网络层若发现冲突不 raise, 交由上层治理.

import asyncio
from dataclasses import dataclass

import zenoh
from typing_extensions import Self

from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub
from ghoshell_moss.core.blueprint.cell import (
    CellEvent,
    Cell,
    CellPresence,
)
from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider
from ghoshell_moss.matrix.networks._utils import CellsKeyspace, CellKeyExpr

import logging

__all__ = ['ZenohCellPresence']


@dataclass
class _ZenohHandles:
    """一次 announce 持有的 zenoh 资源, 用于 revoke 时释放."""
    liveness_token: zenoh.LivelinessToken
    queryable: zenoh.Queryable
    event_publisher: zenoh.Publisher
    host_token: zenoh.LivelinessToken | None = None

    def close(self, logger: logging.Logger) -> None:
        for attr in ('event_publisher', 'queryable', 'liveness_token', 'host_token'):
            resource = getattr(self, attr, None)
            if resource is None:
                continue
            try:
                resource.undeclare()
            except RuntimeError:
                # zenoh 已 undeclare / session 已关: 幂等吞掉.
                pass
            except Exception:
                logger.exception("undeclare %s failed", attr)


class ZenohCellPresence(CellPresence):
    """
    基于 zenoh 的单 cell 入网实现.

    构造时传入初始 Cell 与 hub. __aenter__ 触发首次 announce,
    __aexit__ 触发 revoke. 运行时 payload 变化通过 self._cell_presence.update()
    就地修改, queryable 每次回复读现值; 通过 publish_event 广播 CellEvent
    提示订阅侧 refetch.

    hub 由外部传入 (matrix 层拥有, 与 ZenohCellMesh 共享同一 hub) —
    避免每 cell 建独立 hub 的 O(N) 资源开销.
    """

    def __init__(
            self,
            *,
            session: zenoh.Session,
            logger: logging.Logger,
            keyspace: CellsKeyspace,
            scope: str,
            hub: ZenohChannelHub,
            presence: Cell,
    ):
        self._session = session
        self._logger = logger
        self._keyspace = keyspace
        self._scope = scope
        self._hub = hub
        self._cell_presence = presence

        # per-cell key 打包 — 一次预算, 后续只读.
        self._keys: CellKeyExpr = keyspace.per_cell(presence.address)
        # §ZZ-10 副路径 (deprecated) — 仅 host 在 announce 时使用.
        self._legacy_host_liveness_key: str = keyspace.legacy_host_liveness_key(
            scope, presence.address,
        )
        self._handles: _ZenohHandles | None = None

    # -- CellPresence ABC 实现 -----------------------------------------

    @property
    def this(self) -> Cell:
        return self._cell_presence

    async def provide_channel(self, channel: Channel) -> ChannelProvider:
        """
        声明本 cell 提供 channel 能力.

        1. 从 hub 拿 provider (hub 内部按 address dedup, 二次 provide raise).
        2. 副作用: providing += 'channel', update 时间戳, publish CellEvent
           提示订阅侧 refetch.
        3. 返回 provider — 不起 task. task 归调用方 (MatrixImpl 单根约束).

        :raise RuntimeError: 尚未 __aenter__ (无 announce 上下文).
        :raise RuntimeError: 同 address 已在本 hub 注册 provider (hub dedup).
        """
        if self._handles is None:
            raise RuntimeError(
                f"presence for {self._cell_presence.address} not entered; "
                f"call `async with presence: ...` before provide_channel"
            )

        provider = self._hub.provider(self._cell_presence.address)

        # 副作用: 更新 payload, 广播事件.
        if 'channel' not in self._cell_presence.providing:
            self._cell_presence.providing.append('channel')
        self._cell_presence.update()
        try:
            await self.publish_event('channel added', updated=True)
        except Exception:
            self._logger.exception(
                "publish 'channel added' event failed for %s",
                self._cell_presence.address,
            )
        return provider

    async def publish_event(
            self,
            content: str,
            *,
            updated: bool = True,
    ) -> None:
        if self._handles is None:
            raise RuntimeError(
                f"presence for {self._cell_presence.address} not entered; "
                f"call `async with presence: ...` before publish_event"
            )
        event = CellEvent(
            address=self._cell_presence.address,
            content=content,
            refetch=updated,
        )
        payload = event.model_dump_json().encode('utf-8')
        try:
            await asyncio.to_thread(self._handles.event_publisher.put, payload)
        except Exception:
            self._logger.exception(
                "event publish failed: address=%s content=%s",
                self._cell_presence.address, content,
            )

    async def __aenter__(self) -> Self:
        if self._handles is not None:
            # 幂等: 已 announce 直接返回, 不重装资源.
            return self
        self._cell_presence.update()
        await self._install_handles()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._handles is None:
            return
        handles = self._handles
        self._handles = None
        # close 是同步 zenoh 调用, 放到 to_thread 避免阻塞 event loop.
        await asyncio.to_thread(handles.close, self._logger)

    # -- 内部 ---------------------------------------------------------

    async def _install_handles(self) -> None:
        address = self._cell_presence.address
        cell_key = self._keys.cell_key
        event_key = self._keys.event_key

        def _on_query(query: zenoh.Query):
            # 每次远端 query 时 dump 当前 presence 快照 —
            # self._cell_presence 在 provide_channel 中被 in-place 更新,
            # dump 是原子的.
            try:
                payload = self._cell_presence.model_dump_json().encode('utf-8')
                query.reply(query.key_expr, payload)
            except Exception:
                self._logger.exception(
                    "presence queryable error for %s", address,
                )

        # queryable 必须先于 liveness token — 避免 subscriber 收到 PUT 时
        # queryable 尚未就位导致空回复 (老 ZenohCellNetwork 的 TOCTOU 教训).
        queryable = self._session.declare_queryable(cell_key, _on_query)
        # event publisher 必须落在 events 子命名空间 —
        # ZenohCellMesh 订阅 {cells_ns}/events/**, 老代码在 cell_key 上发布是历史 bug.
        event_publisher = self._session.declare_publisher(event_key)
        liveness_token = self._session.liveliness().declare_token(cell_key)

        host_token: zenoh.LivelinessToken | None = None
        if self._cell_presence.is_host:
            # §ZZ-10 副路径作废但暂不删.
            # 老版做的旁路 host 声明 (为了不依赖 address[0] 保留字机制) — 本轮
            # 承认 address[0]='host/' 保留字 + wildcard subscribe, 副路径语义
            # 已由主路径 (cells_ns 下的 host/xxx address) 承担. 保留代码以免破坏
            # 现有 ZenohLivenessListener 跨域消费者, M10 拉起 cross-scope host
            # discovery 时同步清理.
            host_token = self._session.liveliness().declare_token(
                self._legacy_host_liveness_key,
            )

        self._handles = _ZenohHandles(
            liveness_token=liveness_token,
            queryable=queryable,
            event_publisher=event_publisher,
            host_token=host_token,
        )
        self._logger.debug(
            "presence announced: address=%s is_host=%s membrane=%s",
            address, self._cell_presence.is_host, self._cell_presence.providing,
        )
