"""
ZenohPresence — 单 cell 入网侧实现 (Presence ABC).

一个 Presence 实例治理一个 cell 的入网 zenoh 资源:
  {cells_ns}/{address}                                → liveness token + queryable
  {cells_ns}/events/{address}                         → CellEvent publisher
  {hosts_ns}/{scope}/cells/liveness/{address}         → host 跨域 liveness (仅 host)

Presence 的承诺: 让本 cell 在网络上可被发现、可被查询、可提供 channel.
不观察别人 — 观察是 Watcher 的事 (§UU-7 拆分).

announce payload 只包含膜类型标签 (CellPresence.membrane), 不含膜具体内容 —
channel meta 靠 duplex hub.proxy + refresh_metas 同步 (廉价).
"""
# -- §UU-7: 入网侧成本 O(1) 被动, 每个 cell 永远开.
#    debug 问责单一性: "别人看不见我" → 审讯本对象.
# -- §UU-8: accept/deny 归 Watcher, 不在这里. 本对象只管入网.
# -- check_unique 不实现 (TT-2 已作废): 一致性问题让位给单写者原则 (run_cell 咽喉
#    domain 档查重, host 档 flock). 网络层若发现冲突不 raise, 交由上层治理.

import asyncio
import time
from dataclasses import dataclass

import zenoh
from typing_extensions import Self

from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub
from ghoshell_moss.core.blueprint.cell import (
    CellAddress,
    CellEvent,
    CellPresence,
    Presence,
)
from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace

import logging

__all__ = ['ZenohPresence']


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


class ZenohPresence(Presence):
    """
    基于 zenoh 的单 cell 入网实现.

    构造时传入初始 CellPresence 与 hub. __aenter__ 触发首次 announce,
    __aexit__ 触发 revoke. announce() 可在运行中再次调用更新 payload
    (address 不变则复用资源, 只 touch queryable; address 变了先 revoke 再重建).

    hub 由外部传入 (matrix 层拥有, 与 ZenohWatcher 共享同一 hub) —
    避免每 cell 建独立 hub 的 O(N) 资源开销.
    """

    def __init__(
            self,
            *,
            session: zenoh.Session,
            logger: logging.Logger,
            namespace: MatrixNamespace,
            scope: str,
            hub: ZenohChannelHub,
            presence: CellPresence,
    ):
        self._session = session
        self._logger = logger
        self._ns = namespace
        self._scope = scope
        self._hub = hub
        self._presence = presence

        self._cells_events_ns = f"{self._ns.cells_ns}/events"

        self._handles: _ZenohHandles | None = None
        self._closed = False

    # -- key 构造 ------------------------------------------------------

    def _cell_key(self, address: CellAddress) -> str:
        # {cells_ns}/{address}: liveness token + queryable 共用 key.
        return f"{self._ns.cells_ns}/{address}"

    def _event_key(self, address: CellAddress) -> str:
        # {cells_ns}/events/{address}: CellEvent publisher key.
        # 与 cell_key 分层 (events 段), 避免 Watcher 的 cell_liveness_wildcard
        # 通配到事件 key.
        return f"{self._cells_events_ns}/{address}"

    def _host_liveness_key(self, address: CellAddress) -> str:
        # {hosts_ns}/{scope}/cells/liveness/{address}: host 跨域宣告.
        return f"{self._ns.hosts_ns}/{self._scope}/cells/liveness/{address}"

    # -- ABC 实现 ------------------------------------------------------

    @property
    def this(self) -> CellPresence:
        return self._presence

    async def announce(self, presence: CellPresence) -> None:
        if self._closed:
            raise RuntimeError("ZenohPresence is closed")

        old_address = self._presence.address if self._handles else None
        self._presence = presence
        self._presence.updated = time.time()

        # 首次 announce 或 address 未变: 装好资源即返回.
        if self._handles is None:
            await self._install_handles()
            return
        if presence.address == old_address:
            # queryable 回调读的是 self._presence, 无需重建资源.
            return

        # address 变了: revoke 旧的, 装新的.
        self._handles.close(self._logger)
        self._handles = None
        await self._install_handles()

    async def revoke(self) -> None:
        if self._handles is None:
            return
        handles = self._handles
        self._handles = None
        # close 是同步 zenoh 调用, 放到 to_thread 避免阻塞 event loop.
        await asyncio.to_thread(handles.close, self._logger)

    async def provide(self, channel: Channel) -> ChannelProvider:
        # 膜承诺: cell 必须 provide channel (§UU-2). 无 provide 的 cell 在模型
        # 能力空间里不存在, 应改用 Subprocesses 治理.
        #
        # 副作用:
        #   1. hub.provider(address) 拿 bare provider — matrix 层负责
        #      asyncio.create_task(provider.arun_until_closed(channel)) 启动.
        #   2. 'channel' 加到 presence.membrane (v1 唯一膜类型).
        #   3. touch presence.updated, 让 queryable 下次查询返回新时间戳.
        #   4. 发 refetch=True 的 CellEvent 通知网络: "我这里膜清单变了".
        provider = self._hub.provider(self._presence.address)
        if 'channel' not in self._presence.membrane:
            self._presence.membrane.append('channel')
        self._presence.updated = time.time()
        # publish 是尽力而为, 首次 provide 时若尚未 announce 则跳过 event 广播.
        if self._handles is not None:
            try:
                await self.publish_event(
                    'channel added', refetch=True,
                )
            except Exception:
                self._logger.exception(
                    "publish 'channel added' event failed for %s",
                    self._presence.address,
                )
        return provider

    async def publish_event(
            self,
            content: str,
            *,
            refetch: bool = True,
    ) -> None:
        if self._handles is None:
            raise LookupError(
                f"presence for {self._presence.address} not announced yet"
            )
        event = CellEvent(
            address=self._presence.address,
            content=content,
            timestamp=time.time(),
            refetch=refetch,
        )
        payload = event.model_dump_json().encode('utf-8')
        try:
            await asyncio.to_thread(self._handles.event_publisher.put, payload)
        except Exception:
            self._logger.exception(
                "event publish failed: address=%s content=%s",
                self._presence.address, content,
            )

    # -- 生命周期 ------------------------------------------------------

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeError("ZenohPresence already closed")
        if self._handles is None:
            await self._install_handles()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._closed = True
        await self.revoke()

    # -- 内部 ---------------------------------------------------------

    async def _install_handles(self) -> None:
        address = self._presence.address
        key = self._cell_key(address)

        def _on_query(query: zenoh.Query):
            # 每次远端 query 时 dump 当前 presence 快照 —
            # self._presence 在 announce/provide 中被 in-place 更新, dump 是原子的.
            try:
                payload = self._presence.model_dump_json().encode('utf-8')
                query.reply(query.key_expr, payload)
            except Exception:
                self._logger.exception(
                    "presence queryable error for %s", address,
                )

        # queryable 必须先于 liveness token — 避免 subscriber 收到 PUT 时
        # queryable 尚未就位导致空回复 (老 ZenohCellNetwork 的 TOCTOU 教训).
        queryable = self._session.declare_queryable(key, _on_query)
        event_publisher = self._session.declare_publisher(self._event_key(address))
        liveness_token = self._session.liveliness().declare_token(key)

        host_token: zenoh.LivelinessToken | None = None
        if self._presence.is_host:
            # §ZZ-10 副路径 hosts_ns/{scope}/cells/liveness/{address} 作废但暂不删.
            # 老版做的旁路 host 声明 (为了不依赖 address[0] 保留字机制) — 本轮
            # 承认 address[0]='host/' 保留字 + wildcard subscribe, 副路径语义
            # 已由主路径 (cells_ns 下的 host/xxx address) 承担. 保留代码以免破坏
            # 现有 ZenohLivenessListener 消费者, wire-up 后期或 §AAA 统一清理.
            host_token = self._session.liveliness().declare_token(
                self._host_liveness_key(address),
            )

        self._handles = _ZenohHandles(
            liveness_token=liveness_token,
            queryable=queryable,
            event_publisher=event_publisher,
            host_token=host_token,
        )
        self._logger.debug(
            "presence announced: address=%s is_host=%s membrane=%s",
            address, self._presence.is_host, self._presence.membrane,
        )
