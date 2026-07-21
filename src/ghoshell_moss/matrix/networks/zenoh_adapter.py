"""
Zenoh driver 的 MatrixNetworkAdapter 实现 (§ZZ-3).

承担:
  - 起 zenoh.Session (按 metadata + is_host 起 peer/client)
  - 起 ZenohChannelHub (Presence/Mesh 共享)
  - 拿 CellsKeyspace (Presence/Mesh 共用同一份 key 定义, 无手写魔法值)
  - new_presence → ZenohCellPresence
  - new_watcher  → ZenohCellMesh

import 副作用: register_adapter 自动把 ZenohAdapter 注册进 module-level
adapter registry (get_adapter_class('zenoh') 即可查到).
"""

import asyncio
import logging
from typing import Iterable

from typing_extensions import Self

from ghoshell_container import IoCContainer, Provider, provide

from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub
from ghoshell_moss.core.blueprint.cell import Cell, CellPresence, CellMesh
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import NetworkMetadata
from ghoshell_moss.matrix.adapter import MatrixNetworkAdapter, register_adapter
from ghoshell_moss.matrix.networks._utils import CellsKeyspace
from ghoshell_moss.matrix.networks.zenoh_network import (
    ZENOH_DRIVER, create_zenoh_session_from_metadata,
)
from ghoshell_moss.matrix.networks.zenoh_presence import ZenohCellPresence
from ghoshell_moss.matrix.networks.zenoh_mesh import ZenohCellMesh
from ghoshell_moss.matrix.providers.topic_provider import ZenohTopicServiceProvider
from ghoshell_moss.matrix.providers.moss_session_provider import ProjectZenohSessionProvider
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace

__all__ = ['ZenohAdapter']


@register_adapter
class ZenohAdapter(MatrixNetworkAdapter):
    """Zenoh driver adapter.

    构造签名 (driver 私有, 不进 ABC):
        ZenohAdapter(metadata, *, is_host, cell_type='', logger=None)

    - metadata: NetworkMetadata (含 zenoh config), scope 从 metadata.scope 拿
    - is_host: 显式确认 (§UU-1.2), 决定 zenoh 起 peer(listen) 还是 client(connect)
    - cell_type: 可选 worker 子分类, 走 ZenohNetworkConfig.cell_type_configs 覆盖
    - logger: adapter 内部日志 (与 matrix.logger 平行, 不冒泡到 cell logger)
    """

    def __init__(
            self,
            metadata: NetworkMetadata,
            *,
            is_host: bool,
            cell_type: str = '',
            logger: logging.Logger | None = None,
    ):
        self._metadata = metadata
        self._is_host = is_host
        self._cell_type = cell_type
        self._logger = logger or logging.getLogger('moss.matrix.adapter.zenoh')
        self._scope = metadata.scope
        self._namespace = MatrixNamespace(network_scope=metadata.scope)
        # cells 域下的 key 集合 (presence 与 mesh 共用一份, 无手写魔法值).
        self._cells_keyspace = CellsKeyspace(self._namespace)

        # driver 底层资源, __aenter__ 时装配
        self._session = None
        self._hub: ZenohChannelHub | None = None
        self._started = False
        self._closed = False

    @classmethod
    def driver_name(cls) -> str:
        return ZENOH_DRIVER

    @classmethod
    def from_metadata(
            cls,
            metadata: NetworkMetadata,
            *,
            is_host: bool,
            cell_type: str = '',
            logger: logging.Logger | None = None,
    ) -> 'ZenohAdapter':
        return cls(
            metadata=metadata,
            is_host=is_host,
            cell_type=cell_type,
            logger=logger,
        )

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def namespace(self) -> MatrixNamespace:
        return self._namespace

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True

        # 1. 起 zenoh session (zenoh.open 是 sync blocking, 走 to_thread)
        session = await asyncio.to_thread(
            create_zenoh_session_from_metadata,
            self._metadata,
            is_host=self._is_host,
            cell_type=self._cell_type,
        )
        if session is None:
            raise RuntimeError(
                f"ZenohAdapter failed to open session: metadata.driver="
                f"{self._metadata.driver!r} unsupported by zenoh factory"
            )
        self._session = session

        # 2. 起 hub (共享给 Presence 和 Watcher)
        self._hub = ZenohChannelHub(
            zenoh_session=session,
            scope=self._scope,
            logger=self._logger,
            namespace=self._namespace,
        )
        await self._hub.__aenter__()
        self._logger.debug(
            "ZenohAdapter started: is_host=%s scope=%s namespace=%s",
            self._is_host, self._scope, self._namespace,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._closed:
            return
        self._closed = True

        # hub 先退, session 后关 (hub 内部持 subscriber, session close 会连带断)
        try:
            if self._hub is not None:
                await self._hub.__aexit__(exc_type, exc_val, exc_tb)
        except Exception:
            self._logger.exception("ZenohAdapter hub aexit error")

        if self._session is not None:
            try:
                # zenoh.Session.close 是同步 blocking
                await asyncio.to_thread(self._session.close)
            except Exception:
                self._logger.exception("ZenohAdapter session close error")

    def new_presence(
            self,
            presence_data: Cell,
            *,
            logger: logging.Logger,
    ) -> CellPresence:
        if not self._started or self._session is None or self._hub is None:
            raise RuntimeError(
                "ZenohAdapter not started; call adapter.__aenter__() first"
            )
        return ZenohCellPresence(
            session=self._session,
            logger=logger,
            keyspace=self._cells_keyspace,
            scope=self._scope,
            hub=self._hub,
            presence=presence_data,
        )

    def new_watcher(
            self,
            *,
            env: Environment,
            logger: logging.Logger,
    ) -> CellMesh:
        if not self._started or self._session is None or self._hub is None:
            raise RuntimeError(
                "ZenohAdapter not started; call adapter.__aenter__() first"
            )
        return ZenohCellMesh(
            session=self._session,
            logger=logger,
            keyspace=self._cells_keyspace,
            scope=self._scope,
            hub=self._hub,
            env=env,
        )

    # ---- IoC hooks (§ZZ-5) ------------------------------------------------

    def bind_ioc(self, container: IoCContainer) -> None:
        """注册 lazy zenoh.Session provider (§ZZ-5 provide 语法糖场景).

        闭包捕捉 self, 首次 fetch 时读 self._session — 那时 __aenter__ 已完成
        (matrix 装配次序: sync bind_ioc → sync bootstrap → async adapter.__aenter__
        → async force_fetch(TopicService) 触发本 provider chain).
        """
        import zenoh

        adapter_ref = self

        def _fetch_zenoh_session(_container: IoCContainer) -> zenoh.Session:
            if adapter_ref._session is None:
                raise RuntimeError(
                    "zenoh.Session fetched before ZenohAdapter.__aenter__ completed. "
                    "Check matrix装配次序 (§ZZ-5): adapter 起 driver 是 async 阶段,"
                    "在 container.bootstrap() 之后."
                )
            return adapter_ref._session

        if not container.bound(zenoh.Session):
            container.register(
                provide(zenoh.Session, singleton=True)(_fetch_zenoh_session),
            )

    def default_providers(self) -> Iterable[Provider]:
        """zenoh driver 的默认 provider — TopicService / Session (§ZZ-5).

        matrix 以 "if not bound" 兜底装配, workspace 用户在 MatrixManifest.providers
        里显式覆写即可覆盖.
        """
        yield ZenohTopicServiceProvider()
        yield ProjectZenohSessionProvider()
