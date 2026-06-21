import threading
from datetime import datetime, timezone
from typing import NamedTuple, Literal

from ghoshell_container import IoCContainer

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.cell import normalize
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.tools.zenoh_helper import (
    MOSSScopeNamespace,
    MOSSNamespace,
    ZenohLivenessListener,
)
from ._utils import HubKeyExpr
from ._provider import ZenohChannelProvider
from ._proxy import ZenohProxyChannel

__all__ = ['ZenohChannelHub', 'HubRecord']


class HubRecord(NamedTuple):
    address: str
    status: Literal["online", "offline"]
    updated_at: datetime


class ZenohChannelHub:
    def __init__(
            self,
            zenoh_session: zenoh.Session,
            scope: str,
            container: IoCContainer | None = None,
            liveness_check_interval: float = 3.0,
            max_records: int = 256,
            logger: LoggerItf | None = None,
            namespace: MOSSNamespace | None = None,
    ):
        namespace = namespace or MOSSScopeNamespace(scope=scope)
        self._namespace = namespace
        self._hub_expr = HubKeyExpr(self._namespace.channels_namespace)
        self._scope = scope
        self._container = container
        self._zenoh_session = zenoh_session
        self._liveness_check_interval = liveness_check_interval
        self._max_records = max_records
        self._logger = logger or get_moss_logger()
        self._proxies: dict[str, ZenohProxyChannel] = {}
        self._records: list[HubRecord] = []
        self._lock = threading.Lock()

        # liveness 发现委托给 ZenohLivenessListener
        # 用 hub_expr 推导 prefix，与 provider 端 key 结构自动对齐
        liveness_prefix = self._hub_expr.new_expr('**').provider_liveness_prefix
        self._liveness_listener = ZenohLivenessListener(
            liveness_prefix=liveness_prefix,
            session=zenoh_session,
            logger=self._logger,
            on_online=self._on_provider_online,
            on_offline=self._on_provider_offline,
        )
        self._started = False

    def provider(self, address: str) -> ZenohChannelProvider:
        return ZenohChannelProvider(
            zenoh_session=self._zenoh_session,
            container=self._container,
            address=address,
            scope=self._scope,
            liveness_check_interval=self._liveness_check_interval,
            namespace=self._namespace,
        )

    def proxy(self, address: str, *, name: str | None = None, description: str = '') -> ZenohProxyChannel:
        existing = self._proxies.get(address)
        if existing is not None:
            return existing

        parsed_name = name or address.replace('/', '_')
        if not Channel.validate_name(parsed_name):
            raise ValueError(f"Invalid channel name {name} of proxy {address} ")

        proxy = ZenohProxyChannel(
            zenoh_session=self._zenoh_session,
            address=address,
            scope=self._scope,
            name=parsed_name,
            description=description,
            namespace=self._namespace,
        )
        with self._lock:
            if address in self._proxies:
                return self._proxies[address]
            self._proxies[address] = proxy
            return proxy

    @property
    def proxies(self) -> dict[str, ZenohProxyChannel]:
        return dict(self._proxies)

    @property
    def records(self) -> list[HubRecord]:
        with self._lock:
            return list(self._records)

    def get_liveness_provider_address(self) -> list[str]:
        """同步阻塞查询当前在线的 provider address 列表。"""
        return self._liveness_listener.get_liveness_keys()

    async def get_liveness_provider_address_async(self) -> list[str]:
        return await self._liveness_listener.get_liveness_keys_async()

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        await self._liveness_listener.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._liveness_listener.__aexit__(exc_type, exc_val, exc_tb)
        self._proxies.clear()
        self._started = False

    # -- liveness callbacks -------------------------------------------

    def _on_provider_online(self, address: str) -> None:
        now = datetime.now(timezone.utc)
        try:
            if address not in self._proxies:
                name = normalize(address)
                self.proxy(address, name=name)
            self._add_record(
                HubRecord(address=address, status="online", updated_at=now)
            )
        except Exception:
            self._logger.exception(
                "ZenohChannelHub(%s) failed to handle provider online: %s",
                self._scope, address,
            )

    def _on_provider_offline(self, address: str) -> None:
        now = datetime.now(timezone.utc)
        try:
            self._proxies.pop(address, None)
            self._add_record(
                HubRecord(address=address, status="offline", updated_at=now)
            )
        except Exception:
            self._logger.exception(
                "ZenohChannelHub(%s) failed to handle provider offline: %s",
                self._scope, address,
            )

    def _add_record(self, record: HubRecord) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
