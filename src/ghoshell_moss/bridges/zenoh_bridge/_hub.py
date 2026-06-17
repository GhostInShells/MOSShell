import threading
from datetime import datetime, timezone
from typing import NamedTuple, Literal

from ghoshell_container import IoCContainer

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ._utils import ChannelBridgeHubExpr
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
    ):
        self._hub_expr = ChannelBridgeHubExpr(scope=scope)
        self._scope = scope
        self._container = container
        self._zenoh_session = zenoh_session
        self._liveness_check_interval = liveness_check_interval
        self._max_records = max_records
        self._logger = logger or get_moss_logger()
        self._proxies: dict[str, ZenohProxyChannel] = {}
        self._records: list[HubRecord] = []
        self._lock = threading.Lock()
        self._liveness_subscriber: zenoh.Subscriber | None = None
        self._started = False

    def provider(self, address: str) -> ZenohChannelProvider:
        return ZenohChannelProvider(
            zenoh_session=self._zenoh_session,
            container=self._container,
            address=address,
            scope=self._scope,
            bridge_expr=self._hub_expr.new_expr(address=address),
            liveness_check_interval=self._liveness_check_interval,
        )

    def proxy(self, address: str, *, name: str | None = None, description: str = '') -> ZenohProxyChannel:
        # fast path — 绝大多数调用已在 dict 中
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
            bridge_expr=self._hub_expr.new_expr(address=address),
        )
        # double-check under lock — 防止与 liveness 回调竞态
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
        """同步阻塞查询当前在线的 provider address 列表。

        内部调用 zenoh_session.liveliness().get()，
        同进程查询通常很快，跨网络需注意阻塞 event loop。
        """
        expr = self._hub_expr.new_expr('**')
        prefix = expr.provider_liveness_prefix
        wildcard = expr.provider_liveness_key
        result = []
        for sample in self._zenoh_session.liveliness().get(wildcard):
            if not sample.ok:
                continue
            sample_key_expr = str(sample.result.key_expr)
            if sample_key_expr.startswith(prefix):
                address = sample_key_expr[len(prefix):].strip('/')
                result.append(address)
        return result

    async def get_liveness_provider_address_async(self) -> list[str]:
        """get_liveness_provider_address 的异步版本 — 将阻塞查询卸载到线程池。"""
        import asyncio
        return await asyncio.to_thread(self.get_liveness_provider_address)

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        wildcard_expr = self._hub_expr.new_expr('**')
        liveness_wildcard = wildcard_expr.provider_liveness_key
        prefix = wildcard_expr.provider_liveness_prefix
        self._liveness_subscriber = self._zenoh_session.liveliness().declare_subscriber(
            liveness_wildcard,
            lambda sample: self._on_provider_liveness(sample, prefix),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._liveness_subscriber is not None:
            try:
                self._liveness_subscriber.undeclare()
            except RuntimeError:
                pass
            self._liveness_subscriber = None
        self._proxies.clear()
        self._started = False

    def _on_provider_liveness(self, sample: zenoh.Sample, prefix: str) -> None:
        sample_key = str(sample.key_expr)
        if not sample_key.startswith(prefix):
            return
        address = sample_key[len(prefix):].strip('/')
        if not address:
            return
        now = datetime.now(timezone.utc)

        if sample.kind == zenoh.SampleKind.PUT:
            try:
                if address not in self._proxies:
                    self.proxy(address)
                self._add_record(HubRecord(address=address, status="online", updated_at=now))
            except Exception:
                self._logger.exception(
                    "ZenohChannelHub(%s) failed to handle provider online: %s",
                    self._scope, address,
                )
        elif sample.kind == zenoh.SampleKind.DELETE:
            try:
                self._proxies.pop(address, None)
                self._add_record(HubRecord(address=address, status="offline", updated_at=now))
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