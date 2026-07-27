import threading
from datetime import datetime, timezone
from typing import Callable, NamedTuple, Literal

from ghoshell_container import IoCContainer

from ghoshell_moss.depends import depend_matrix

depend_matrix()
import zenoh

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.cell import normalize
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.tools.zenoh_helper import (
    MatrixNamespace,
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
    """Zenoh channel provider/proxy 注册中心 + 上下线感知.

    职责:
      - provider/proxy 单例去重 (按 address dedup)
      - liveness listener 转 callback fan-out (4 路: provider/proxy × online/offline)
      - 启动唯一性: 同 address 重复 declare provider 第二次 raise (hub 层 dedup, 非协议层 TOCTOU)

    不职责 (留给 ZenohChannelProvider / ZenohProxyChannel 内部):
      - 实际通讯 channel 建立, key 暴露
      - proxy / provider 握手与连接感知 — 由 duplex 协议自动完成
        (core/duplex/protocol.py 的 CreateSession / SessionCreated / Heartbeat
        事件链), hub 层不介入.

    已知遗留 (v0.2 处理):
      - 跨进程同 address declare provider 无二次网络确认 (TOCTOU). §TT-2 uuid
        派生让本场景罕见, 但真发生时底层 zenoh 都会 accept, 需要网络级 confirm.
    """

    def __init__(
            self,
            zenoh_session: zenoh.Session,
            scope: str,
            container: IoCContainer | None = None,
            liveness_check_interval: float = 3.0,
            max_records: int = 256,
            logger: LoggerItf | None = None,
            namespace: MatrixNamespace | None = None,
    ):
        namespace = namespace or MatrixNamespace(network_scope=scope)
        self._namespace = namespace
        self._hub_expr = HubKeyExpr(self._namespace.channels_ns)
        self._scope = scope
        self._container = container
        self._zenoh_session = zenoh_session
        self._liveness_check_interval = liveness_check_interval
        self._max_records = max_records
        self._logger = logger or get_moss_logger()
        self._providers: dict[str, ZenohChannelProvider] = {}
        self._proxies: dict[str, ZenohProxyChannel] = {}
        # name registry: 派生 proxy name 时保证 hub 内唯一.
        # 策略: 无冲突用 hint 本身; 冲突则 hint_2/hint_3... counter 单调递增, 不复用.
        # drop_proxy 释放 zenoh 资源但不回收 name — cell 走了再来 (即使同 hint) 也拿新编号.
        # 语义: 编号是 "这个 hub 一生看过的第 N 个同名 cell", 不误认.
        self._used_names: set[str] = set()
        self._name_counter: dict[str, int] = {}
        self._records: list[HubRecord] = []
        self._lock = threading.Lock()

        # user-registered callbacks (fan-out from listener / explicit calls)
        self._cb_provider_online: list[Callable[[str], None]] = []
        self._cb_provider_offline: list[Callable[[str], None]] = []
        self._cb_proxy_online: list[Callable[[str], None]] = []
        self._cb_proxy_offline: list[Callable[[str], None]] = []

        # liveness 发现委托给 ZenohLivenessListener
        # 用 hub_expr 推导 prefix, 与 provider 端 key 结构自动对齐
        liveness_prefix = self._hub_expr.new_expr('**').provider_liveness_prefix
        self._liveness_listener = ZenohLivenessListener(
            liveness_prefix=liveness_prefix,
            session=zenoh_session,
            logger=self._logger,
            on_online=self._on_provider_online,
            on_offline=self._on_provider_offline,
        )
        self._started = False

    # ── provider ──────────────────────────────────────────────────────

    def provider(self, address: str) -> ZenohChannelProvider:
        """声明 address 为本 hub 提供 channel.

        :raise RuntimeError: 同 address 已在本 hub 注册 provider (dedup).
                            注意: 这是 hub 实例内 dedup, 不是网络级唯一性检查 —
                            网络级唯一性应由 CellNetwork.check_unique 完成.
        """
        with self._lock:
            existing = self._providers.get(address)
            if existing is not None:
                raise RuntimeError(
                    f"hub already has provider for address={address!r}; "
                    f"call get_provider() instead of provider() to retrieve it"
                )
        new_provider = ZenohChannelProvider(
            zenoh_session=self._zenoh_session,
            container=self._container,
            address=address,
            scope=self._scope,
            liveness_check_interval=self._liveness_check_interval,
            namespace=self._namespace,
        )
        with self._lock:
            existing = self._providers.get(address)
            if existing is not None:
                # 并发 race: 让 caller 拿到先到的 (此处仅为防御)
                return existing
            self._providers[address] = new_provider
        return new_provider

    def get_provider(self, address: str) -> ZenohChannelProvider | None:
        """按 address 取已注册 provider. 不存在返回 None."""
        with self._lock:
            return self._providers.get(address)

    # ── proxy ─────────────────────────────────────────────────────────

    def proxy(
            self,
            address: str,
            *,
            name_hint: str | None = None,
            description: str = '',
    ) -> ZenohProxyChannel:
        """获取或创建 address 对应的 proxy. 已存在则返回现有实例.

        与 provider() 不同, proxy() 允许多次 call 同 address — 多个 caller 共享同一 proxy.

        name 派生: name_hint 无冲突 → 直接用; 冲突 → hint_2/hint_3... (hub 内计数单调).
        drop_proxy 后 name 不复用: 同 hint 再来会拿下一个编号.
        """
        existing = self._proxies.get(address)
        if existing is not None:
            return existing

        # 兜底 hint: 未传时用 normalize(address). 保证 hub 内部有派生源.
        hint = name_hint or address.replace('/', '_')
        with self._lock:
            derived_name = self._derive_name_locked(hint)

        if not Channel.validate_name(derived_name):
            raise ValueError(
                f"Invalid channel name {derived_name!r} derived from hint {hint!r} for proxy {address}"
            )

        new_proxy = ZenohProxyChannel(
            zenoh_session=self._zenoh_session,
            address=address,
            scope=self._scope,
            name=derived_name,
            description=description,
            namespace=self._namespace,
        )
        with self._lock:
            existing = self._proxies.get(address)
            if existing is not None:
                # 并发 race: 让 caller 拿到先到的. 已 add 的 derived_name 保留占位
                # (单调策略不回滚, 下次同 hint 拿再下一个编号 — 少见但代价可接受).
                return existing
            self._proxies[address] = new_proxy
        # proxy 创建 = "proxy online" 事件 fan-out (不依赖 liveness listener)
        # 设计选择: proxy 上线由 hub 显式触发 (调用 self.proxy()), 不依赖远端 liveness.
        # 而 provider online/offline 来自 ZenohLivenessListener 的 zenoh 后台线程回调.
        self._fan_out(self._cb_proxy_online, address)
        return new_proxy

    def _derive_name_locked(self, hint: str) -> str:
        """派生一个 hub 内唯一的 proxy name (调用方须持 self._lock).

        无冲突 → hint 本身; 冲突 → hint_N (N 从 counter[hint] 起, 单调).
        写入 self._used_names + 更新 self._name_counter[hint].
        """
        if hint not in self._used_names:
            self._used_names.add(hint)
            return hint
        # 冲突: 从 counter 记录的下一个编号找起.
        next_n = self._name_counter.get(hint, 2)
        candidate = f"{hint}_{next_n}"
        while candidate in self._used_names:
            next_n += 1
            candidate = f"{hint}_{next_n}"
        self._name_counter[hint] = next_n + 1
        self._used_names.add(candidate)
        return candidate

    def get_proxy(self, address: str) -> ZenohProxyChannel | None:
        """按 address 取已注册 proxy. 不存在返回 None."""
        return self._proxies.get(address)

    def drop_proxy(self, address: str) -> bool:
        """主动移除 proxy. 触发 on_proxy_offline 回调.

        :return: 真的 pop 了返回 True.
        """
        with self._lock:
            removed = self._proxies.pop(address, None)
        if removed is None:
            return False
        self._fan_out(self._cb_proxy_offline, address)
        return True

    @property
    def proxies(self) -> dict[str, ZenohProxyChannel]:
        return dict(self._proxies)

    @property
    def providers(self) -> dict[str, ZenohChannelProvider]:
        return dict(self._providers)

    @property
    def records(self) -> list[HubRecord]:
        with self._lock:
            return list(self._records)

    def get_liveness_provider_address(self) -> list[str]:
        """同步阻塞查询当前在线的 provider address 列表."""
        return self._liveness_listener.get_liveness_keys()

    async def get_liveness_provider_address_async(self) -> list[str]:
        return await self._liveness_listener.get_liveness_keys_async()

    # ── callback 注册 (返回 unsubscribe 函数) ─────────────────────────

    def on_provider_online(self, cb: Callable[[str], None]) -> Callable[[], None]:
        return self._register(self._cb_provider_online, cb)

    def on_provider_offline(self, cb: Callable[[str], None]) -> Callable[[], None]:
        return self._register(self._cb_provider_offline, cb)

    def on_proxy_online(self, cb: Callable[[str], None]) -> Callable[[], None]:
        return self._register(self._cb_proxy_online, cb)

    def on_proxy_offline(self, cb: Callable[[str], None]) -> Callable[[], None]:
        return self._register(self._cb_proxy_offline, cb)

    @staticmethod
    def _register(target: list, cb: Callable[[str], None]) -> Callable[[], None]:
        target.append(cb)

        def _unsubscribe() -> None:
            try:
                target.remove(cb)
            except ValueError:
                pass
        return _unsubscribe

    def _fan_out(self, callbacks: list[Callable[[str], None]], address: str) -> None:
        for cb in list(callbacks):
            try:
                cb(address)
            except Exception:
                self._logger.exception(
                    "ZenohChannelHub(%s) callback raised for address=%s",
                    self._scope, address,
                )

    # ── lifecycle ─────────────────────────────────────────────────────

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        await self._liveness_listener.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._liveness_listener.__aexit__(exc_type, exc_val, exc_tb)
        self._proxies.clear()
        self._providers.clear()
        # hub 生命周期结束, name registry 归零 (下次启动重新计数).
        self._used_names.clear()
        self._name_counter.clear()
        self._cb_provider_online.clear()
        self._cb_provider_offline.clear()
        self._cb_proxy_online.clear()
        self._cb_proxy_offline.clear()
        self._started = False

    # ── liveness listener callbacks (zenoh 后台线程) ─────────────────

    def _on_provider_online(self, address: str) -> None:
        """ZenohLivenessListener 回调 — provider 在网络上上线."""
        now = datetime.now(timezone.utc)
        try:
            self._add_record(
                HubRecord(address=address, status="online", updated_at=now)
            )
        except Exception:
            self._logger.exception(
                "ZenohChannelHub(%s) failed to record provider online: %s",
                self._scope, address,
            )
        # 注意: hub 不再自动 build proxy — 由上层 CellNetwork 决定建不建 proxy.
        # 仅 fan-out 通知, 上层 (CellNetwork) 自己调 self.proxy(address) 完成 build.
        self._fan_out(self._cb_provider_online, address)

    def _on_provider_offline(self, address: str) -> None:
        """ZenohLivenessListener 回调 — provider 在网络上下线."""
        now = datetime.now(timezone.utc)
        try:
            # 自动 pop 对应 proxy, 防止悬挂
            with self._lock:
                popped = self._proxies.pop(address, None)
            self._add_record(
                HubRecord(address=address, status="offline", updated_at=now)
            )
        except Exception:
            self._logger.exception(
                "ZenohChannelHub(%s) failed to record provider offline: %s",
                self._scope, address,
            )
            popped = None
        if popped is not None:
            self._fan_out(self._cb_proxy_offline, address)
        self._fan_out(self._cb_provider_offline, address)

    def _add_record(self, record: HubRecord) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
