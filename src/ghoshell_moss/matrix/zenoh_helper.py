import asyncio
import threading
from typing import Callable

from ghoshell_moss.depends import depend_matrix

depend_matrix()

from ghoshell_moss.core.blueprint.environment import Environment
import zenoh
import logging

__all__ = [
    "MatrixNamespace",
    "MatrixEnvNamespace",
    "ZenohLivenessListener",
]


class MatrixNamespace:
    def __init__(self, network_scope: str):
        self.hosts_ns = "MOSS/matrix/hosts"
        self.network_ns = f"MOSS/matrix/scopes/{network_scope}"

        namespace = self.network_ns.strip("/")
        self.namespace = namespace

        self.host_alive_key = f"{self.network_ns}/host-alive"
        self.channels_ns = '/'.join([namespace, 'channels'])
        self.cells_ns = '/'.join([namespace, 'cells'])
        self.topic_ns = '/'.join([namespace, 'topics'])
        self.signal_ns = '/'.join([namespace, 'signals'])
        self.stream_ns = '/'.join([namespace, 'streams'])
        self.output_ns = '/'.join([namespace, 'outputs'])

    def __str__(self):
        return self.namespace


class MatrixEnvNamespace(MatrixNamespace):

    def __init__(self, env: Environment):
        network_scope = env.network_scope
        super().__init__(network_scope)


# ── ZenohLivenessListener ───────────────────────────────────────────


class ZenohLivenessListener:
    """监听指定 prefix 下所有 liveness token 的在线/离线变化。

    启动时全量获取 + 订阅 liveness 事件 + 慢周期 reconcile。
    ``live_keys`` 属性返回零延时缓存快照。
    """

    def __init__(
            self,
            *,
            liveness_prefix: str,
            session: zenoh.Session,
            logger: logging.Logger | None = None,
            on_online: Callable[[str], None] | None = None,
            on_offline: Callable[[str], None] | None = None,
            reconcile_interval: float = 10.0,
    ):
        self._liveness_prefix = liveness_prefix.rstrip("/")
        self._liveness_wildcard = f"{self._liveness_prefix}/**"
        self._zenoh_session = session
        self._logger = logger or logging.getLogger('moss')
        self._on_online = on_online
        self._on_offline = on_offline
        self._reconcile_interval = reconcile_interval

        self._live_keys: set[str] = set()
        self._lock = threading.Lock()
        self._subscriber: zenoh.Subscriber | None = None
        self._reconcile_task: asyncio.Task | None = None
        self._started = False
        self._closed = False

    # -- public -------------------------------------------------------

    @property
    def live_keys(self) -> list[str]:
        """零延时: 返回当前缓存的在线 key 列表 (去除 prefix 后的相对路径)。"""
        with self._lock:
            return list(self._live_keys)

    def get_liveness_keys(self) -> list[str]:
        """同步阻塞: 全量查询当前在线的 liveness key。

        返回去除 prefix 后的相对路径列表。
        """
        prefix = self._liveness_prefix + '/'
        result: list[str] = []
        for sample in self._zenoh_session.liveliness().get(
                self._liveness_wildcard
        ):
            if not sample.ok:
                continue
            key = str(sample.result.key_expr)
            if key.startswith(prefix):
                result.append(key[len(prefix):])
        return result

    async def get_liveness_keys_async(self) -> list[str]:
        """get_liveness_keys 的异步版本，卸载到线程池。"""
        return await asyncio.to_thread(self.get_liveness_keys)

    # -- lifecycle ----------------------------------------------------

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True

        # 1. initial full query
        initial = await self.get_liveness_keys_async()
        with self._lock:
            self._live_keys = set(initial)

        # 2. subscribe liveness changes
        self._subscriber = self._zenoh_session.liveliness().declare_subscriber(
            self._liveness_wildcard,
            self._on_sample,
        )

        # 3. fire on_online for initial set (after subscriber is live,
        #    so any subscriber-replayed PUT is harmless — idempotent)
        for key in initial:
            if self._on_online:
                try:
                    self._on_online(key)
                except Exception:
                    self._logger.exception(
                        "initial on_online error for %s", key,
                    )

        # 4. background reconcile loop
        loop = asyncio.get_running_loop()
        self._reconcile_task = loop.create_task(self._reconcile_loop())

        self._logger.debug(
            "ZenohLivenessListener started: prefix=%s initial=%d",
            self._liveness_prefix, len(initial),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True

        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
            self._reconcile_task = None

        if self._subscriber is not None:
            try:
                self._subscriber.undeclare()
            except RuntimeError:
                pass
            self._subscriber = None

        with self._lock:
            self._live_keys.clear()
        self._started = False
        self._logger.debug("ZenohLivenessListener stopped: prefix=%s", self._liveness_prefix)

    # -- internal -----------------------------------------------------

    def _extract_identity(self, full_key: str) -> str | None:
        prefix = self._liveness_prefix + '/'
        if not full_key.startswith(prefix):
            return None
        identity = full_key[len(prefix):]
        return identity or None

    def _on_sample(self, sample: zenoh.Sample) -> None:
        """zenoh liveness 回调 (zenoh 后台线程)。"""
        identity = self._extract_identity(str(sample.key_expr))
        if identity is None:
            return

        if sample.kind == zenoh.SampleKind.PUT:
            with self._lock:
                self._live_keys.add(identity)
            if self._on_online:
                try:
                    self._on_online(identity)
                except Exception:
                    self._logger.exception(
                        "on_online callback error for %s", identity,
                    )
        elif sample.kind == zenoh.SampleKind.DELETE:
            with self._lock:
                self._live_keys.discard(identity)
            if self._on_offline:
                try:
                    self._on_offline(identity)
                except Exception:
                    self._logger.exception(
                        "on_offline callback error for %s", identity,
                    )

    async def _reconcile_loop(self):
        """慢周期全量查询，消除事件丢失导致的不一致。"""
        while not self._closed:
            try:
                await asyncio.sleep(self._reconcile_interval)
                if self._closed:
                    return
                current = await self.get_liveness_keys_async()
                with self._lock:
                    added = set(current) - self._live_keys
                    removed = self._live_keys - set(current)
                    self._live_keys = set(current)
                for key in added:
                    if self._on_online:
                        try:
                            self._on_online(key)
                        except Exception:
                            self._logger.exception(
                                "reconcile on_online error for %s", key,
                            )
                for key in removed:
                    if self._on_offline:
                        try:
                            self._on_offline(key)
                        except Exception:
                            self._logger.exception(
                                "reconcile on_offline error for %s", key,
                            )
            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.exception("liveness reconcile error")
