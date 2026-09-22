"""
ZenohParametersBroadcaster — Parameters 的 zenoh transport.

与 MemoryParametersBroadcaster 同接口, 把参数 key 组映射到 zenoh key expr
(prefix 由 ParameterNamespace 派生自 MatrixNamespace.param_ns):

    {param_ns}/host/truth/{key}          host 广播真值
    {param_ns}/worker/declaration/{key}  节点发布声明 (声明即 require)
    {param_ns}/host/liveness/{address}   host 上线令牌 (address 即化身)

key 组 address-free — address 只活在 ParameterData.meta 里, 不参与寻址.
"""

import contextlib
from typing import Callable

from ghoshell_moss.depends import depend_matrix

depend_matrix()

import zenoh

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.parameter import ParameterData
from ghoshell_moss.core.parameter import ParametersBroadcaster
from ghoshell_moss.matrix.zenoh_helper import MatrixNamespace, ZenohLivenessListener

__all__ = ["ParameterNamespace", "ZenohParametersBroadcaster"]


class ParameterNamespace:
    """parameter 面 key 组 — address-free (见 ParametersBroadcaster).

    只从 MatrixNamespace.param_ns 派生, 不污染 zenoh_helper.
    """

    def __init__(self, namespace: MatrixNamespace):
        self._param_ns = namespace.param_ns
        self.truth_ns = '/'.join([self._param_ns, 'host', 'truth'])
        self.declaration_ns = '/'.join([self._param_ns, 'worker', 'declaration'])
        self.liveness_ns = '/'.join([self._param_ns, 'host', 'liveness'])

    def truth_key(self, key: str) -> str:
        return '/'.join([self.truth_ns, key])

    def declaration_key(self, key: str) -> str:
        return '/'.join([self.declaration_ns, key])

    def declaration_wildcard(self) -> str:
        return f"{self.declaration_ns}/**"

    def liveness_key(self, address: str) -> str:
        return '/'.join([self.liveness_ns, address])


class ZenohParametersBroadcaster(ParametersBroadcaster):

    def __init__(
            self,
            session: zenoh.Session,
            namespace: MatrixNamespace,
            *,
            logger: LoggerItf | None = None,
    ):
        self._session = session
        self._ns = ParameterNamespace(namespace)
        self._logger = logger or get_moss_logger()
        self._host_token: zenoh.LivelinessToken | None = None
        self._liveness_listener: ZenohLivenessListener | None = None

    # -- 生命周期 --------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._host_token is not None:
            with contextlib.suppress(Exception):
                self._host_token.undeclare()
            self._host_token = None
        if self._liveness_listener is not None:
            await self._liveness_listener.__aexit__(exc_type, exc_val, exc_tb)
            self._liveness_listener = None

    # -- 真值通道 --------------------------------------------------------

    async def subscribe_host_truth(self, key: str, callback: Callable[[ParameterData], None]) -> Callable[[], None]:
        subscriber = self._session.declare_subscriber(
            self._ns.truth_key(key), self._decode(callback),
        )

        def dispose() -> None:
            with contextlib.suppress(Exception):
                subscriber.undeclare()

        return dispose

    async def publish_host_truth(self, parameter: ParameterData) -> None:
        self._session.put(self._ns.truth_key(parameter.key), parameter.model_dump_json())

    # -- 声明通道 --------------------------------------------------------

    async def publish_declaration(self, parameter: ParameterData) -> None:
        self._session.put(self._ns.declaration_key(parameter.key), parameter.model_dump_json())

    async def subscribe_declarations(self, callback: Callable[[ParameterData], None]) -> Callable[[], None]:
        subscriber = self._session.declare_subscriber(
            self._ns.declaration_wildcard(), self._decode(callback),
        )

        def dispose() -> None:
            with contextlib.suppress(Exception):
                subscriber.undeclare()

        return dispose

    # -- host 化身 --------------------------------------------------------

    def set_host(self, address: str, epoch: str) -> None:
        # 存在性靠 liveness token (host 死则自动下线); 化身 = host 地址, 进 key 便于 debug.
        # epoch == address (见 TruthHostParameters), 故只用 address.
        self._host_token = self._session.liveliness().declare_token(self._ns.liveness_key(address))

    async def on_host_alive(self, callback: Callable[[str, str], None]) -> None:
        self._liveness_listener = ZenohLivenessListener(
            liveness_prefix=self._ns.liveness_ns,
            session=self._session,
            logger=self._logger,
            on_online=lambda address: callback(address, address),
        )
        await self._liveness_listener.__aenter__()

    # -- 内部 -----------------------------------------------------------

    def _decode(self, callback: Callable[[ParameterData], None]):
        def _on_sample(sample: zenoh.Sample) -> None:
            try:
                callback(ParameterData.model_validate_json(sample.payload.to_string()))
            except Exception:
                self._logger.exception("parameter sample decode failed")

        return _on_sample
