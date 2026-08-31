"""ZenohParameters — matrix 层 point-to-point parameter 实现.

单声明者, 无仲裁: 每个 cell 一个实例, declare 的 key 由 cell address 命名空间
隔离, subscribe 按 address 点对点定向.

key 结构:  {param_ns}/{address}/{key}
  - 声明者 declare 时, 挂一个 wildcard queryable ({param_ns}/{自己}/**) 服务初值查询;
  - set 时 put 到 {param_ns}/{自己}/{key} 推给订阅者;
  - 订阅者 subscribe 时, query 一次初值 + declare_subscriber 持续收推.
"""

import asyncio
from typing import Callable

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.parameter import ParameterModel
from ghoshell_moss.core.parameter import AbsParameters
from ghoshell_moss.depends import depend_matrix
from ghoshell_moss.matrix.zenoh_helper import MatrixNamespace

depend_matrix()

import zenoh

__all__ = ["ZenohParameters"]


class ZenohParameters(AbsParameters):

    def __init__(
            self,
            zenoh_session: zenoh.Session,
            namespace: MatrixNamespace,
            address: str,
            *,
            logger: LoggerItf | None = None,
    ):
        super().__init__(logger=logger)
        self._session = zenoh_session
        self._param_ns = namespace.param_ns
        self._address = address.strip("/")
        self._queryable: zenoh.Queryable | None = None

    # -- key expr -------------------------------------------------------

    def _key_expr(self, address: str, key: str) -> str:
        return "/".join([self._param_ns, address.strip("/"), key])

    def _own_wildcard(self) -> str:
        return f"{self._param_ns}/{self._address}/**"

    def _extract_key(self, key_expr: str) -> str | None:
        prefix = f"{self._param_ns}/{self._address}/"
        if key_expr.startswith(prefix):
            return key_expr[len(prefix):]
        return None

    # -- 生命周期 -------------------------------------------------------

    async def __aenter__(self):
        await super().__aenter__()
        self._queryable = self._session.declare_queryable(self._own_wildcard(), self._on_query)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._queryable is not None:
            try:
                self._queryable.undeclare()
            except Exception:
                pass
            self._queryable = None
        await super().__aexit__(exc_type, exc_val, exc_tb)

    # -- transport ------------------------------------------------------

    def _on_query(self, query: zenoh.Query) -> None:
        key = self._extract_key(str(query.key_expr))
        if key is None:
            return
        declaration = self._declarations.get(key)
        if declaration is None:
            return
        query.reply(query.key_expr, declaration.value.model_dump_json())

    async def _publish_declaration(self, key: str, parameter: ParameterModel) -> None:
        key_expr = self._key_expr(self._address, key)
        self._session.put(key_expr, parameter.model_dump_json())  # put 非阻塞 (zenoh 内部队列)

    async def _subscribe_parameter(
            self,
            *,
            key: str,
            model,
            address: str | None,
            callback: Callable[[ParameterModel], None],
    ) -> Callable[[], None]:
        target = (address or self._address).strip("/")
        key_expr = self._key_expr(target, key)

        def _on_sample(sample: zenoh.Sample) -> None:
            try:
                value = model.model_validate_json(sample.payload.to_string())
                callback(value)
            except Exception:
                self._logger.exception("parameter %s push decode failed", key)

        # 先订阅再 query — 避免订阅与查询之间漏掉一次 push.
        subscriber = self._session.declare_subscriber(key_expr, _on_sample)
        initial = await asyncio.to_thread(self._query_initial, key_expr, model)
        if initial is not None:
            callback(initial)

        def dispose() -> None:
            try:
                subscriber.undeclare()
            except Exception:
                pass

        return dispose

    def _query_initial(self, key_expr: str, model) -> ParameterModel | None:
        for reply in self._session.get(key_expr):
            if reply.ok:
                return model.model_validate_json(reply.ok.payload.to_string())
        return None
