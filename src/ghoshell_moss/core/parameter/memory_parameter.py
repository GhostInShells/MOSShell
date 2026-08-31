"""MemoryParameters — 单进程内存 Parameters, 参考实现."""

from typing import Callable

from ghoshell_moss.core.blueprint.parameter import ParameterModel
from ghoshell_moss.core.parameter._base import AbsParameters

__all__ = ["MemoryParameters"]


class MemoryParameters(AbsParameters):
    """单进程: declare 与 subscribe 都在本 store, transport = 本地路由, 无 address 概念."""

    def __init__(self, *, logger=None):
        super().__init__(logger=logger)
        self._routes: dict[str, list[Callable[[ParameterModel], None]]] = {}

    async def _publish_declaration(self, key: str, parameter: ParameterModel) -> None:
        for callback in list(self._routes.get(key, [])):
            callback(parameter)

    async def _subscribe_parameter(
            self,
            *,
            key: str,
            model,
            address: str | None,
            callback: Callable[[ParameterModel], None],
    ) -> Callable[[], None]:
        if key in self._declarations:
            callback(self._declarations[key].value)
        self._routes.setdefault(key, []).append(callback)

        def dispose() -> None:
            try:
                self._routes.get(key, []).remove(callback)
            except ValueError:
                pass

        return dispose
