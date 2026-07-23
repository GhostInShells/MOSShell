"""
Matrix 网络 driver 分发的 adapter 抽象 (§ZZ-3).

一个 adapter 承担 "从 NetworkMetadata 起 driver 底层资源 + 提供 Presence /
Watcher factory" 的职责. matrix 层不知道具体 driver (zenoh / redis / mqtt+ws+http
/ api_key 混合客户端), 只知道 MatrixNetworkAdapter.

窄接口纪律 (§ZZ-3): driver_name / async lifecycle / new_presence / new_watcher —
仅四件事. adapter 内部结构 (session 数量 / hub 数量 / 连接形态 / api_key 池) 全部
私有. 未来 driver 混合形态 (redis + mqtt + ws + http) 可自由在 __aenter__ 里
持多客户端, 对 matrix 层无感.

本模块一期放实现层, 未来倾向迁 blueprint (blueprint = interface, adapter =
abstract, 现在暂时不进 L2 推理上下文).
"""

from abc import ABC, abstractmethod
from typing import Iterable
from typing_extensions import Self

from ghoshell_container import Provider, IoCContainer
from ghoshell_moss.core.blueprint.cell import Cell, CellPresence, CellNetwork
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import NetworkMetadata

import logging

__all__ = [
    'MatrixNetworkAdapter',
    'register_adapter',
    'get_adapter_class',
    'list_adapter_drivers',
]


class MatrixNetworkAdapter(ABC):
    """
    Matrix 网络 driver 的 adapter — driver 私有, matrix 层无感.

    构造签名不进 ABC (刻意): 各 driver 自己决定构造参数 (zenoh 要 metadata +
    is_host + scope; redis 可能要 api_key + endpoint list; ws+http 混合要
    key 池 + endpoint 表). factory 层按 driver_name 查到 class 后, 显式给
    构造参数.
    """

    @classmethod
    @abstractmethod
    def driver_name(cls) -> str:
        """驱动名 (与 NetworkMetadata.driver / NetworkConfig.driver_name 同构)."""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """起 driver 底层资源 (session / hub / 连接池 / api_key 池 / ...)."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """清理 driver 底层资源."""
        ...

    @abstractmethod
    def new_presence(
            self,
            presence_data: Cell,
            *,
            logger: logging.Logger,
    ) -> CellPresence:
        """
        构造本 cell 的入网侧对象.

        返回未 enter 的 Presence — matrix 层把它加进 async exit stack 触发
        __aenter__ 完成首次 announce.
        """
        ...

    @abstractmethod
    def new_watcher(
            self,
            *,
            env: Environment,
            logger: logging.Logger,
    ) -> CellNetwork:
        """
        构造网络观察侧对象.

        返回未 enter 的 Watcher — matrix.mesh() 惰性调用, 首次调用时把它加进
        exit stack. opt-in by usage (§UU-7): 纯 worker cell 不调 mesh() 即不
        创建, 网络观察成本 O(N) 只对必要消费者付出.

        :param env: 环境载体, 用于 cell.is_local(env) 判定 (§UU-7 local/foreign 分档).
        """
        ...

    # ---- optional driver-specific IoC hooks (default = no-op) ----

    def bind_ioc(self, container: IoCContainer) -> None:
        """
        driver-specific 直接绑 driver 底层对象到 IoC (如 zenoh.Session).

        matrix 调用时机: adapter.__aenter__ 完成 → bind_ioc → container.bootstrap.
        default: no-op. driver 有需要就 override.
        """
        return None

    def default_providers(self) -> Iterable[Provider]:
        """
        driver-specific 默认 provider (如 zenoh 场景的 topic_service / session).

        matrix 用 "if not bound" 兜底方式装配, workspace 用户在 MatrixManifest
        里显式覆写即可覆盖 driver default.
        """
        return []

    # ---- factory method hint (non-abstract, 各 driver 自己覆写) ----

    @classmethod
    def from_metadata(
            cls,
            metadata: NetworkMetadata,
            *,
            is_host: bool,
            **kwargs,
    ) -> 'MatrixNetworkAdapter':
        """
        统一工厂入口 — factory._create_matrix 从 driver_name → adapter_cls 后
        经此调用 driver-specific 构造. 默认 raise, driver 自己覆写.

        本方法**不进 abstract 契约** (§ZZ-3 构造签名不承诺), 是 factory 层的
        便捷 helper. driver 也可以完全绕开它, 走自己私有的构造签名.
        """
        raise NotImplementedError(
            f"{cls.__name__}.from_metadata not implemented — use driver-specific "
            f"constructor or override this classmethod."
        )


# ---- module-level driver registry ---------------------------------
# §ZZ-3: driver_name → adapter class. 装饰器 register_adapter 用于自动注册,
# adapter 模块 (如 zenoh_adapter.py) import 副作用即注册.

_ADAPTER_REGISTRY: dict[str, type[MatrixNetworkAdapter]] = {}


def register_adapter(cls: type[MatrixNetworkAdapter]) -> type[MatrixNetworkAdapter]:
    """装饰器 or 直接调用: 注册一个 MatrixNetworkAdapter 实现.

    用法:

        @register_adapter
        class ZenohAdapter(MatrixNetworkAdapter):
            ...
    """
    name = cls.driver_name()
    _ADAPTER_REGISTRY[name] = cls
    return cls


def get_adapter_class(driver_name: str) -> type[MatrixNetworkAdapter] | None:
    """按 driver name 查询已注册的 adapter class. 未注册返回 None."""
    return _ADAPTER_REGISTRY.get(driver_name)


def list_adapter_drivers() -> list[str]:
    """列出已注册的 driver 名 (用于 CLI 展示 / 错误信息)."""
    return sorted(_ADAPTER_REGISTRY.keys())
