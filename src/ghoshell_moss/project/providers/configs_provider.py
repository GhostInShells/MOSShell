from typing import Type

from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.core.blueprint.project import Project

__all__ = [
    'EnvConfigStoreProvider',
]


class EnvConfigStoreProvider(Provider):
    # ConfigStore 装配的 workspace 声明入口. 老实现是 BootstrapProvider — bootstrap
    # 方法读老 core.blueprint.manifests.Manifests, 逐个 config_info 装载. 那条路径
    # 已由 Project.container 承接 (ProjectManifest.configs() → ConfigInstanceRegisterBootstrapper),
    # 老 bootstrap 是死代码, 一并清除, 类型降级为普通 Provider.
    #
    # 构造逻辑已收口到 Project.configs (mode-aware, 懒加载单例). 本 provider 只做
    # 委托 — matrix 装配时 Project 已作为单例 set 进 container (MatrixImpl._prepare_container),
    # 不走发现.

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ConfigStore:
        return con.force_fetch(Project).configs

    def contract(self) -> Type[INSTANCE]:
        return ConfigStore
