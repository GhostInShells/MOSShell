from typing import Type, Iterable

from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.contracts.configs import ConfigStore, YamlConfigStore

__all__ = [
    'EnvConfigStoreProvider',
]


class EnvConfigStoreProvider(Provider):
    # ConfigStore 装配的 workspace 声明入口. 老实现是 BootstrapProvider — bootstrap
    # 方法读老 core.blueprint.manifests.Manifests, 逐个 config_info 装载. 那条路径
    # 已由 MatrixImpl._container_lifecycle_ctx 承接 (新 MatrixManifest.configs()),
    # 老 bootstrap 是死代码, 一并清除, 类型降级为普通 Provider.

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ConfigStore:
        ws = con.force_fetch(Workspace)
        storage = ws.configs()

        config_store = YamlConfigStore(storage)

        return config_store

    def contract(self) -> Type[INSTANCE]:
        return ConfigStore

    def aliases(self) -> Iterable[Type[INSTANCE]]:
        yield YamlConfigStore
