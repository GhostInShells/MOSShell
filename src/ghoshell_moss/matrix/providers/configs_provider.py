from typing import Type, Iterable

from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.contracts.configs import ConfigStore, YamlConfigStore
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = [
    'EnvConfigStoreProvider',
]


class EnvConfigStoreProvider(Provider):
    # ConfigStore 装配的 workspace 声明入口. 老实现是 BootstrapProvider — bootstrap
    # 方法读老 core.blueprint.manifests.Manifests, 逐个 config_info 装载. 那条路径
    # 已由 MatrixImpl._container_lifecycle_ctx 承接 (新 MatrixManifest.configs()),
    # 老 bootstrap 是死代码, 一并清除, 类型降级为普通 Provider.
    #
    # mode_name 从 Environment 注入 (mode-aware ConfigStore §Config-3). no_mode
    # 场景传空串, 走 base 视图.

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ConfigStore:
        ws = con.force_fetch(Workspace)
        storage = ws.configs()

        env = con.get(Environment)
        mode_name = ''
        if env is not None and not env.no_mode:
            mode_name = env.mode_name

        config_store = YamlConfigStore(storage, mode_name=mode_name)

        return config_store

    def contract(self) -> Type[INSTANCE]:
        return ConfigStore

    def aliases(self) -> Iterable[Type[INSTANCE]]:
        yield YamlConfigStore
