from typing import Type, Iterable

from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss.contracts.configs import ConfigStore, YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = [
    'EnvConfigStoreProvider',
]


class EnvConfigStoreProvider(Provider):
    # ConfigStore 装配的 workspace 声明入口. 老实现是 BootstrapProvider — bootstrap
    # 方法读老 core.blueprint.manifests.Manifests, 逐个 config_info 装载. 那条路径
    # 已由 Project.container 承接 (ProjectManifest.configs() → ConfigInstanceRegisterBootstrapper),
    # 老 bootstrap 是死代码, 一并清除, 类型降级为普通 Provider.
    #
    # ConfigStore 的构造点 (Project.configs 只 force_fetch 本 provider 的产物), 从
    # Environment 推导两层: workspace (mode-aware) + ghost home/configs (覆盖层).
    # 参数全部来自 env (mode_name / ghost_name), 因此与 mode 走同一条子进程继承通道
    # — node/cell 里 discover 到的 ghost 身份会让它们也装上同一条链.

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type[INSTANCE]]:
        yield YamlConfigStore

    def factory(self, con: IoCContainer) -> ConfigStore:
        env = Environment.discover(bootstrap=False)

        base = YamlConfigStore(
            storage=env.workspace.configs(),
            mode_name=env.mode_name,
        )
        if env.no_ghost:
            return base
        # ghost 层: 自己也是 mode-aware 的 store (mode → 通用), 只是本层没命中时
        # 递归到 workspace 层. materialize=False → get_or_create 不往 ghost home 落
        # 种子, 只有显式 save 才写. create=False → 目录缺失是正常态, 不凭空造空目录.
        return YamlConfigStore(
            storage=LocalStorage(env.ghost_home / 'configs', create=False),
            mode_name=env.mode_name,
            inherit=base,
            materialize=False,
        )

    def contract(self) -> Type[INSTANCE]:
        return ConfigStore
