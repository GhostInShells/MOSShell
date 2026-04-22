from typing import Type, Iterable

from ghoshell_container import IoCContainer, BootstrapProvider, INSTANCE
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.contracts.configs import ConfigStore, YamlConfigStore
from ghoshell_moss.host.abcd.manifests import Manifest

__all__ = [
    'HostEnvConfigStoreProvider',
]


class HostEnvConfigStoreProvider(BootstrapProvider):

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

    def bootstrap(self, container: IoCContainer) -> None:
        this = container.force_fetch(ConfigStore)
        manifest = container.get(Manifest)
        if manifest:
            for config_info in manifest.configs().values():
                this.get_or_create(config_info.config)
