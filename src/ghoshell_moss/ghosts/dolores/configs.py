from ghoshell_container import INSTANCE, IoCContainer, Provider

from ghoshell_moss.contracts.configs import YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.ghost import GhostHome
from ghoshell_moss.ghosts.dolores.constants import DOLORES_CONFIGS_DIRECTORY

__all__ = ['DoloresConfigStore', 'DoloresConfigStoreProvider']


class DoloresConfigStore(YamlConfigStore):
    """The ghost's own typed configs, rooted at ``<ghost_home>/.dolores/configs``.

    Separate from the MOSS-global ``ConfigStore``: this store belongs to this ghost instance,
    is resolved by its concrete class, and is meant to be mutated at runtime (``save``).
    """


class DoloresConfigStoreProvider(Provider[DoloresConfigStore]):
    """Locate the store through the narrow ghost-home contract — no environment discovery.

    A ghost without a home is not bound to ``GhostHome``, so the fetch fails with a clear
    configuration error instead of guessing a path that belongs to another instance.
    """

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        ghost_home = con.force_fetch(GhostHome)
        home = ghost_home.home_meta_dir() / DOLORES_CONFIGS_DIRECTORY
        home.mkdir(parents=True, exist_ok=True)
        # materialize=True: a missing config is written from the caller's default on first use;
        # runtime mutation (save / activate) is the point of this store.
        return DoloresConfigStore(LocalStorage(home), mode_name='', materialize=True)
