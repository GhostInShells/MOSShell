from ghoshell_moss.moss.abcd.manifests import Manifests, ConfigInfo, TopicInfo, AppInfo, ContractInfo
from ghoshell_moss.moss.environment import Environment
from .configs import search_config_infos_from_package
from .contracts import search_contract_infos_from_package
from .topics import search_topic_infos_from_package


class WorkspaceManifests(Manifests):
    """
    基于 workspace 发现的各种声明.
    """

    def __init__(
            self,
            env: Environment | None = None,
    ):
        self.env = env or Environment.discover()
        self.env.bootstrap()
        self._config_infos: dict[str, ConfigInfo] | None = None
        self._contract_infos: list[ContractInfo] | None = None
        self._topic_infos: dict[str, TopicInfo] | None = None

    def apps(self) -> list[AppInfo]:
        pass

    def configs(self) -> dict[str, ConfigInfo]:
        if self._config_infos is None:
            self._config_infos = search_config_infos_from_package()
        return self._config_infos

    def topics(self) -> dict[str, TopicInfo]:
        if self._topic_infos is None:
            self._topic_infos = search_topic_infos_from_package()
        return self._topic_infos

    def contracts(self) -> list[ContractInfo]:
        if self._contract_infos is None:
            self._contract_infos = list(search_contract_infos_from_package())
        return self._contract_infos
