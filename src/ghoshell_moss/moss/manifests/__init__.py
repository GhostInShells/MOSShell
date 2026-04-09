from typing_extensions import Self
from ghoshell_moss.moss.abcd.manifests import Manifests, ConfigInfo, TopicInfo, ContractInfo
from .configs import search_config_infos_from_package
from .contracts import search_contract_infos_from_package
from .topics import search_topic_infos_from_package
from .channels import search_channels_from_package
from .primitives import search_primitives_from_package
from ghoshell_moss.moss.environment import Environment
from ghoshell_moss.core.concepts.channel import Channel, ChannelName
from ghoshell_moss.core.concepts.command import Command

__all__ = ['PackageManifests', 'MergedManifests']

ENVIRONMENT_MANIFESTS_ROOT_PACKAGE = 'MOSS.manifests'
ENVIRONMENT_MODE_MANIFESTS_ROOT_PACKAGE = 'MOSS.modes.{mode_name}'


class PackageManifests(Manifests):
    """
    基于 workspace 发现的各种声明.
    """

    def __init__(
            self,
            root_package_name: str,
    ):
        self.root_package_name = root_package_name
        self._config_infos: dict[str, ConfigInfo] | None = None
        self._contract_infos: list[ContractInfo] | None = None
        self._topic_infos: dict[str, TopicInfo] | None = None
        self._channels: dict[str, Channel] | None = None
        self._primitives: dict[str, Command] | None = None

    @classmethod
    def from_environment(cls, env: Environment | None = None) -> Self:
        """
        找到环境下的声明资源.
        """
        env = env or Environment.discover()
        env.bootstrap()
        return cls(ENVIRONMENT_MANIFESTS_ROOT_PACKAGE)

    @classmethod
    def from_environment_moss_mode(cls, mode: str, env: Environment | None = None) -> Self:
        """
        找到模式下的声明资源.
        """
        env = env or Environment.discover()
        env.bootstrap()
        root_package_name = ENVIRONMENT_MODE_MANIFESTS_ROOT_PACKAGE.format(mode=mode)
        return cls(root_package_name)

    def channels(self) -> dict[str, Channel]:
        if self._channels is None:
            channels_package = '.'.join([self.root_package_name, 'channels'])
            self._channels = search_channels_from_package(channels_package)
        return self._channels

    def primitives(self) -> dict[str, Command]:
        """
        find moss shell primitive in the package.
        """
        if self._primitives is None:
            primitives_package = '.'.join([self.root_package_name, 'primitives'])
            self._primitives = search_primitives_from_package(primitives_package)
        return self._primitives

    def configs(self) -> dict[str, ConfigInfo]:
        if self._config_infos is None:
            configs_package = '.'.join([self.root_package_name, 'configs'])
            self._config_infos = search_config_infos_from_package(configs_package)
        return self._config_infos

    def topics(self) -> dict[str, TopicInfo]:
        if self._topic_infos is None:
            topics_package = '.'.join([self.root_package_name, 'topics'])
            self._topic_infos = search_topic_infos_from_package(topics_package)
        return self._topic_infos

    def contracts(self) -> list[ContractInfo]:
        if self._contract_infos is None:
            contracts_package = '.'.join([self.root_package_name, 'contracts'])
            self._contract_infos = list(search_contract_infos_from_package(contracts_package))
        return self._contract_infos


class MergedManifests(Manifests):
    """
    合并多个 manifests. 通常是右边优先级高.
    """

    def __init__(self, manifests: list[Manifests]):
        self._config_infos: dict[str, ConfigInfo] = {}
        self._contract_infos: list[ContractInfo] = []
        self._topic_infos: dict[str, TopicInfo] = {}
        self._channels: dict[str, Channel] = {}
        self._primitives: dict[str, Command] = {}
        for manifest in manifests:
            # 右边优先级更高.
            self._config_infos.update(manifest.configs())
            self._contract_infos.extend(manifest.contracts())
            self._topic_infos.update(manifest.topics())
            self._channels.update(manifest.channels())
            self._primitives.update(manifest.primitives())

    @classmethod
    def from_environment_mode(cls, *, mode: str = '', env: Environment | None = None) -> Manifests:
        """
        默认根据模式来生成.
        """
        env = env or Environment.discover()
        env.bootstrap()
        env_manifests = PackageManifests.from_environment(env)
        if mode:
            mode_manifests = PackageManifests.from_environment_moss_mode(mode, env)
            return cls([env_manifests, mode_manifests])
        return env_manifests

    def channels(self) -> dict[ChannelName, Channel]:
        return self._channels

    def primitives(self) -> dict[str, Command]:
        return self._primitives

    def configs(self) -> dict[str, ConfigInfo]:
        return self._config_infos

    def topics(self) -> dict[str, TopicInfo]:
        return self._topic_infos

    def contracts(self) -> list[ContractInfo]:
        return self._contract_infos
