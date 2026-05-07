from typing_extensions import Self
from ghoshell_moss.core.blueprint.manifests import (
    Manifests, ConfigInfo, TopicInfo, ProviderInfo, CtmlVersionInfo,
)
from .configs import search_config_infos_from_package
from .providers import search_provider_infos_from_package
from .topics import search_topic_infos_from_package
from .channels import search_channels_from_package
from .primitives import search_primitives_from_package
from ghoshell_moss.host.abcd.environment import Environment
from ghoshell_moss.core.concepts.channel import Channel, ChannelName
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.ctml.versions import (
    default_moss_ctml_meta_instruction_directory,
    search_version_file_in_dir,
)

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
            ctml_versions: dict[str, CtmlVersionInfo] | None = None,
    ):
        self.root_package_name = root_package_name
        self._config_infos: dict[str, ConfigInfo] | None = None
        self._provider_infos: list[ProviderInfo] | None = None
        self._topic_infos: dict[str, TopicInfo] | None = None
        self._channels: dict[str, Channel] | None = None
        self._primitives: dict[str, Command] | None = None
        self._ctml_versions: dict[str, CtmlVersionInfo] = ctml_versions or {}

    @classmethod
    def from_environment(cls, env: Environment | None = None) -> Self:
        """
        找到环境下的声明资源.
        """
        env = env or Environment.discover()
        env.bootstrap()
        ctml_versions = cls.find_ctml_versions_from_env(env=env)
        return cls(ENVIRONMENT_MANIFESTS_ROOT_PACKAGE, ctml_versions=ctml_versions)

    @classmethod
    def find_ctml_versions_from_env(cls, env: Environment) -> dict[str, CtmlVersionInfo]:
        ctml_versions: dict[str, CtmlVersionInfo] = {}
        for file in search_version_file_in_dir(default_moss_ctml_meta_instruction_directory()):
            ctml_version = CtmlVersionInfo(file=file)
            ctml_versions[ctml_version.version] = ctml_version
        root_dir = env.ctml_prompts_dir()
        if root_dir.exists():
            for file in search_version_file_in_dir(root_dir):
                ctml_version_info = CtmlVersionInfo(file=file)
                ctml_versions[ctml_version_info.version] = ctml_version_info
        return ctml_versions

    @classmethod
    def from_environment_moss_mode(cls, mode: str, env: Environment | None = None) -> Self:
        """
        找到模式下的声明资源.
        """
        env = env or Environment.discover()
        env.bootstrap()
        root_package_name = ENVIRONMENT_MODE_MANIFESTS_ROOT_PACKAGE.format(mode=mode)
        ctml_versions = cls.find_ctml_versions_from_env(env=env)
        return cls(root_package_name, ctml_versions=ctml_versions)

    def channels(self) -> dict[str, Channel]:
        if self._channels is None:
            channels_package = '.'.join([self.root_package_name, 'channels'])
            self._channels = search_channels_from_package(channels_package)
        return self._channels

    def ctml_versions(self) -> dict[str, CtmlVersionInfo]:
        return self._ctml_versions

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

    def providers(self) -> list[ProviderInfo]:
        if self._provider_infos is None:
            providers_package = '.'.join([self.root_package_name, 'providers'])
            self._provider_infos = list(search_provider_infos_from_package(providers_package))
        return self._provider_infos


class MergedManifests(Manifests):
    """
    合并多个 manifests. 通常是右边优先级高.
    """

    def __init__(self, manifests: list[Manifests]):
        self._config_infos: dict[str, ConfigInfo] = {}
        self._contract_infos: list[ProviderInfo] = []
        self._topic_infos: dict[str, TopicInfo] = {}
        self._channels: dict[str, Channel] = {}
        self._primitives: dict[str, Command] = {}
        self._ctml_versions: dict[str, CtmlVersionInfo] = {}
        for manifest in manifests:
            # 右边优先级更高.
            self._config_infos.update(manifest.configs())
            self._contract_infos.extend(manifest.providers())
            self._topic_infos.update(manifest.topics())
            self._channels.update(manifest.channels())
            self._primitives.update(manifest.primitives())
            self._ctml_versions.update(manifest.ctml_versions())

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

    def ctml_versions(self) -> dict[str, CtmlVersionInfo]:
        return self._ctml_versions

    def primitives(self) -> dict[str, Command]:
        return self._primitives

    def configs(self) -> dict[str, ConfigInfo]:
        return self._config_infos

    def topics(self) -> dict[str, TopicInfo]:
        return self._topic_infos

    def providers(self) -> list[ProviderInfo]:
        return self._contract_infos
