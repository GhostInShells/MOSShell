from typing_extensions import Self
from ghoshell_moss.core.blueprint.manifests import (
    Manifests, ConfigInfo, TopicInfo, ProviderInfo, CtmlVersionInfo,
    ResourceStorageItem, NucleusMetaInfo,
)
from ghoshell_moss.core.codex.discover import ScanError
from .configs import search_config_infos_from_package
from .providers import search_provider_infos_from_package
from .topics import search_topic_infos_from_package
from .channels import search_channels_from_package, search_main_channel_from_manifest
from .resource_storages import PackageResourceStorages
from .nuclei import search_nucleus_infos
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.concepts.channel import Channel, ChannelName
from ghoshell_moss.core.ctml.versions import (
    default_moss_ctml_meta_instruction_directory,
    search_version_file_in_dir,
)

__all__ = ['PackageManifests', 'MergedManifests']

ENVIRONMENT_MANIFESTS_ROOT_PACKAGE = 'MOSS.manifests'


class PackageManifests(Manifests):
    """
    基于 workspace 发现的各种声明.
    """

    def __init__(
            self,
            root_package_name: str,
            ctml_versions: dict[str, CtmlVersionInfo] | None = None,
            *,
            strict: bool = False,
            errors: list[ScanError] | None = None,
    ):
        self.root_package_name = root_package_name
        self._strict = strict
        self._errors = errors
        self._config_infos: dict[str, ConfigInfo] | None = None
        self._provider_infos: list[ProviderInfo] | None = None
        self._topic_infos: dict[str, TopicInfo] | None = None
        self._channels: dict[str, Channel] | None = None
        self._main_found_module: str | None = None
        self._ctml_versions: dict[str, CtmlVersionInfo] = ctml_versions or {}
        self._resource_storages: PackageResourceStorages | None = None
        self._nuclei: dict[str, NucleusMetaInfo] | None = None

    @property
    def scan_errors(self) -> list[ScanError]:
        """Errors collected during the last scan, if an error collector was provided."""
        return self._errors or []

    @classmethod
    def from_environment(
        cls, env: Environment | None = None,
        *, strict: bool = False, errors: list[ScanError] | None = None,
    ) -> Self:
        """
        找到环境下的声明资源.
        """
        env = env or Environment.discover()
        env.seal()
        ctml_versions = cls.find_ctml_versions_from_env(env=env)
        return cls(ENVIRONMENT_MANIFESTS_ROOT_PACKAGE, ctml_versions=ctml_versions,
                   strict=strict, errors=errors)

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

    def channels(self) -> dict[str, Channel]:
        if self._channels is None:
            channels_package = '.'.join([self.root_package_name, 'channels'])
            found = search_main_channel_from_manifest(
                channels_package, strict=self._strict, errors=self._errors,
            )
            if found is not None:
                channel, found_module = found
                self._channels = {Channel.MAIN_CHANNEL_NAME: channel}
                self._main_found_module = found_module
            else:
                self._channels = {}
        return self._channels

    def main_channel_source(self) -> str | None:
        """返回 __main__ channel 被发现时的 Python 模块路径，未找到则 None。"""
        self.channels()  # 触发搜索
        return self._main_found_module

    def ctml_versions(self) -> dict[str, CtmlVersionInfo]:
        return self._ctml_versions

    def configs(self) -> dict[str, ConfigInfo]:
        if self._config_infos is None:
            configs_package = '.'.join([self.root_package_name, 'configs'])
            self._config_infos = search_config_infos_from_package(
                configs_package, strict=self._strict, errors=self._errors,
            )
        return self._config_infos

    def topics(self) -> dict[str, TopicInfo]:
        if self._topic_infos is None:
            topics_package = '.'.join([self.root_package_name, 'topics'])
            self._topic_infos = search_topic_infos_from_package(
                topics_package, strict=self._strict, errors=self._errors,
            )
        return self._topic_infos

    def providers(self) -> list[ProviderInfo]:
        if self._provider_infos is None:
            providers_package = '.'.join([self.root_package_name, 'providers'])
            self._provider_infos = list(search_provider_infos_from_package(
                providers_package, strict=self._strict, errors=self._errors,
            ))
        return self._provider_infos

    def resource_storages(self) -> PackageResourceStorages:
        if self._resource_storages is None:
            resources_package = '.'.join([self.root_package_name, 'resources'])
            self._resource_storages = PackageResourceStorages(
                resources_package, strict=self._strict, errors=self._errors,
            )
        return self._resource_storages

    NUCLEI_SUB_PACKAGE = 'nuclei'

    def nuclei(self) -> dict[str, NucleusMetaInfo]:
        if self._nuclei is None:
            nuclei_package = '.'.join([self.root_package_name, self.NUCLEI_SUB_PACKAGE])
            self._nuclei = search_nucleus_infos(
                nuclei_package, strict=self._strict, errors=self._errors,
            )
        return self._nuclei

    def resource_storage_manifests(self) -> list[ResourceStorageItem]:
        items = []
        for meta_info in self.resource_storages().list_metas_sync(limit=-1):
            item = self.resource_storages().get_sync(meta_info.path)
            items.append(item)
        return items

    def explain(self) -> str:
        return (
            f"此清单扫描自 workspace 中的 Python 包 `{self.root_package_name}` "
            f"及其子模块。\n\n"
            "### 目录约定\n\n"
            f"workspace 的 `src/{self.root_package_name.replace('.', '/')}/` 目录下，\n"
            "按子模块名约定各类声明：\n"
            "- `channels.py` — 定义 `__main__` channel（FastAPI-like 入口），原语也在此注册\n"
            "- `providers.py` — 定义 IoC Provider\n"
            "- `configs.py` — 定义配置模型\n"
            "- `topics.py` — 定义事件协议\n"
            "- `resources.py` — 定义资源存储\n"
            "- `nuclei.py` — 定义感知核\n\n"
            "### Channels 发现规则\n\n"
            "扫描 `channels` 子模块，寻找 name == '__main__' 的 Channel 实例。\n"
            "若存在，整个对象作为 CTML shell 的主 channel，"
            "所有 import_channels / with_state / with_module 组合在定义时已完成。\n"
            "若不存在，MossRuntime 使用空白默认 main channel。"
        )


class MergedManifests(Manifests):
    """
    Deprecated: manifests 统一采用显式继承模式（每个 mode 的 manifest 文件
    通过 ``from MOSS.manifests.xxx import *`` + 扩展实现，不再需要隐式合并）。

    保留此类仅为代码引用和未来参考，当前 Host 不再使用。
    """

    def __init__(self, manifests: list[Manifests]):
        self._manifests_list = manifests
        self._config_infos: dict[str, ConfigInfo] = {}
        self._contract_infos: list[ProviderInfo] = []
        self._topic_infos: dict[str, TopicInfo] = {}
        self._channels: dict[str, Channel] = {}
        self._main_channel_source: str | None = None
        self._ctml_versions: dict[str, CtmlVersionInfo] = {}
        self._resource_storages: PackageResourceStorages = PackageResourceStorages("")
        self._nuclei: dict[str, NucleusMetaInfo] = {}
        for manifest in manifests:
            # 右边优先级更高.
            self._config_infos.update(manifest.configs())
            self._contract_infos.extend(manifest.providers())
            self._topic_infos.update(manifest.topics())
            # channel: mode 的 __main__ 完全覆盖全局 (K5)
            manifest_channels = manifest.channels()
            if "__main__" in manifest_channels:
                self._channels = {"__main__": manifest_channels["__main__"]}
                if isinstance(manifest, PackageManifests):
                    self._main_channel_source = manifest.main_channel_source()
            self._ctml_versions.update(manifest.ctml_versions())
            # merge resource storages (右边优先)
            # 屏蔽 storage 实现.
            for item in manifest.resource_storage_manifests():
                self._resource_storages.add_item(item)
            # merge nuclei (右边优先)
            self._nuclei.update(manifest.nuclei())

    # from_environment_mode() removed — deprecated class, no longer used by Host.

    def channels(self) -> dict[ChannelName, Channel]:
        return self._channels

    def main_channel_source(self) -> str | None:
        """返回最终生效的 __main__ channel 的发现位置（Python 模块路径），未找到则 None。"""
        return self._main_channel_source

    def ctml_versions(self) -> dict[str, CtmlVersionInfo]:
        return self._ctml_versions

    def configs(self) -> dict[str, ConfigInfo]:
        return self._config_infos

    def topics(self) -> dict[str, TopicInfo]:
        return self._topic_infos

    def providers(self) -> list[ProviderInfo]:
        return self._contract_infos

    def resource_storage_manifests(self) -> list[ResourceStorageItem]:
        items = []
        for _, item in self._resource_storages.items():
            items.append(item)
        return items

    def resource_storages(self) -> PackageResourceStorages:
        return self._resource_storages

    def nuclei(self) -> dict[str, NucleusMetaInfo]:
        return self._nuclei

    def explain(self) -> str:
        parts = [super().explain()]
        for i, m in enumerate(self._manifests_list):
            parts.append(f"### 源 {i + 1}: {type(m).__name__}\n{m.explain()}")
        parts.append(
            "### 合并规则\n\n"
            f"以上 {len(self._manifests_list)} 个源按序合并（先全局，后 mode），右边覆盖左边。\n"
            "- Channels：mode 若定义了 `__main__`，完全替代全局的 main channel。\n"
            "- 其余类型（providers/configs/topics/nuclei/resources）："
            "dict.update / list.extend 叠加。\n"
            "- 不同类别之间不互相影响。"
        )
        return "\n\n".join(parts)
