from ghoshell_moss.core.blueprint.manifests import Manifests

__all__ = ['ManifestsInspector']


class ManifestsInspector:
    """用于在 REPL 中观测 Manifest 资源的工具集"""

    def __init__(self, manifests: Manifests):
        self._manifests = manifests

    def configs(self) -> dict:
        """列出所有配置实例及其默认值。"""
        return {
            name: {"desc": info.description, "model": info.model_path}
            for name, info in self._manifests.configs().items()
        }

    def providers(self) -> list[dict]:
        """列出所有已注册的 IoC Provider。"""
        return [
            {"contract": p.name, "singleton": p.singleton, "desc": p.description}
            for p in self._manifests.providers()
        ]

    def topics(self) -> list[dict]:
        """列出环境发现的所有 Topic 及其元数据。"""
        return [
            {"name": topic_name, "type": topic_info.type, "description": topic_info.description}
            for topic_name, topic_info in self._manifests.topics().items()
        ]

    def channels(self) -> list[dict]:
        """列出环境中注册的 channels """
        return [
            {"name": name, "description": channel.description}
            for name, channel in self._manifests.channels()
        ]

    def resource_storages(self) -> list[dict]:
        """列出环境中发现的 ResourceStorageMeta 声明。"""
        items = self._manifests.resource_storage_manifests()
        return [
            {
                "scheme": item.info.storage_scheme,
                "host": item.info.storage_host,
                "description": item.info.description,
                "found_at": item.info.found_module,
            }
            for item in items
        ]

    def nuclei(self) -> list[dict]:
        """列出环境中发现的 NucleusFactory 声明。"""
        return [
            {
                "name": info.name,
                "description": info.description,
                "signal_names": info.signal_names,
                "found_at": info.found_module,
            }
            for info in self._manifests.nuclei().values()
        ]
