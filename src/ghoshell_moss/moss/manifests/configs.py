from typing import Iterable, Dict, Any
from dataclasses import dataclass
from ghoshell_moss.contracts.configs import ConfigType, ConfigSchema
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_common.helpers import generate_import_path
import inspect

MANIFEST_CONFIG_PATH = 'MOSS.manifests.configs'


@dataclass
class ConfigInfo:
    """
    Configuration model information
    """
    found: str  # 发现 config 的 module name, 如 MOSS.manifests.topics
    file: str  # 发现 config 的 module filename
    config: ConfigType  # config 是一个实例, 一定要有默认值. 真实的值会被 config store 以 yaml 保存到目录里. 不过那是运行时配置.

    @property
    def schema(self) -> ConfigSchema:
        return self.config.to_config_schema()

    @property
    def name(self) -> str:
        return self.config.conf_name()

    @property
    def source(self) -> str:
        return inspect.getsource(type(self.config))

    @property
    def model_path(self) -> str:
        return generate_import_path(type(self.config))

    @property
    def description(self) -> str:
        return self.config.to_config_schema().description

    def default_value(self) -> dict[str, Any]:
        return self.config.model_dump()

    def dump_yaml(self) -> str:
        return self.config.to_yaml()


def is_config(name: str, value: Any) -> bool:
    return isinstance(value, ConfigType)


def search_config_infos_from_package(
        package_import_path: str = MANIFEST_CONFIG_PATH,
) -> Dict[str, ConfigInfo]:
    """
    扫描逻辑：寻找在 manifest 模块中定义的 ConfigType 实例。
    """
    configs: Dict[str, ConfigInfo] = {}

    # 递归扫描
    for manifest in scan_package(package_import_path, max_depth=2):
        if manifest.is_package:
            continue

        # 遍历模块内的所有成员
        for name, obj in manifest.module.__dict__.items():
            # 过滤掉私有成员和不符合 ConfigType 的对象
            if name.startswith('_') or not isinstance(obj, ConfigType):
                continue

            # 这里的逻辑：我们认为在 manifest 包下定义的变量名即为“发现”
            info = ConfigInfo(
                found=manifest.module_path,
                file=manifest.file_path,
                config=obj
            )

            # 以 conf_name 作为唯一键
            configs[info.name] = info

    return configs
