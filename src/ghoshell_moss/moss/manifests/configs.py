from typing import Dict
from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_moss.moss.concepts.manifests import ConfigInfo

__all__ = ['search_config_infos_from_package', 'ConfigInfo', 'MANIFEST_CONFIG_PATH']

MANIFEST_CONFIG_PATH = 'MOSS.manifests.configs'


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
