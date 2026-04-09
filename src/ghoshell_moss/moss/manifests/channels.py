from typing import Dict
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_moss.core.concepts.channel import Channel

__all__ = ['search_channels_from_package']

MANIFEST_CONFIG_PATH = 'MOSS.manifests.channels'


def search_channels_from_package(
        package_import_path: str = MANIFEST_CONFIG_PATH,
) -> Dict[str, Channel]:
    """
    扫描逻辑：寻找在 manifest 模块中定义的 Channel 实例。
    有重名直接覆盖, 不关心 module name.
    """
    found: Dict[str, Channel] = {}

    # 递归扫描
    for manifest in scan_package(package_import_path, max_depth=2):
        if manifest.is_package:
            continue

        # 遍历模块内的所有成员
        for name, obj in manifest.module.__dict__.items():
            # 过滤掉私有成员和不符合 ConfigType 的对象
            if name.startswith('_') or not isinstance(obj, Channel):
                continue

            # 这里的逻辑：我们认为在 manifest 包下定义的变量名即为“发现”
            # 以 attr name 作为唯一键
            found[name] = obj

    return found
