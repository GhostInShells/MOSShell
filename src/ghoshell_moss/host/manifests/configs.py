import inspect
from typing import Dict
from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.core.blueprint.manifests import ConfigInfo

__all__ = ['search_config_infos_from_package', 'ConfigInfo', 'MANIFEST_CONFIG_PATH']

MANIFEST_CONFIG_PATH = 'MOSS.manifests.configs'


def search_config_infos_from_package(
        package_import_path: str = MANIFEST_CONFIG_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Dict[str, ConfigInfo]:
    """
    扫描逻辑：寻找在 manifest 模块中定义的 ConfigType 子类（类型注册）和 ConfigType 实例（值覆盖）。

    两种注册语义：

    1. Type[ConfigType] (类本身) → is_override=False
       系统默认配置。Bootstrap 时 get_or_create() — 读 YAML，不存在才写默认值。
       文件优先的持久化配置。

    2. ConfigType 实例 (带值) → is_override=True
       运行时覆盖。Bootstrap 时仅 set_config(override=False) — 写内存缓存，
       绝不触碰 YAML 文件。Mode 级别的覆盖使用此语义。
    """
    configs: Dict[str, ConfigInfo] = {}

    # 递归扫描
    for manifest in scan_package(package_import_path, max_depth=2, strict=strict, errors=errors):

        try:
            # TODO: 遍历模块内的所有成员
            for name, obj in manifest.module.__dict__.items():
                # 过滤掉私有成员
                if name.startswith('_'):
                    continue

                # 1. Type[ConfigType] — 类型注册（系统默认）
                if inspect.isclass(obj) and issubclass(obj, ConfigType) and not inspect.isabstract(obj):
                    default_instance = obj()
                    info = ConfigInfo(
                        found_import_path=manifest.module_path,
                        found_at_file=manifest.file_path,
                        config=default_instance,
                        is_override=False,
                    )
                    configs[info.name] = info

                # 2. ConfigType 实例 — 运行时覆盖（mode 级）
                elif isinstance(obj, ConfigType):
                    info = ConfigInfo(
                        found_import_path=manifest.module_path,
                        found_at_file=manifest.file_path,
                        config=obj,
                        is_override=True,
                    )
                    configs[info.name] = info

        except Exception as e:
            if strict:
                raise
            if errors is not None:
                errors.append(ScanError(module_path=manifest.module_path, exception=e, stage="iterate"))
            continue

    return configs
