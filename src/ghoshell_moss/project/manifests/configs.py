import inspect
from typing import Iterable
from pathlib import Path

from ghoshell_moss.contracts.configs import ConfigType, ConfigSchema
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['ConfigManifest', 'search_config_manifests', 'MANIFEST_CONFIGS_PATH']

MANIFEST_CONFIGS_PATH = 'MOSS.manifests.configs'


class ConfigManifest(ScannedManifest[ConfigType]):
    """ConfigType 实例的 Manifest 封装.

    value() 返回 ConfigType 实例.
    schema() 返回 json_schema 等结构信息.
    """

    def schema(self) -> ConfigSchema:
        return self._value.to_config_schema()


def search_config_manifests(
        package_import_path: str = MANIFEST_CONFIGS_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Iterable[Manifest[ConfigType]]:
    """扫描一个 Python 包, 返回所有 ConfigType 实例的 Manifest 封装."""
    for module_manifest in scan_package(
        package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                if isinstance(obj, ConfigType):
                    yield ConfigManifest(
                        name=obj.conf_name(),
                        value=obj,
                        found_at=Path(module_manifest.file_path),
                        import_path=f'{module_manifest.module_path}:{name}',
                        description=type(obj).__doc__ or '',
                    )
        except Exception as e:
            if strict:
                raise
            if errors is not None:
                errors.append(ScanError(
                    module_path=module_manifest.module_path, exception=e, stage="iterate",
                ))
            yield ConfigManifest(
                name=module_manifest.module_path,
                error=e,
                found_at=Path(module_manifest.file_path),
                import_path=module_manifest.module_path,
                description=f'scan error: {e}',
            )
