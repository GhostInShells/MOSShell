from typing import Iterable
from pathlib import Path

from ghoshell_moss.contracts.resource import ResourceStorageMeta
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['ResourceManifest', 'search_resource_manifests', 'MANIFEST_RESOURCES_PATH']

MANIFEST_RESOURCES_PATH = 'MOSS.manifests.resources'


class ResourceManifest(ScannedManifest[ResourceStorageMeta]):
    """ResourceStorageMeta 实例的 Manifest 封装."""


def search_resource_manifests(
        package_import_path: str = MANIFEST_RESOURCES_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Iterable[Manifest[ResourceStorageMeta]]:
    """扫描一个 Python 包, 返回所有 ResourceStorageMeta 实例的 Manifest 封装."""
    for module_manifest in scan_package(
            package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                if isinstance(obj, ResourceStorageMeta):
                    yield ResourceManifest(
                        name=obj.scheme(),
                        value=obj,
                        found_at=Path(module_manifest.file_path),
                        import_path=f'{module_manifest.module_path}:{name}',
                        description=obj.description(),
                    )
        except Exception as e:
            if strict:
                raise
            if errors is not None:
                errors.append(ScanError(
                    module_path=module_manifest.module_path, exception=e, stage="iterate",
                ))
            yield ResourceManifest(
                name=module_manifest.module_path,
                error=e,
                found_at=Path(module_manifest.file_path),
                import_path=module_manifest.module_path,
                description=f'scan error: {e}',
            )
