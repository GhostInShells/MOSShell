from typing import Iterable
from pathlib import Path

from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['NucleusManifest', 'search_nucleus_manifests', 'MANIFEST_NUCLEI_PATH']

MANIFEST_NUCLEI_PATH = 'MOSS.manifests.nuclei'


class NucleusManifest(ScannedManifest[NucleusMeta]):
    """NucleusMeta 实例的 Manifest 封装."""


def search_nucleus_manifests(
        package_import_path: str = MANIFEST_NUCLEI_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Iterable[Manifest[NucleusMeta]]:
    """扫描一个 Python 包, 返回所有 NucleusMeta 实例的 Manifest 封装."""
    for module_manifest in scan_package(
            package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                if isinstance(obj, NucleusMeta):
                    yield NucleusManifest(
                        name=obj.name(),
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
            yield NucleusManifest(
                name=module_manifest.module_path,
                error=e,
                found_at=Path(module_manifest.file_path),
                import_path=module_manifest.module_path,
                description=f'scan error: {e}',
            )
