from typing import Iterable
from pathlib import Path

from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.blueprint.environment import GHOST_MANIFESTS_PACKAGE
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['GhostManifest', 'search_ghost_manifests']


class GhostManifest(ScannedManifest[GhostMeta]):
    """GhostMeta 实例的 Manifest 封装."""


def search_ghost_manifests(
        package_import_path: str = GHOST_MANIFESTS_PACKAGE,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Iterable[Manifest[GhostMeta]]:
    found: set[int] = set()

    for module_manifest in scan_package(
            package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                if not isinstance(obj, GhostMeta):
                    continue
                if id(obj) in found:
                    continue
                found.add(id(obj))
                yield GhostManifest(
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
            yield GhostManifest(
                name=module_manifest.module_path,
                error=e,
                found_at=Path(module_manifest.file_path),
                import_path=module_manifest.module_path,
                description=f'scan error: {e}',
            )
