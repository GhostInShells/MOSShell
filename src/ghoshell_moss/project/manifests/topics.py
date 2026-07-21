import inspect
from typing import Iterable
from pathlib import Path

from ghoshell_moss.core.concepts.topic import TopicModel, TopicSchema
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['TopicManifest', 'search_topic_manifests', 'MANIFEST_TOPICS_PATH']

MANIFEST_TOPICS_PATH = 'MOSS.manifests.topics'


class TopicManifest(ScannedManifest[TopicSchema]):
    """Topic 的 Manifest 封装 — 扫描 TopicModel 子类或 TopicSchema 实例."""


def search_topic_manifests(
        package_import_path: str = MANIFEST_TOPICS_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Iterable[Manifest[TopicSchema]]:
    """扫描一个 Python 包, 返回 TopicModel 子类或 TopicSchema 实例的 Manifest 封装."""
    for module_manifest in scan_package(
        package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                m = _to_topic_manifest(name, obj, module_manifest.module_path, module_manifest.file_path)
                if m is not None:
                    yield m
        except Exception as e:
            if strict:
                raise
            if errors is not None:
                errors.append(ScanError(
                    module_path=module_manifest.module_path, exception=e, stage="iterate",
                ))
            yield TopicManifest(
                name=module_manifest.module_path,
                error=e,
                found_at=Path(module_manifest.file_path),
                import_path=module_manifest.module_path,
                description=f'scan error: {e}',
            )


def _to_topic_manifest(
        attr_name: str,
        obj: object,
        module_path: str,
        file_path: str,
) -> TopicManifest | None:
    # TopicModel 子类 → topic_schema()
    if inspect.isclass(obj) and issubclass(obj, TopicModel) and not inspect.isabstract(obj):
        schema = obj.topic_schema()
        return TopicManifest(
            name=schema.topic_name,
            value=schema,
            found_at=Path(file_path),
            import_path=f'{module_path}:{attr_name}',
            description=obj.__doc__ or '',
        )

    # TopicSchema 实例
    if isinstance(obj, TopicSchema):
        return TopicManifest(
            name=obj.topic_name,
            value=obj,
            found_at=Path(file_path),
            import_path=f'{module_path}:{attr_name}',
            description=obj.description or '',
        )

    return None
