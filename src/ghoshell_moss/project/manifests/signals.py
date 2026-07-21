import inspect
from typing import Iterable
from pathlib import Path

from ghoshell_moss.core.blueprint.mindflow import SignalMeta, SignalSchema
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['SignalManifest', 'search_signal_manifests', 'MANIFEST_SIGNALS_PATH']

MANIFEST_SIGNALS_PATH = 'MOSS.manifests.signals'


class SignalManifest(ScannedManifest[SignalSchema]):
    """Signal 的 Manifest 封装 — 扫描 SignalMeta 子类或 SignalSchema 实例.

    - SignalMeta 子类: 调用 to_signal_schema() 转为 SignalSchema, source() 返回类源码
    - SignalSchema 实例: 直接使用, source() 返回空字符串
    """


def search_signal_manifests(
        package_import_path: str = MANIFEST_SIGNALS_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Iterable[Manifest[SignalSchema]]:
    """扫描一个 Python 包, 返回 SignalMeta 子类或 SignalSchema 实例的 Manifest 封装."""
    for module_manifest in scan_package(
        package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                m = _to_signal_manifest(name, obj, module_manifest.module_path, module_manifest.file_path)
                if m is not None:
                    yield m
        except Exception as e:
            if strict:
                raise
            if errors is not None:
                errors.append(ScanError(
                    module_path=module_manifest.module_path, exception=e, stage="iterate",
                ))
            yield SignalManifest(
                name=module_manifest.module_path,
                error=e,
                found_at=Path(module_manifest.file_path),
                import_path=module_manifest.module_path,
                description=f'scan error: {e}',
            )


def _to_signal_manifest(
        attr_name: str,
        obj: object,
        module_path: str,
        file_path: str,
) -> SignalManifest | None:
    # SignalMeta 子类 → 转为 SignalSchema
    if inspect.isclass(obj) and issubclass(obj, SignalMeta) and not inspect.isabstract(obj):
        schema = obj.to_signal_schema()
        try:
            source = inspect.getsource(obj)
        except (TypeError, OSError):
            source = ''
        return SignalManifest(
            name=schema.name,
            value=schema,
            found_at=Path(file_path),
            import_path=f'{module_path}:{attr_name}',
            description=obj.__doc__ or '',
            source=source,
        )

    # SignalSchema 实例
    if isinstance(obj, SignalSchema):
        return SignalManifest(
            name=obj.name,
            value=obj,
            found_at=Path(file_path),
            import_path=f'{module_path}:{attr_name}',
            description=obj.description or '',
            source='',
        )

    return None
