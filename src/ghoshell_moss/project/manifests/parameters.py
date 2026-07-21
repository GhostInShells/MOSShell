import inspect
from pathlib import Path

from ghoshell_moss.core.blueprint.parameter import ParameterModel, ParameterSchema
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['ParameterManifest', 'search_parameter_manifests', 'MANIFEST_PARAMETERS_PATH']

MANIFEST_PARAMETERS_PATH = 'MOSS.manifests.parameters'


class ParameterManifest(ScannedManifest[ParameterSchema]):
    """Parameter 的 Manifest 封装 — 扫描 ParameterModel 子类或 ParameterSchema 实例."""


def search_parameter_manifests(
        package_import_path: str = MANIFEST_PARAMETERS_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Manifest[ParameterSchema]:
    """扫描一个 Python 包, 返回第一个 ParameterModel 子类或 ParameterSchema 实例.

    多个参数时取第一个; 没有时返回 error manifest.
    """
    first_error: Exception | None = None
    for module_manifest in scan_package(
        package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                m = _to_parameter_manifest(name, obj, module_manifest.module_path, module_manifest.file_path)
                if m is not None:
                    return m
        except Exception as e:
            if strict:
                raise
            if errors is not None:
                errors.append(ScanError(
                    module_path=module_manifest.module_path, exception=e, stage="iterate",
                ))
            if first_error is None:
                first_error = e

    return ParameterManifest(
        name=package_import_path,
        error=first_error or LookupError(f'no parameter found in {package_import_path}'),
        found_at=Path('.'),
        import_path=package_import_path,
        description='no parameter manifest found',
    )


def _to_parameter_manifest(
        attr_name: str,
        obj: object,
        module_path: str,
        file_path: str,
) -> ParameterManifest | None:
    # ParameterModel 子类 → to_parameter_schema()
    if inspect.isclass(obj) and issubclass(obj, ParameterModel) and not inspect.isabstract(obj):
        schema = obj.to_parameter_schema()
        return ParameterManifest(
            name=schema.name,
            value=schema,
            found_at=Path(file_path),
            import_path=f'{module_path}:{attr_name}',
            description=obj.__doc__ or '',
        )

    # ParameterSchema 实例
    if isinstance(obj, ParameterSchema):
        return ParameterManifest(
            name=obj.name,
            value=obj,
            found_at=Path(file_path),
            import_path=f'{module_path}:{attr_name}',
            description=obj.description or '',
        )

    return None
