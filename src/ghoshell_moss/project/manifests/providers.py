import inspect
from typing import Iterable
from pathlib import Path

from ghoshell_container import Provider, IoCContainer
from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['ProviderManifest', 'search_provider_manifests', 'MANIFEST_PROVIDERS_PATH']

MANIFEST_PROVIDERS_PATH = 'MOSS.manifests.providers'


class ProviderManifest(ScannedManifest[Provider]):
    """Provider 的 Manifest 封装. 唯一覆盖 update_container 的 Manifest 子类型."""

    def update_container(self, container: IoCContainer) -> None:
        if self.is_error():
            return
        container.register(self._value)


def search_provider_manifests(
        package_import_path: str = MANIFEST_PROVIDERS_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Iterable[Manifest[Provider]]:
    """扫描一个 Python 包，返回所有 Provider 的 Manifest 封装."""
    found: set[int] = set()  # id 去重

    for module_manifest in scan_package(
        package_import_path, max_depth=2, strict=strict, errors=errors,
    ):
        try:
            for name, obj in module_manifest.iter_members(respect_all=True):
                if not isinstance(obj, Provider):
                    continue
                if id(obj) in found:
                    continue
                found.add(id(obj))
                yield _provider_to_manifest(
                    obj,
                    module_path=module_manifest.module_path,
                    file_path=Path(module_manifest.file_path),
                    attr_name=name,
                )
        except Exception as e:
            if strict:
                raise
            if errors is not None:
                errors.append(ScanError(
                    module_path=module_manifest.module_path, exception=e, stage="iterate",
                ))
            # 不静默 — 返回一个 is_error() 为 True 的 Manifest
            yield ProviderManifest(
                name=module_manifest.module_path,
                error=e,
                found_at=Path(module_manifest.file_path),
                import_path=module_manifest.module_path,
                description=f'scan error: {e}',
            )


def _provider_to_manifest(
        provider: Provider,
        *,
        module_path: str,
        file_path: Path,
        attr_name: str,
) -> ProviderManifest:
    contract = provider.contract()
    name = _import_path(contract) if inspect.isclass(contract) else str(contract)
    description = inspect.getdoc(contract) or ''
    import_path = f'{module_path}:{attr_name}'
    try:
        source = inspect.getsource(contract)
    except (TypeError, OSError):
        source = ''

    return ProviderManifest(
        name=name,
        value=provider,
        found_at=file_path,
        import_path=import_path,
        description=description.split('\n')[0] if description else '',
        source=source,
    )


def _import_path(cls: type) -> str:
    module = getattr(cls, '__module__', '')
    qualname = getattr(cls, '__qualname__', cls.__name__)
    if module:
        return f'{module}.{qualname}'
    return qualname
