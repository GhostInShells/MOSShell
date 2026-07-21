from pathlib import Path

from ghoshell_moss.core.blueprint.project import Manifest
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.codex.discover import scan_package, ScanError
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['ChannelManifest', 'search_main_channel', 'MANIFEST_CHANNELS_PATH']

MANIFEST_CHANNELS_PATH = 'MOSS.manifests.channels'


class ChannelManifest(ScannedManifest[PrimeChannel]):
    """Main channel 的 Manifest 封装."""


def search_main_channel(
        package_import_path: str = MANIFEST_CHANNELS_PATH,
        *,
        strict: bool = False,
        errors: list[ScanError] | None = None,
) -> Manifest[PrimeChannel]:
    """扫描包中 ``name() == '__main__'`` 的 Channel，返回第一个匹配.

    无匹配或包不存在时返回 error manifest (不抛异常).
    """
    first_error: Exception | None = None
    try:
        for module_manifest in scan_package(
                package_import_path, max_depth=2, strict=strict, errors=errors,
        ):
            try:
                for name, obj in module_manifest.iter_members(respect_all=True):
                    if isinstance(obj, Channel) and obj.name() == '__main__':
                        return ChannelManifest(
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
                if first_error is None:
                    first_error = e
    except Exception as e:
        # 包不存在等扫描级错误
        if strict:
            raise
        first_error = e

    return ChannelManifest(
        name=package_import_path,
        error=first_error or LookupError(f'no __main__ channel found in {package_import_path}'),
        found_at=Path('.'),
        import_path=package_import_path,
        description='no main channel found',
    )
