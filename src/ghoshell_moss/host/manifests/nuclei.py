"""
Discover NucleusFactory instances from Python packages.

NucleusMetaInfo is a lightweight frozen dataclass (similar to ProviderInfo, TopicInfo).
It describes a discovered NucleusFactory: name, description, signal_names, and where found.

This is declared for test/dev infrastructure — listing and filtering nuclei
without requiring a full Ghost runtime.
"""

from typing import Iterable

from ghoshell_moss.core.blueprint.manifests import NucleusMetaInfo
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.codex.discover import scan_package

__all__ = [
    "NucleusMetaInfo",
    "MANIFEST_NUCLEI_PATH",
    "find_nucleus_metas",
    "search_nucleus_infos",
    "match_nucleus_infos",
]

MANIFEST_NUCLEI_PATH = "MOSS.manifests.nuclei"


# -- Discovery ---------------------------------------------------------


def find_nucleus_metas(
    package_import_path: str,
) -> Iterable[tuple[str, str, str, NucleusMeta]]:
    """
    Scan a package for NucleusFactory instances.

    Yields: (module_file, module_path, attr_name, NucleusFactory)
    """
    for manifest in scan_package(package_import_path, max_depth=2):
        try:
            for name, obj in manifest.iter_members(respect_all=True):
                if isinstance(obj, NucleusMeta):
                    yield manifest.file_path, manifest.module_path, name, obj
        except Exception:
            continue


def search_nucleus_infos(
    package_import_path: str = MANIFEST_NUCLEI_PATH,
) -> dict[str, NucleusMetaInfo]:
    """
    Scan and collect NucleusFactory instances into NucleusMetaInfo dict.

    Keyed by NucleusFactory.name().
    """
    results: dict[str, NucleusMetaInfo] = {}
    for file_path, module_path, attr_name, obj in find_nucleus_metas(
        package_import_path
    ):
        info = NucleusMetaInfo(
            nucleus_meta=obj,
            found_module=f"{module_path}:{attr_name}",
            found_file=file_path,
        )
        results[obj.name()] = info
    return results


def match_nucleus_infos(
    metas: dict[str, NucleusMetaInfo],
    search: str,
) -> list[NucleusMetaInfo]:
    """Fuzzy match by name, description, or signal names."""
    search_lower = search.lower()
    results: list[NucleusMetaInfo] = []
    for info in metas.values():
        if (
            search_lower in info.name.lower()
            or search_lower in info.description.lower()
            or any(search_lower in s.lower() for s in info.signal_names)
        ):
            results.append(info)
    return results
