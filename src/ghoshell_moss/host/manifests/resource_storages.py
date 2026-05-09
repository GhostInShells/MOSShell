"""
Discover ResourceStorageMeta instances from Python packages as resources.

Each ResourceStorageMeta found in a package is exposed as a ResourceItem
whose meta describes the storage (scheme, host, description) and whose
get() lazily instantiates the real ResourceStorage via factory().

This is the meta-layer: resources that describe available resource storages.
"""

from typing import Iterable, Sequence

from ghoshell_moss.contracts.resource import (
    ResourceMeta,
    ResourceItem,
    ResourceStorage,
    ResourceStorageFactory,
)
from ghoshell_moss.core.blueprint.manifests import ResourceStorageManifest, ResourceStorageMetaInfo
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_container import IoCContainer
from pydantic import Field

__all__ = [
    "ResourceStorageMetaInfo",
    "ResourceStorageManifestItem",
    "PackageResourceStorages",
    "MANIFEST_RESOURCES_PATH",
    "find_resource_storage_metas",
    "search_resource_storage_metas",
    "match_resource_storage_metas",
]

MANIFEST_RESOURCES_PATH = "MOSS.manifests.resources"


# -- Meta & Item -------------------------------------------------------


class ResourceStorageManifestItem(ResourceStorageManifest):
    """Wraps a ResourceStorageMeta. get() instantiates via factory()."""

    def __init__(
            self,
            meta: ResourceStorageMetaInfo,
            storage_meta: ResourceStorageFactory,
    ) -> None:
        self._meta = meta
        self._storage_meta = storage_meta

    @classmethod
    def meta_type(cls) -> type[ResourceStorageMetaInfo]:
        return ResourceStorageMetaInfo

    @property
    def meta(self) -> ResourceStorageMetaInfo:
        return self._meta

    async def get(self) -> ResourceStorageFactory:
        return self._storage_meta

    def get_sync(self) -> ResourceStorageFactory:
        return self._storage_meta


# -- Discovery ---------------------------------------------------------


def find_resource_storage_metas(
        package_import_path: str,
) -> Iterable[tuple[str, str, str, ResourceStorageFactory]]:
    """
    Scan a package for ResourceStorageMeta instances.

    Yields: (module_file, module_path, attr_name, ResourceStorageMeta)
    """
    for manifest in scan_package(package_import_path, max_depth=2):
        try:
            for name, obj in manifest.iter_members(respect_all=True):
                if isinstance(obj, ResourceStorageFactory):
                    yield manifest.file_path, manifest.module_path, name, obj
        except Exception:
            continue


def search_resource_storage_metas(
        package_import_path: str = MANIFEST_RESOURCES_PATH,
) -> dict[str, ResourceStorageMetaInfo]:
    """
    Scan and collect ResourceStorageMeta instances into StorageMetaInfo dict.

    Keyed by path: {storage_scheme}:{storage_host}
    """
    results: dict[str, ResourceStorageMetaInfo] = {}
    for file_path, module_path, attr_name, obj in find_resource_storage_metas(
            package_import_path
    ):
        path = f"{obj.scheme()}:{obj.host}"
        info = ResourceStorageMetaInfo(
            host=package_import_path,
            path=path,
            description=obj.description(),
            found_module=f"{module_path}:{attr_name}",
            found_file=file_path,
            storage_scheme=obj.scheme(),
            storage_host=obj.host,
        )
        results[path] = info
    return results


def match_resource_storage_metas(
        metas: dict[str, ResourceStorageMetaInfo],
        search: str,
) -> list[ResourceStorageMetaInfo]:
    """Fuzzy match by path, storage_scheme, storage_host, or description."""
    search_lower = search.lower()
    results: list[ResourceStorageMetaInfo] = []
    for info in metas.values():
        if (
                search_lower in info.path.lower()
                or search_lower in info.storage_scheme.lower()
                or search_lower in info.storage_host.lower()
                or search_lower in info.description.lower()
        ):
            results.append(info)
    return results


# -- Storage -----------------------------------------------------------


class PackageResourceStorages(ResourceStorage[ResourceStorageMetaInfo, ResourceStorage]):
    """
    ResourceStorage that aggregates ResourceStorageMeta instances from a package.

    list_metas() returns discovered StorageMetaInfo items.
    get(path) returns a StorageMetaItem that can instantiate the real storage.

    Usage:
        storages = PackageResourceStorages("MOSS.manifests.resources")
        storages.scan()
        for meta in await storages.list_metas():
            print(meta.locator)
    """

    def __init__(
            self,
            package_name: str,
            host: str | None = None,
    ) -> None:
        self._package_name = package_name
        self._host = host or package_name
        self._items: dict[str, ResourceStorageManifest] = {}
        self._scanned = False

    def scan(self) -> None:
        """Scan the package for ResourceStorageMeta instances."""
        if self._scanned:
            return
        for file_path, module_path, attr_name, obj in find_resource_storage_metas(
                self._package_name
        ):
            path = f"{obj.scheme()}:{obj.host}"
            info = ResourceStorageMetaInfo(
                host=self._host,
                path=path,
                description=obj.description(),
                found_module=f"{module_path}:{attr_name}",
                found_file=file_path,
                storage_scheme=obj.scheme(),
                storage_host=obj.host,
            )
            item = ResourceStorageManifestItem(
                meta=info,
                storage_meta=obj,
            )
            self._items[path] = item
        self._scanned = True

    # -- ResourceStorage interface -------------------------------------

    def add_item(self, item: ResourceStorageManifest) -> None:
        self._items[item.meta.path] = item

    @classmethod
    def scheme(cls) -> str:
        return ResourceStorageMetaInfo.scheme()

    @classmethod
    def scheme_description(cls) -> str:
        return ResourceStorageMetaInfo.scheme_description()

    @property
    def host(self) -> str:
        return self._host

    def usage(self) -> str:
        return """\
resource-storage: discover available ResourceStorage instances in the environment.

Query syntax: keyword (matches storage_scheme, storage_host, description)
  resource-storage list              → list all discovered resource storages
  resource-storage list "pil-image"  → find storages providing pil-image scheme

StorageMetaInfo fields:
  storage_scheme, storage_host, description, found_module, found_file

get(path) returns a StorageMetaItem, whose get() instantiates the real ResourceStorage.
Requires an IoC container to be set on the PackageResourceStorages."""

    async def help(self, question: str | None = None) -> str:
        if question is None:
            return self.usage()
        q = question.lower()
        if "factory" in q or "instantiate" in q:
            return (
                "Call item.get() on a StorageMetaItem to instantiate the real "
                "ResourceStorage via ResourceStorageMeta.factory(container). "
                "Requires set_container() to be called first."
            )
        if "scheme" in q:
            return (
                "Each discovered ResourceStorageMeta declares what scheme it provides "
                "via scheme(). storage_scheme in StorageMetaInfo reflects this."
            )
        return f"[resource-storage help] {question}\nUsage overview:\n{self.usage()}"

    async def list_metas(
            self, query: str | None = None, limit: int = -1
    ) -> Sequence[ResourceStorageMetaInfo]:
        return self.list_metas_sync(query, limit)

    def list_metas_sync(
            self, query: str | None = None, limit: int = -1
    ) -> Sequence[ResourceStorageMetaInfo]:
        self.scan()
        results: list[ResourceStorageMetaInfo] = []
        for item in self._items.values():
            info = item.meta
            if query and not self._match(info, query):
                continue
            results.append(info)
            if limit >= 0 and len(results) >= limit:
                break
        return results

    def items(self) -> Iterable[tuple[str, ResourceStorageManifest]]:
        yield from self._items.items()

    async def get(self, path: str) -> ResourceStorageManifest | None:
        return self.get_sync(path)

    def get_sync(self, path: str) -> ResourceStorageManifest | None:
        self.scan()
        item = self._items.get(path)
        if item is None:
            return None
        return item

    async def put(
            self, item: ResourceItem[ResourceStorageMetaInfo, ResourceStorage]
    ) -> str:
        raise NotImplementedError("PackageResourceStorages is read-only")

    async def delete(self, path: str) -> bool:
        raise NotImplementedError("PackageResourceStorages is read-only")

    # -- internal -----------------------------------------------------

    def _match(self, info: ResourceStorageMetaInfo, query: str) -> bool:
        q = query.lower()
        return (
                q in info.storage_scheme.lower()
                or q in info.storage_host.lower()
                or q in info.description.lower()
        )
