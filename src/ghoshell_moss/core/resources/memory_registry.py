"""
InMemoryRegistry — 基于内存 dict 的 ResourceRegistry 最小实现.

与 contracts/resource.py 的旧版不同, 这里使用 abcd.py 的新接口:
  - 以 (scheme, host) 二元组为主键
  - get(locator) 解析 scheme://host/path 路由
"""

from typing import Sequence

from ghoshell_moss.contracts.resource import (
    ResourceMeta,
    ResourceItem,
    ResourceStorage,
    ResourceRegistry,
    R,
)
from ghoshell_container import Provider, IoCContainer, INSTANCE

__all__ = ["InMemoryResourcesRegistry", "InMemoryResourceRegistryProvider"]


def _parse_locator(locator: str) -> tuple[str, str, str]:
    """解析 scheme://host/path → (scheme, host, path)."""
    if "://" not in locator:
        raise ValueError(f"Invalid locator (missing ://): {locator}")
    scheme, rest = locator.split("://", 1)
    if "/" not in rest:
        raise ValueError(f"Invalid locator (missing /path): {locator}")
    host, path = rest.split("/", 1)
    return scheme, host, path


class InMemoryResourcesRegistry(ResourceRegistry):
    """
    内存级别的 ResourceRegistry 实现.
    单进程, (scheme, host) → storage dict.

    用法:
        registry = InMemoryRegistry()
        kb = MarkdownKnowledgeBase(name="moss-howto", root=...)
        kb.scan()
        await registry.register(kb)
        item = await registry.get("markdown-kb://moss-howto/how-to-make-how-to.md")
        metas = await registry.list_metas("markdown-kb")
    """

    def __init__(self):
        self._storages: dict[tuple[str, str], ResourceStorage] = {}

    def _key(self, scheme: str, host: str) -> tuple[str, str]:
        return (scheme, host)

    def register(self, storage: ResourceStorage) -> None:
        key = self._key(storage.scheme(), storage.host)
        self._storages[key] = storage

    def unregister(self, scheme: str, host: str) -> bool:
        key = self._key(scheme, host)
        return self._storages.pop(key, None) is not None

    def schemes(self) -> Sequence[str]:
        return list({k[0] for k in self._storages})

    def hosts(self, scheme: str) -> Sequence[str]:
        return [k[1] for k in self._storages if k[0] == scheme]

    async def get(self, locator: str) -> ResourceItem | None:
        scheme, host, path = _parse_locator(locator)
        storage = self._storages.get(self._key(scheme, host))
        if storage is None:
            return None
        return await storage.get(path)

    async def get_by_item_type(
            self, item_cls: type[R], locator: str
    ) -> R | None:
        scheme, host, path = _parse_locator(locator)
        # verify scheme matches
        if scheme != item_cls.scheme():
            return None
        storage = self._storages.get(self._key(scheme, host))
        if storage is None:
            return None
        return await storage.get(path)

    async def list_metas(
            self,
            scheme: str,
            host: str | None = None,
            query: str | None = None,
            limit: int = 50,
    ) -> Sequence[ResourceMeta]:
        if host is not None:
            storage = self._storages.get(self._key(scheme, host))
            if storage is None:
                return []
            return await storage.list_metas(query, limit)

        # host=None: aggregate across all hosts for this scheme
        results: list[ResourceMeta] = []
        for (s, h), storage in self._storages.items():
            if s != scheme:
                continue
            metas = await storage.list_metas(query, limit)
            results.extend(metas)
            if limit >= 0 and len(results) >= limit:
                results = results[:limit]
                break
        return results

    async def help(
            self,
            scheme: str,
            host: str | None = None,
            question: str | None = None,
    ) -> str:
        if host is not None:
            storage = self._storages.get(self._key(scheme, host))
            if storage is None:
                return f"(scheme={scheme}, host={host}) 未注册. 已注册: {list(self._storages.keys())}"
            return await storage.help(question)

        # host=None: list all hosts for this scheme
        hosts = self.hosts(scheme)
        if not hosts:
            return f"scheme '{scheme}' 未注册. 已注册 schemes: {self.schemes()}"
        lines = [f"scheme '{scheme}' 有 {len(hosts)} 个 host 实例:"]
        for h in hosts:
            storage = self._storages.get(self._key(scheme, h))
            if storage:
                lines.append(f"  - {h}: {storage.scheme_description()}")
        return "\n".join(lines)

    async def usage(self, scheme: str, host: str | None = None) -> str:
        if host is not None:
            storage = self._storages.get(self._key(scheme, host))
            if storage is None:
                return f"(scheme={scheme}, host={host}) 未注册."
            return storage.usage()

        hosts = self.hosts(scheme)
        if not hosts:
            return f"scheme '{scheme}' 未注册."
        lines = [f"scheme '{scheme}' usage (first host '{hosts[0]}'):"]
        storage = self._storages.get(self._key(scheme, hosts[0]))
        if storage:
            lines.append(storage.usage())
        return "\n".join(lines)


class InMemoryResourceRegistryProvider(Provider[ResourceRegistry]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        return InMemoryResourcesRegistry()
