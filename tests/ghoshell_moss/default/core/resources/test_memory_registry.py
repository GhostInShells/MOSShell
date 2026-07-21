"""
InMemoryRegistry 单元测试.
"""

import pytest
from PIL import Image as PILImage

from ghoshell_moss.core.resources.local_image import (
    LocalImageInfo,
    LocalImageItem,
    LocalImageStorage,
)
from ghoshell_moss.core.resources.memory_registry import InMemoryResourcesRegistry


class _InMemoryImageItem(LocalImageItem):
    def __init__(self, meta: LocalImageInfo, image: PILImage.Image):
        self._meta = meta
        self._image = image

    async def get(self) -> PILImage.Image:
        return self._image


def _red_image(w=100, h=50) -> PILImage.Image:
    return PILImage.new("RGB", (w, h), color="red")


@pytest.fixture
def registry():
    return InMemoryResourcesRegistry()


async def _populated(registry, tmp_path):
    """helper: 注册一个 LocalImageStorage 并返回 (registry, storage)."""
    storage = LocalImageStorage(tmp_path)
    registry.register(storage)
    return registry, storage


# -- register / unregister / schemes / hosts ---------------------------

@pytest.mark.asyncio
async def test_register_and_schemes(registry, tmp_path):
    assert registry.schemes() == []

    registry.register(LocalImageStorage(tmp_path))
    assert "pil-image" in registry.schemes()


@pytest.mark.asyncio
async def test_unregister(registry, tmp_path):
    registry.register(LocalImageStorage(tmp_path))
    assert registry.unregister("pil-image", "default") is True
    assert registry.schemes() == []
    assert registry.unregister("pil-image", "default") is False


@pytest.mark.asyncio
async def test_hosts(registry, tmp_path):
    registry.register(LocalImageStorage(tmp_path))
    assert registry.hosts("pil-image") == ["default"]


# -- list_metas via registry -------------------------------------------

@pytest.mark.asyncio
async def test_list_metas_via_registry(registry, tmp_path):
    reg, storage = await _populated(registry, tmp_path)
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="x", description="test"), _red_image()))

    metas = await reg.list_infos("pil-image")
    assert len(metas) == 1
    assert metas[0].locator == "pil-image://default/x"


@pytest.mark.asyncio
async def test_list_metas_unknown_scheme(registry):
    metas = await registry.list_infos("no-such-scheme")
    assert metas == []


# -- get / get_by_item_type --------------------------------------------

@pytest.mark.asyncio
async def test_get_via_registry(registry, tmp_path):
    reg, storage = await _populated(registry, tmp_path)
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="beach", description="海滩"), _red_image()))

    item = await reg.get("pil-image://default/beach")
    assert item is not None
    assert item.info.locator == "pil-image://default/beach"


@pytest.mark.asyncio
async def test_get_unknown_scheme(registry):
    item = await registry.get("no-such://host/x")
    assert item is None


@pytest.mark.asyncio
async def test_get_by_item_type(registry, tmp_path):
    reg, storage = await _populated(registry, tmp_path)
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="beach", description="海滩"), _red_image()))

    item = await reg.get_by_item_type(LocalImageItem, "pil-image://default/beach")
    assert item is not None
    assert isinstance(item, LocalImageItem)
    assert item.info.locator == "pil-image://default/beach"


# -- help / usage via registry -----------------------------------------

@pytest.mark.asyncio
async def test_help_via_registry(registry, tmp_path):
    reg, storage = await _populated(registry, tmp_path)
    text = await reg.help("pil-image")
    assert "pil-image" in text


@pytest.mark.asyncio
async def test_help_unknown_scheme(registry):
    text = await registry.help("no-such")
    assert "未注册" in text


@pytest.mark.asyncio
async def test_usage_via_registry(registry, tmp_path):
    reg, storage = await _populated(registry, tmp_path)
    text = await reg.usage("pil-image")
    assert "pil-image" in text


@pytest.mark.asyncio
async def test_usage_unknown_scheme(registry):
    text = await registry.usage("no-such")
    assert "未注册" in text
