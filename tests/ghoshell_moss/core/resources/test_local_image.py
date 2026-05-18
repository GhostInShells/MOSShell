"""
LocalImageStorage 单元测试. 基础 CRUD + query + help 覆盖.
"""

import pytest
from PIL import Image as PILImage

from ghoshell_moss.contracts.resource import ClarifyError
from ghoshell_moss.core.resources.local_image import (
    LocalImageInfo,
    LocalImageItem,
    LocalImageStorage,
)


# -- helpers -----------------------------------------------------------

class _InMemoryImageItem(LocalImageItem):
    """用于 put 测试: 包装一个内存中的 PIL Image, 不从磁盘读取."""

    def __init__(self, meta: LocalImageInfo, image: PILImage.Image):
        self._meta = meta
        self._image = image

    async def get(self) -> PILImage.Image:
        return self._image


def _red_image(w=100, h=50) -> PILImage.Image:
    return PILImage.new("RGB", (w, h), color="red")


# -- fixtures ----------------------------------------------------------

@pytest.fixture
def storage(tmp_path):
    return LocalImageStorage(tmp_path)


# -- put ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_without_path_generates_one(storage):
    img = _red_image()
    meta = LocalImageInfo(description="红图", tags=["red"])
    item = _InMemoryImageItem(meta, img)

    locator = await storage.put(item)
    assert locator
    # locator 是完整地址: pil-image://default/<uuid12>
    assert locator.startswith("pil-image://default/")
    assert len(locator) > len("pil-image://default/")


@pytest.mark.asyncio
async def test_put_with_caller_path_respects_it(storage):
    img = _red_image()
    meta = LocalImageInfo(path="my-red", description="红图")
    item = _InMemoryImageItem(meta, img)

    locator = await storage.put(item)
    assert locator == "pil-image://default/my-red"


@pytest.mark.asyncio
async def test_put_updates_meta_dimensions_and_format(storage):
    img = _red_image(200, 100)
    meta = LocalImageInfo(description="尺寸测试")
    item = _InMemoryImageItem(meta, img)

    locator = await storage.put(item)

    stored = await storage.get("default/my-red" if meta.path else meta.path)
    # use get by path, which is derived from meta
    # Actually, use the path returned
    path = locator.removeprefix("pil-image://default/")
    stored = await storage.get(path)
    assert stored is not None
    assert stored.info.width == 200
    assert stored.info.height == 100
    assert stored.info.format in ("PNG", "JPEG")  # PIL default
    assert stored.info.file_size > 0


@pytest.mark.asyncio
async def test_put_overwrite_updates_existing(storage):
    img1 = _red_image(10, 10)
    img2 = _red_image(20, 20)
    meta = LocalImageInfo(path="overwrite", description="first")
    await storage.put(_InMemoryImageItem(meta, img1))

    meta2 = LocalImageInfo(path="overwrite", description="second")
    await storage.put(_InMemoryImageItem(meta2, img2))

    metas = await storage.list_infos(limit=-1)
    assert len(metas) == 1
    assert metas[0].description == "second"
    assert metas[0].width == 20


# -- list_metas --------------------------------------------------------

@pytest.mark.asyncio
async def test_list_metas_empty(storage):
    assert await storage.list_infos(limit=-1) == []


@pytest.mark.asyncio
async def test_list_metas_all(storage):
    for i in range(3):
        meta = LocalImageInfo(path=f"img-{i}", description=f"图{i}")
        await storage.put(_InMemoryImageItem(meta, _red_image()))

    metas = await storage.list_infos(limit=-1)
    assert len(metas) == 3


@pytest.mark.asyncio
async def test_list_metas_with_query_matches_description(storage):
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="a", description="沙滩日落", tags=[]), _red_image()))
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="b", description="山顶日出", tags=[]), _red_image()))

    metas = await storage.list_infos(query="日落")
    assert len(metas) == 1
    assert metas[0].path == "a"


@pytest.mark.asyncio
async def test_list_metas_with_query_matches_tags(storage):
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="a", description="", tags=["beach", "sunset"]), _red_image()))
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="b", description="", tags=["mountain"]), _red_image()))

    metas = await storage.list_infos(query="sunset")
    assert len(metas) == 1
    assert metas[0].path == "a"


@pytest.mark.asyncio
async def test_list_metas_query_case_insensitive(storage):
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="x", description="SUNSET Beach"), _red_image()))

    metas = await storage.list_infos(query="sunset")
    assert len(metas) == 1


@pytest.mark.asyncio
async def test_list_metas_respects_limit(storage):
    for i in range(10):
        await storage.put(_InMemoryImageItem(
            LocalImageInfo(path=f"img-{i}", description=""), _red_image()))

    metas = await storage.list_infos(limit=3)
    assert len(metas) == 3


# -- get ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_existing(storage):
    img = _red_image(64, 64)
    meta = LocalImageInfo(path="avatar", description="头像")
    await storage.put(_InMemoryImageItem(meta, img))

    item = await storage.get("avatar")
    assert item is not None
    assert item.info.path == "avatar"
    assert item.info.locator == "pil-image://default/avatar"
    retrieved = await item.get()
    assert retrieved.size == (64, 64)


@pytest.mark.asyncio
async def test_get_nonexistent(storage):
    item = await storage.get("nonexistent")
    assert item is None


# -- delete ------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_existing(storage):
    await storage.put(_InMemoryImageItem(
        LocalImageInfo(path="del-me", description=""), _red_image()))

    assert await storage.delete("del-me") is True
    assert await storage.get("del-me") is None
    assert await storage.list_infos(limit=-1) == []


@pytest.mark.asyncio
async def test_delete_nonexistent(storage):
    assert await storage.delete("no-such") is False


# -- scheme / meta_type ------------------------------------------------

def test_scheme_consistency():
    assert LocalImageInfo.scheme() == "pil-image"
    assert LocalImageItem.meta_type() is LocalImageInfo
    assert LocalImageItem.scheme() == "pil-image"
    assert LocalImageStorage.scheme() == "pil-image"


# -- help / usage ------------------------------------------------------

def test_usage_returns_string(storage):
    text = storage.usage()
    assert "pil-image" in text
    assert "query" in text.lower() or "keyword" in text


@pytest.mark.asyncio
async def test_help_none_returns_usage(storage):
    text = await storage.help()
    assert "pil-image" in text


@pytest.mark.asyncio
async def test_help_format_question(storage):
    text = await storage.help("支持什么格式")
    assert "PNG" in text


# -- as_content --------------------------------------------------------

def test_as_content_returns_json():
    meta = LocalImageInfo(path="test", description="测试", tags=["a", "b"])
    content = meta.as_content()
    assert '"path": "test"' in content
    assert '"description": "测试"' in content
    assert '"tags": ["a", "b"]' in content
    # locator is injected by as_content()
    assert '"locator": "pil-image://default/test"' in content
