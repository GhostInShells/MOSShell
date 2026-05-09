"""
LocalImageStorage — JSONL + 文件系统的本地图片资源存储.

最小实现, 证明 contracts/resource.py 的抽象可行.
JSONL 适合百级数据量, 量大了替换为 SQLite 或其他后端即可.
"""

import asyncio
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Sequence

from PIL import Image
from pydantic import Field

from ghoshell_moss.contracts.resource import (
    ResourceMeta,
    ResourceItem,
    ResourceStorage,
    ResourceStorageFactory,
)
from ghoshell_container import IoCContainer, INSTANCE


# -- Meta & Item -------------------------------------------------------

class LocalImageMeta(ResourceMeta):
    """图片资源元信息."""

    host: str = Field(default="default", description="Storage 实例标识")
    path: str = Field(default="", description="host 内的唯一标识")
    description: str = Field(default="", description="描述信息")
    file_name: str = Field(default="", description="文件系统上的文件名")
    width: int | None = Field(default=None, description="图片宽度 (px)")
    height: int | None = Field(default=None, description="图片高度 (px)")
    format: str = Field(default="", description="图片格式: PNG, JPEG, WEBP, ...")
    file_size: int = Field(default=0, description="文件大小 (bytes)")
    tags: list[str] = Field(default_factory=list, description="标签")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="创建时间 (ISO 8601)",
    )

    @classmethod
    def scheme(cls) -> str:
        return "pil-image"

    @classmethod
    def scheme_description(cls) -> str:
        return "本地图片资源, PIL/Pillow 可读写"


class LocalImageItem(ResourceItem[LocalImageMeta, Image.Image]):
    """图片资源项. meta 立即可用, get() 从磁盘读取."""

    def __init__(self, meta: LocalImageMeta, file_path: str) -> None:
        self._meta = meta
        self._file_path = file_path

    @classmethod
    def meta_type(cls) -> type[LocalImageMeta]:
        return LocalImageMeta

    @property
    def meta(self) -> LocalImageMeta:
        return self._meta

    async def get(self) -> Image.Image:
        return await asyncio.to_thread(Image.open, self._file_path)


# -- Storage -----------------------------------------------------------

class LocalImageStorage(ResourceStorage[LocalImageMeta, Image.Image]):
    """
    JSONL + 文件系统的本地图片存储.

    目录结构:
      {data_dir}/
        pil-image.jsonl       # 索引文件 (每行一个 LocalImageMeta JSON)
        pil-image/            # 图片文件
          beach_sunset.png
          profile_avatar.jpg

    query 支持: keyword 字符串匹配 (description + tags).
    """

    INDEX_FILE = "pil-image.jsonl"
    FILES_DIR = "pil-image"

    def __init__(self, data_dir: str | Path, host: str = "default") -> None:
        self._host = host
        self._data_dir = Path(data_dir)
        self._index_path = self._data_dir / self.INDEX_FILE
        self._files_dir = self._data_dir / self.FILES_DIR

    # -- class-level --------------------------------------------------

    @classmethod
    def scheme(cls) -> str:
        return LocalImageMeta.scheme()

    @classmethod
    def scheme_description(cls) -> str:
        return LocalImageMeta.scheme_description()

    # -- instance-level ------------------------------------------------

    @property
    def host(self) -> str:
        return self._host

    # -- self-describing -----------------------------------------------

    def usage(self) -> str:
        return """\
pil-image: 本地图片资源存储

查询语法: keyword (匹配 description 和 tags 字段)
  pil-image list "sunset"     → 搜索包含 sunset 的图片
  pil-image list              → 列出全部图片 (最多 50)

返回的 ResourceMeta 字段:
  host, path, description, file_name, width, height,
  format, file_size, tags, created_at

限制:
  - query 仅支持简单 keyword 匹配, 不支持正则/布尔/语义搜索
  - 删除和按 path 查找为 O(n) 全表扫描, 适几百条的数据量"""

    async def help(self, question: str | None = None) -> str:
        if question is None:
            return self.usage()
        q = question.lower()
        if "格式" in q or "format" in q:
            return "支持: PNG, JPEG, WEBP, GIF, BMP. 通过 ResourceMeta.format 查看具体格式."
        if "尺寸" in q or "过滤" in q or "filter" in q or "size" in q:
            return "不支持按尺寸/文件大小过滤. list 返回的 ResourceMeta 包含 width/height/file_size, 可自行筛选."
        if "标签" in q or "tag" in q:
            return "query 会同时匹配 description 和 tags 字段, 大小写不敏感."
        return f"[pil-image help] {question}\n此问题无预设答案. 用法概览:\n{self.usage()}"

    # -- CRUD ----------------------------------------------------------

    async def list_metas(
            self, query: str | None = None, limit: int = 50
    ) -> Sequence[LocalImageMeta]:
        metas: list[LocalImageMeta] = []
        for line in self._read_lines():
            meta = LocalImageMeta.model_validate_json(line)
            if query and not self._match(meta, query):
                continue
            metas.append(meta)
            if limit >= 0 and len(metas) >= limit:
                break
        return metas

    async def get(self, path: str) -> LocalImageItem | None:
        meta = self._find_meta(path)
        if meta is None:
            return None
        file_path = self._files_dir / meta.file_name
        if not file_path.exists():
            return None
        return LocalImageItem(meta, str(file_path))

    async def put(
            self, item: ResourceItem[LocalImageMeta, Image.Image]
    ) -> str:
        meta = item.meta
        image = await item.get()

        # path — 用调用者给的, 或生成
        path = meta.path
        if not path:
            path = uuid.uuid4().hex[:12]
            meta.path = path
        meta.host = self._host

        # file name — 用调用者给的, 或基于 path
        file_name = meta.file_name
        if not file_name:
            fmt = (image.format or "PNG").lower()
            file_name = f"{path}.{fmt}"
            meta.file_name = file_name

        # 保存图片
        self._files_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._files_dir / file_name
        await asyncio.to_thread(image.save, str(file_path))

        # 更新 meta 字段
        meta.format = image.format or "PNG"
        meta.width, meta.height = image.size
        meta.file_size = file_path.stat().st_size
        if not meta.created_at:
            meta.created_at = datetime.now(timezone.utc).isoformat()

        # 写入索引
        self._upsert_meta(meta)
        return meta.locator

    async def delete(self, path: str) -> bool:
        lines = self._read_lines()
        found = False
        kept: list[str] = []
        for line in lines:
            meta = LocalImageMeta.model_validate_json(line)
            if meta.path == path:
                found = True
                file_path = self._files_dir / meta.file_name
                if file_path.exists():
                    file_path.unlink()
            else:
                kept.append(line)
        if found:
            self._write_lines(kept)
        return found

    # -- internal ------------------------------------------------------

    def _match(self, meta: LocalImageMeta, query: str) -> bool:
        q = query.lower()
        if q in meta.description.lower():
            return True
        for tag in meta.tags:
            if q in tag.lower():
                return True
        return False

    def _find_meta(self, path: str) -> LocalImageMeta | None:
        for line in self._read_lines():
            meta = LocalImageMeta.model_validate_json(line)
            if meta.path == path:
                return meta
        return None

    def _upsert_meta(self, meta: LocalImageMeta) -> None:
        """追加或原地更新. put 场景: 删除旧条目 + append 新条目."""
        lines = self._read_lines()
        kept = [
            line for line in lines
            if LocalImageMeta.model_validate_json(line).path != meta.path
        ]
        kept.append(meta.model_dump_json())
        self._write_lines(kept)

    def _read_lines(self) -> list[str]:
        if not self._index_path.exists():
            return []
        text = self._index_path.read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.strip()]

    def _write_lines(self, lines: list[str]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines)
        if content:
            content += "\n"
        self._index_path.write_text(content, encoding="utf-8")


# -- Provider -------------------------------------------------------------


class LocalImageResourceFactory(ResourceStorageFactory):
    """
    Register LocalImageStorage into the ResourceRegistry during bootstrap.

    Also implements ResourceStorageMeta so it can be discovered by
    MOSS.manifests.resources scanning.

    Uses workspace.assets()/pil-images as the data directory.
    """

    def __init__(
            self,
            host: str = "workspace-assets",
            assets_sub_path: str = "pil-images",
    ):
        self._host = host
        self._assets_sub_path = assets_sub_path

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.contracts.workspace import Workspace

        workspace = con.force_fetch(Workspace)
        data_dir = workspace.assets().abspath() / self._assets_sub_path
        return LocalImageStorage(data_dir, host=self._host)

    # -- ResourceStorageMeta --------------------------------------------

    @classmethod
    def scheme(cls) -> str:
        return LocalImageStorage.scheme()

    @property
    def host(self) -> str:
        return self._host

    def description(self) -> str:
        return "Local image resource storage backed by JSONL + filesystem"
