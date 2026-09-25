"""
LocalVideoStorage — JSONL + 文件系统的本地视频资源存储.

scheme=local-video, get() 返回文件路径, 对接 starlette FileResponse 的零拷贝流式传输.
目录扫描仅用于引导已有的视频文件; 后续增删通过 put()/delete() 走 JSONL 索引.
"""

import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Sequence

from pydantic import Field

from ghoshell_moss.contracts.resource import (
    ResourceInfo,
    ResourceItem,
    ResourceStorage,
    ResourceStorageMeta,
)
from ghoshell_container import IoCContainer, INSTANCE

_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".ogv"}


class LocalVideoInfo(ResourceInfo):
    """视频资源元信息."""

    host: str = Field(default="default", description="Storage 实例标识")
    path: str = Field(default="", description="文件相对于 root_dir 的路径")
    description: str = Field(default="", description="描述信息")
    file_name: str = Field(default="", description="文件名")
    file_size: int = Field(default=0, description="文件大小 (bytes)")
    content_type: str = Field(default="video/mp4", description="MIME type")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="创建时间 (ISO 8601)",
    )

    @classmethod
    def scheme(cls) -> str:
        return "local-video"

    @classmethod
    def scheme_description(cls) -> str:
        return "本地视频资源 — 适用于 mp4/webm/mov 等视频文件"


class LocalVideoItem(ResourceItem[LocalVideoInfo, Path]):
    """视频资源项. get() 返回文件路径, 供 starlette FileResponse 做 Range/流式传输."""

    def __init__(self, meta: LocalVideoInfo, file_path: Path) -> None:
        self._meta = meta
        self._file_path = file_path

    @classmethod
    def meta_type(cls) -> type[LocalVideoInfo]:
        return LocalVideoInfo

    @property
    def info(self) -> LocalVideoInfo:
        return self._meta

    async def get(self) -> Path:
        return self._file_path


class LocalVideoStorage(ResourceStorage[LocalVideoInfo, Path]):
    """JSONL + 文件系统的本地视频存储.

    目录结构:
      {root_dir}/
        local-video.jsonl    # 索引文件 (每行一个 LocalVideoInfo JSON)
        local-video/         # 视频文件
          some_video.mp4
    """

    INDEX_FILE = "local-video.jsonl"
    FILES_DIR = "local-video"

    def __init__(self, root_dir: str | Path, host: str = "default") -> None:
        self._root_dir = Path(root_dir).resolve()
        self._host = host
        self._index_path = self._root_dir / self.INDEX_FILE
        self._files_dir = self._root_dir / self.FILES_DIR
        self._files_dir.mkdir(parents=True, exist_ok=True)

    def scheme(self) -> str:
        return LocalVideoInfo.scheme()

    def scheme_description(self) -> str:
        return LocalVideoInfo.scheme_description()

    @property
    def host(self) -> str:
        return self._host

    def usage(self) -> str:
        return (
            "本地视频资源存储.\n"
            "查询语法: 文件名关键词匹配 (大小写不敏感).\n"
            "locator 格式: local-video://{host}/{path}\n"
            "支持格式: mp4, webm, mov, avi, mkv, m4v, ogv"
        )

    async def help(self, question: str | None = None) -> str:
        if question is None:
            return self.usage()
        return f"LocalVideoStorage host={self._host}, root={self._root_dir}"

    # -- CRUD ----------------------------------------------------------

    async def list_infos(
        self, query: str | None = None, limit: int = -1,
    ) -> Sequence[LocalVideoInfo]:
        metas: list[LocalVideoInfo] = []
        for line in self._read_lines():
            meta = LocalVideoInfo.model_validate_json(line)
            if query and query.lower() not in meta.file_name.lower():
                continue
            metas.append(meta)
            if limit >= 0 and len(metas) >= limit:
                break
        return metas

    async def get(self, path: str) -> LocalVideoItem | None:
        meta = self._find_meta(path)
        if meta is None:
            return None
        file_path = self._files_dir / meta.file_name
        if not file_path.exists() or not file_path.is_file():
            return None
        return LocalVideoItem(meta, file_path)

    async def put(self, item: ResourceItem) -> str:
        meta = item.info
        source = await item.get()
        source_path = Path(source)

        path = meta.path
        if not path:
            path = source_path.stem
            meta.path = path
        meta.host = self._host

        file_name = meta.file_name or source_path.name
        meta.file_name = file_name

        # 复制文件到存储目录
        dest = self._files_dir / file_name
        if source_path.resolve() != dest.resolve():
            shutil.copy2(source_path, dest)

        # 补全元信息
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(dest))
        meta.content_type = meta.content_type or mime_type or "video/mp4"
        meta.file_size = dest.stat().st_size
        if not meta.created_at:
            meta.created_at = datetime.now(timezone.utc).isoformat()

        self._upsert_meta(meta)
        return meta.locator

    async def delete(self, path: str) -> bool:
        lines = self._read_lines()
        found = False
        kept: list[str] = []
        for line in lines:
            meta = LocalVideoInfo.model_validate_json(line)
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

    def _find_meta(self, path: str) -> LocalVideoInfo | None:
        for line in self._read_lines():
            meta = LocalVideoInfo.model_validate_json(line)
            if meta.path == path:
                return meta
        return None

    def _upsert_meta(self, meta: LocalVideoInfo) -> None:
        lines = self._read_lines()
        kept = [
            line for line in lines
            if LocalVideoInfo.model_validate_json(line).path != meta.path
        ]
        kept.append(meta.model_dump_json())
        self._write_lines(kept)

    def _read_lines(self) -> list[str]:
        if not self._index_path.exists():
            return []
        text = self._index_path.read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.strip()]

    def _write_lines(self, lines: list[str]) -> None:
        content = "\n".join(lines)
        if content:
            content += "\n"
        self._index_path.write_text(content, encoding="utf-8")


class LocalVideoResourceMeta(ResourceStorageMeta):

    def __init__(self, host: str = "workspace-assets", assets_sub_path: str = "videos"):
        self._host = host
        self._assets_sub_path = assets_sub_path

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.contracts.workspace import Workspace
        workspace = con.force_fetch(Workspace)
        data_dir = workspace.assets().abspath() / self._assets_sub_path
        data_dir.mkdir(parents=True, exist_ok=True)
        return LocalVideoStorage(data_dir, host=self._host)

    @classmethod
    def scheme(cls) -> str:
        return LocalVideoInfo.scheme()

    @property
    def host(self) -> str:
        return self._host

    def description(self) -> str:
        return "Local video resource storage — JSONL-indexed, Path-returning"


# -- local-webm ----------------------------------------------------------------


class LocalWebmInfo(LocalVideoInfo):
    """WebM 视频资源元信息."""

    content_type: str = Field(default="video/webm")

    @classmethod
    def scheme(cls) -> str:
        return "local-webm"

    @classmethod
    def scheme_description(cls) -> str:
        return "WebM 视频资源 — VP8/VP9 编码，开源 Chromium (Qt WebEngine) 兼容"


class LocalWebmItem(LocalVideoItem):
    """WebM 视频资源项."""

    def __init__(self, meta: LocalWebmInfo, file_path: Path) -> None:
        super().__init__(meta, file_path)

    @classmethod
    def meta_type(cls) -> type[LocalWebmInfo]:
        return LocalWebmInfo


class LocalWebmStorage(LocalVideoStorage):
    """WebM 视频存储.

    继承 LocalVideoStorage，put() 时用 ffmpeg 将源视频转为 WebM (VP9 + Opus)。
    """

    INDEX_FILE = "local-webm.jsonl"
    FILES_DIR = "local-webm"

    def scheme(self) -> str:
        return LocalWebmInfo.scheme()

    def scheme_description(self) -> str:
        return LocalWebmInfo.scheme_description()

    # -- CRUD overrides --------------------------------------------------

    async def list_infos(self, query: str | None = None, limit: int = -1):
        metas: list[LocalWebmInfo] = []
        for line in self._read_lines():
            meta = LocalWebmInfo.model_validate_json(line)
            if query and query.lower() not in meta.file_name.lower():
                continue
            metas.append(meta)
            if limit >= 0 and len(metas) >= limit:
                break
        return metas

    async def get(self, path: str) -> LocalWebmItem | None:
        meta = self._find_meta(path)
        if meta is None:
            return None
        file_path = self._files_dir / meta.file_name
        if not file_path.exists() or not file_path.is_file():
            return None
        return LocalWebmItem(meta, file_path)

    async def put(self, item: ResourceItem) -> str:
        source = await item.get()
        source_path = Path(source)

        path = Path(item.info.path or source_path.stem).stem + ".webm"

        stem = source_path.stem
        webm_filename = f"{stem}.webm"
        dest = self._files_dir / webm_filename

        await self._encode_to_webm(source_path, dest)

        meta = LocalWebmInfo(
            host=self._host,
            path=path,
            description=item.info.description or "",
            file_name=webm_filename,
            file_size=dest.stat().st_size,
            content_type="video/webm",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._upsert_meta(meta)
        return meta.locator

    def _find_meta(self, path: str) -> LocalWebmInfo | None:
        for line in self._read_lines():
            meta = LocalWebmInfo.model_validate_json(line)
            if meta.path == path:
                return meta
        return None

    def _to_info(self, file_path: Path) -> LocalWebmInfo:
        stat = file_path.stat()
        return LocalWebmInfo(
            host=self._host,
            path=file_path.stem,
            description=file_path.stem.replace("_", " ").replace("-", " "),
            file_name=file_path.name,
            file_size=stat.st_size,
            content_type="video/webm",
        )

    async def _encode_to_webm(self, src: Path, dest: Path) -> None:
        import asyncio

        def _run() -> None:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-c:v", "libvpx-vp9",
                "-b:v", "0",
                "-c:a", "libopus",
                str(dest),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

        await asyncio.to_thread(_run)


class LocalWebmResourceMeta(ResourceStorageMeta):

    def __init__(self, host: str = "workspace-assets", assets_sub_path: str = "videos"):
        self._host = host
        self._assets_sub_path = assets_sub_path

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.contracts.workspace import Workspace
        workspace = con.force_fetch(Workspace)
        data_dir = workspace.assets().abspath() / self._assets_sub_path
        data_dir.mkdir(parents=True, exist_ok=True)
        return LocalWebmStorage(data_dir, host=self._host)

    @classmethod
    def scheme(cls) -> str:
        return LocalWebmInfo.scheme()

    @property
    def host(self) -> str:
        return self._host

    def description(self) -> str:
        return "Local WebM video resource storage — ffmpeg VP9 transcoding, Qt WebEngine compatible"
