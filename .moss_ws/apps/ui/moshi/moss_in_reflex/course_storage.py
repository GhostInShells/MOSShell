"""CourseStorage — JSON 文件系统的本地课程资源存储。

每个课程一个目录，meta.json + 各章 JSON 文件。
注册进 ResourceRegistry 后，Ghost 通过 scheme://host/path 访问，
和 pil-image 用完全相同的 API。
"""

import json
from pathlib import Path
from typing import Sequence

from ghoshell_moss.contracts.resource import (
    ResourceInfo,
    ResourceItem,
    ResourceStorage,
    ResourceStorageMeta,
)
from ghoshell_container import IoCContainer, INSTANCE


# -- Meta & Item -------------------------------------------------------

class CourseInfo(ResourceInfo):
    """课程资源元信息。"""

    host: str = "workspace-assets"
    path: str = ""
    description: str = ""
    title: str = ""
    chapters_count: int = 0

    @classmethod
    def scheme(cls) -> str:
        return "course"

    @classmethod
    def scheme_description(cls) -> str:
        return "课程资源，包含备课章节的结构化数据"


class CourseItem(ResourceItem[CourseInfo, dict]):
    """课程资源项。meta 立即可用，get() 直接返回内存 dict。"""

    def __init__(self, meta: CourseInfo, data: dict) -> None:
        self._meta = meta
        self._data = data

    @classmethod
    def meta_type(cls) -> type[CourseInfo]:
        return CourseInfo

    @property
    def info(self) -> CourseInfo:
        return self._meta

    async def get(self) -> dict:
        return self._data


# -- Storage -----------------------------------------------------------

class CourseStorage(ResourceStorage[CourseInfo, dict]):
    """JSON 文件系统的本地课程存储。

    目录结构:
      {data_dir}/
        水调歌头/
          meta.json          # 课程元信息
          ch00_导语.json     # 章节文件
          ch01_背景.json
          ...

    query 支持: 无参数浏览全部课程，keyword 匹配课程 title。
    """

    META_FILE = "meta.json"

    def __init__(self, data_dir: str | Path, host: str = "workspace-assets") -> None:
        self._host = host
        self._data_dir = Path(data_dir)

    # -- class-level --------------------------------------------------

    @classmethod
    def scheme(cls) -> str:
        return CourseInfo.scheme()

    @classmethod
    def scheme_description(cls) -> str:
        return CourseInfo.scheme_description()

    # -- instance-level ------------------------------------------------

    @property
    def host(self) -> str:
        return self._host

    # -- self-describing -----------------------------------------------

    def usage(self) -> str:
        return """\
course: 本地课程资源存储

查询语法: keyword（匹配课程标题，大小写不敏感）
  course list              → 列出全部课程
  course list "水调歌头"   → 搜索包含关键字的课程

返回的 ResourceMeta 字段:
  host, path, description, title, chapters_count

路径格式: {课程名}/{章节名}  或  {课程名}/meta
  course://workspace-assets/水调歌头/meta         → 课程元信息
  course://workspace-assets/水调歌头/ch01_背景     → 章节内容"""

    async def help(self, question: str | None = None) -> str:
        if question is None:
            return self.usage()
        q = question.lower()
        if "格式" in q or "format" in q or "字段" in q:
            return (
                "章节 JSON 格式: {version, sub_title, main_text, annotations, "
                "appreciation, images (locator 列表)}"
            )
        if "查询" in q or "query" in q:
            return "query 支持 keyword 匹配课程 title，大小写不敏感。不传 query 列出全部课程。"
        return f"[course help] {question}\n此问题无预设答案。用法概览:\n{self.usage()}"

    # -- CRUD (writing) -------------------------------------------------

    async def put(self, item: ResourceItem[CourseInfo, dict]) -> str:
        meta = item.info
        data = await item.get()

        path = meta.path
        if not path:
            raise ValueError("CourseInfo.path is required for put")

        meta.host = self._host

        file_path = self._data_dir / f"{path}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, ensure_ascii=False, default=str)
        file_path.write_text(content, encoding="utf-8")

        return meta.locator

    async def delete(self, path: str) -> bool:
        file_path = self._data_dir / f"{path}.json"
        if not file_path.exists():
            return False
        file_path.unlink()

        # 如果所属目录下无文件，清理空目录
        parent = file_path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
        return True

    # -- CRUD (reading) --------------------------------------------------

    async def get(self, path: str) -> CourseItem | None:
        file_path = self._data_dir / f"{path}.json"
        if not file_path.exists():
            return None

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        meta = CourseInfo(
            host=self._host,
            path=path,
            title=data.get("title", ""),
            chapters_count=data.get("chapters_count", 0),
        )
        return CourseItem(meta, data)

    async def list_infos(
        self, query: str | None = None, limit: int = 50
    ) -> Sequence[CourseInfo]:
        infos: list[CourseInfo] = []

        if not self._data_dir.exists():
            return infos

        for course_dir in sorted(self._data_dir.iterdir()):
            if not course_dir.is_dir():
                continue

            meta_file = course_dir / self.META_FILE
            if not meta_file.exists():
                continue

            try:
                meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            title = meta_data.get("title", course_dir.name)
            if query and query.lower() not in title.lower():
                continue

            infos.append(CourseInfo(
                host=self._host,
                path=f"{course_dir.name}/meta",
                title=title,
                description=meta_data.get("title", ""),
                chapters_count=meta_data.get("chapters_count", 0),
            ))

            if 0 <= limit <= len(infos):
                break

        return infos


# -- ResourceStorageMeta (for manifest discovery) -----------------------

class CourseStorageMeta(ResourceStorageMeta):
    """IoC 工厂：从 Workspace 拿 data_dir，注册 CourseStorage 到 ResourceRegistry。

    Matrix 扫描 mode manifests 时通过 isinstance(obj, ResourceStorageMeta)
    发现此实例，自动调用 factory() 并注册到 ResourceRegistry(scheme="course")。
    """

    def __init__(
        self,
        host: str = "workspace-assets",
        assets_sub_path: str = "courses",
    ):
        self._host = host
        self._assets_sub_path = assets_sub_path

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.contracts.workspace import Workspace

        workspace = con.force_fetch(Workspace)
        data_dir = workspace.assets().abspath() / self._assets_sub_path
        return CourseStorage(data_dir, host=self._host)

    # -- ResourceStorageMeta --------------------------------------------

    @classmethod
    def scheme(cls) -> str:
        return CourseStorage.scheme()

    @property
    def host(self) -> str:
        return self._host

    def description(self) -> str:
        return "Local course resource storage backed by JSON filesystem"
