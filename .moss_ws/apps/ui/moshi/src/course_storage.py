"""CourseResourceStorage — moshi-course 资源存储后端.

基于 ghoshell_moss.contracts.resource 抽象实现。
scheme = "moshi-course", host = "workspace-courses".
path 编码: "{course_name}/_meta" (课程级) 或 "{course_name}/{chapter_file}" (章节级).

扫描 assets/moshi_courses/ 下的课程目录，每个 .meta.md 定义课程，
0*.md 章节文件各自携带 YAML frontmatter。
"""

import yaml
from pathlib import Path
from typing import Sequence

from pydantic import Field

from ghoshell_moss.contracts.resource import (
    ResourceInfo,
    ResourceItem,
    ResourceStorage,
    ResourceStorageMeta,
)
from ghoshell_container import IoCContainer, INSTANCE

__all__ = [
    "CourseResourceInfo",
    "CourseResourceItem",
    "CourseResourceStorage",
    "CourseResourceStorageMeta",
]


# ── helpers ───────────────────────────────────────────────────────────────

def _split_frontmatter(text: str) -> tuple[dict, str]:
    """分离 YAML frontmatter 和 markdown body。"""
    text = text.strip()
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    return yaml.safe_load(parts[1]) or {}, parts[2].strip()


# ── Meta ──────────────────────────────────────────────────────────────────


class CourseResourceInfo(ResourceInfo):
    """moshi-course 资源元信息。

    一个类型覆盖两种资源：
    - 课程级 (path={course}/_meta): course_name, total_duration, chapter_count, performance_*
    - 章节级 (path={course}/{file}): chapter_id, order, theme, suggested_layout, duration
    """

    host: str = Field(default="workspace-courses")
    path: str = Field(default="")
    description: str = Field(default="")

    # 共用
    course_name: str = Field(default="")
    title: str = Field(default="")

    # 课程级字段
    total_duration: str = Field(default="")
    chapter_count: int = Field(default=0)
    performance_rhythm: str = Field(default="")
    performance_continuity: str = Field(default="")
    performance_fallback: str = Field(default="")
    # 课程 body（AIOS 知识背景），仅课程级有值
    knowledge: str = Field(default="")

    # 章节级字段
    chapter_id: str = Field(default="")
    order: int = Field(default=0)
    theme: str = Field(default="")
    suggested_layout: str = Field(default="")
    duration: str = Field(default="")

    @classmethod
    def scheme(cls) -> str:
        return "moshi-course"

    @classmethod
    def scheme_description(cls) -> str:
        return "Moshi 导演模式课程资源 — 章节化演示剧本，含课程级元信息和章节级剧本数据"


# ── Item ──────────────────────────────────────────────────────────────────


class CourseResourceItem(ResourceItem[CourseResourceInfo, str]):
    """课程资源项。meta 立即可用，get() 返回文件内容（章节 markdown 或课程 body）。"""

    def __init__(self, meta: CourseResourceInfo) -> None:
        self._meta = meta

    @classmethod
    def meta_type(cls) -> type[CourseResourceInfo]:
        return CourseResourceInfo

    @property
    def info(self) -> CourseResourceInfo:
        return self._meta

    async def get(self) -> str:
        """返回章节 markdown 全文或课程 knowledge 文本。"""
        # knowledge 已在 meta 上，章节内容通过 _content 注入
        return getattr(self, "_content", self._meta.knowledge)


# ── Storage ───────────────────────────────────────────────────────────────


class CourseResourceStorage(ResourceStorage[CourseResourceInfo, str]):
    """moshi-course 资源存储。

    启动时 scan() 扫描 assets_dir 下所有课程目录。
    结果全量缓存在内存中（课程数据量小，百 KB 级）。
    """

    def __init__(self, assets_dir: str | Path, host: str = "workspace-courses") -> None:
        self._host = host
        self._assets_dir = Path(assets_dir)
        self._metas: list[CourseResourceInfo] = []
        self._by_path: dict[str, CourseResourceInfo] = {}
        self._file_paths: dict[str, Path] = {}  # path → 文件路径，供 item.get() 用
        self._scanned = False

    # ── class-level ──────────────────────────────────────────────────

    def scheme(self) -> str:
        return CourseResourceInfo.scheme()

    def scheme_description(self) -> str:
        return CourseResourceInfo.scheme_description()

    @property
    def host(self) -> str:
        return self._host

    # ── self-describing ──────────────────────────────────────────────

    def usage(self) -> str:
        lines = [
            f"moshi-course://{self._host} — Moshi 课程资源",
            "",
            "路径格式:",
            "  {course}/_meta          课程级元信息（描述、表演纪律、知识背景）",
            "  {course}/0X-chapter.md  章节剧本（含 frontmatter 元信息 + markdown 剧本）",
            "",
            f"已扫描课程 ({len(self._course_names())} 门):",
        ]
        for name in self._course_names():
            chapters = [m for m in self._metas if m.course_name == name and m.chapter_id]
            lines.append(f"  {name}: {len(chapters)} 章")
        lines.append("")
        lines.append("查询: keyword 匹配 description / title / theme")
        return "\n".join(lines)

    async def help(self, question: str | None = None) -> str:
        if question is None:
            return self.usage()
        q = question.lower()
        if "locator" in q or "地址" in q:
            return (
                f"完整句柄格式: moshi-course://{self._host}/<course>/<path>\n"
                f"示例: moshi-course://{self._host}/show_moshi/_meta"
            )
        if "查询" in q or "搜索" in q or "query" in q:
            return "query 支持 keyword 匹配 description / title / theme 字段，大小写不敏感。"
        return f"[moshi-course help] {question}\n概览:\n{self.usage()}"

    # ── CRUD ─────────────────────────────────────────────────────────

    async def list_infos(
        self, query: str | None = None, limit: int = -1
    ) -> Sequence[CourseResourceInfo]:
        self._ensure_scanned()
        if query is None:
            if limit < 0 or limit >= len(self._metas):
                return list(self._metas)
            return self._metas[:limit]

        q = query.lower()
        result: list[CourseResourceInfo] = []
        for m in self._metas:
            if (
                q in m.description.lower()
                or q in m.title.lower()
                or q in m.theme.lower()
            ):
                result.append(m)
                if limit >= 0 and len(result) >= limit:
                    break
        return result

    async def get(self, path: str) -> CourseResourceItem | None:
        self._ensure_scanned()
        meta = self._by_path.get(path)
        if meta is None:
            return None
        item = CourseResourceItem(meta)
        # 注入文件路径用于 get() 时读取内容
        file_path = self._file_paths.get(path)
        if file_path and file_path.exists():
            item._content = file_path.read_text(encoding="utf-8")
        return item

    async def put(self, item: ResourceItem[CourseResourceInfo, str]) -> str:
        raise NotImplementedError("CourseResourceStorage is read-only")

    async def delete(self, path: str) -> bool:
        raise NotImplementedError("CourseResourceStorage is read-only")

    # ── scan ─────────────────────────────────────────────────────────

    def scan(self) -> None:
        """扫描 assets_dir 下所有课程目录，构建 meta 索引。"""
        self._metas.clear()
        self._by_path.clear()
        self._file_paths.clear()

        if not self._assets_dir.exists():
            self._scanned = True
            return

        for course_dir in sorted(self._assets_dir.iterdir()):
            if not course_dir.is_dir():
                continue
            meta_file = course_dir / ".meta.md"
            if not meta_file.exists():
                continue
            self._scan_course(course_dir, meta_file)

        self._scanned = True

    def refresh(self) -> None:
        """重新扫描（scan 的别名）。"""
        self.scan()

    def _ensure_scanned(self) -> None:
        if not self._scanned:
            self.scan()

    def _scan_course(self, course_dir: Path, meta_file: Path) -> None:
        """扫描单个课程目录。"""
        course_name = course_dir.name
        fm, knowledge = _split_frontmatter(meta_file.read_text())

        perf = fm.get("performance", {})
        perf_rhythm = perf.get("rhythm", "")
        perf_continuity = perf.get("continuity", "")
        perf_fallback = perf.get("fallback", "")

        # 扫描章节文件（按文件名排序，0*.md）
        chapters: list[CourseResourceInfo] = []
        for ch_file in sorted(course_dir.glob("0*.md")):
            ch_fm, _ = _split_frontmatter(ch_file.read_text())
            ch_id = ch_fm.get("id", "")
            if not ch_id:
                continue

            path = f"{course_name}/{ch_file.name}"
            description = (
                f"第{ch_fm.get('order', '?')}章「{ch_fm.get('title', '')}」"
                f" — {ch_fm.get('theme', '')}"
            )

            info = CourseResourceInfo(
                host=self._host,
                path=path,
                description=description,
                course_name=course_name,
                title=ch_fm.get("title", ""),
                chapter_id=ch_id,
                order=ch_fm.get("order", 0),
                theme=ch_fm.get("theme", ""),
                suggested_layout=ch_fm.get("suggested_layout", ""),
                duration=ch_fm.get("duration", ""),
            )
            chapters.append(info)
            self._metas.append(info)
            self._by_path[path] = info
            self._file_paths[path] = ch_file

        # 课程级 meta
        description = fm.get("description", course_name)
        meta_path = f"{course_name}/_meta"
        course_info = CourseResourceInfo(
            host=self._host,
            path=meta_path,
            description=description,
            course_name=course_name,
            title=description,
            total_duration=fm.get("total_duration", ""),
            chapter_count=len(chapters),
            performance_rhythm=perf_rhythm,
            performance_continuity=perf_continuity,
            performance_fallback=perf_fallback,
            knowledge=knowledge,
        )
        self._metas.append(course_info)
        self._by_path[meta_path] = course_info

    def _course_names(self) -> list[str]:
        """返回已扫描的课程名列表（去重，保持顺序）。"""
        seen = set()
        names = []
        for m in self._metas:
            if m.course_name and m.course_name not in seen and m.chapter_id == "":
                seen.add(m.course_name)
                names.append(m.course_name)
        return names


# ── Meta (Provider) ───────────────────────────────────────────────────────


class CourseResourceStorageMeta(ResourceStorageMeta):
    """注册 CourseResourceStorage 到 ResourceRegistry。

    同时实现 ResourceStorageMeta，可被 MOSS.manifests.resources 扫描发现。
    """

    def __init__(
        self,
        host: str = "workspace-courses",
        assets_sub_path: str = "moshi_courses",
    ) -> None:
        self._host = host
        self._assets_sub_path = assets_sub_path

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.contracts.workspace import Workspace

        workspace = con.force_fetch(Workspace)
        assets_dir = workspace.assets().abspath() / self._assets_sub_path
        storage = CourseResourceStorage(assets_dir, host=self._host)
        storage.scan()
        return storage

    # ── ResourceStorageMeta ──────────────────────────────────────────

    def scheme(self) -> str:
        return CourseResourceInfo.scheme()

    @property
    def host(self) -> str:
        return self._host

    def description(self) -> str:
        return "Moshi 导演模式课程资源存储 — 章节化演示剧本的扫描与管理"
