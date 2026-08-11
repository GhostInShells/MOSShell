"""
MarkdownKnowledgeBase — 树形 Markdown 文档知识库.

基于 abcd.py 的抽象, 将目录下的 .md 文件映射为资源:
  - README.md = 目录自身的资源 (类似 __init__.py)
  - 其他 .md = 子资源
  - 递归子目录

Meta 先序遍历, 全量缓存. description 优先 YAML frontmatter, 否则正文第一句.

scheme = "markdown-kb" (类级别, 稳定)
host   = name (实例级, 如 "moss-howto")
path   = 相对于 root 的路径, 如 "subdir/doc.md"
locator = "markdown-kb://moss-howto/subdir/doc.md" (计算属性)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from pydantic import Field
import frontmatter

from ghoshell_moss.contracts.resource import (
    ResourceInfo,
    ResourceItem,
    ResourceStorage,
)

__all__ = ["MarkdownInfo", "MarkdownItem", "MarkdownKnowledgeBase"]


# -- Meta ---------------------------------------------------------------


class MarkdownInfo(ResourceInfo):
    """Markdown 文档资源元信息."""

    host: str = Field(
        default="",
        description="知识库实例名, 如 'moss-howto'",
    )
    path: str = Field(
        default="",
        description="文档相对路径, 如 'subdir/doc.md'",
    )
    description: str = Field(
        default="",
        description="资源描述 (AI 可读)",
    )
    title: str = Field(
        default="",
        description="文档标题 (从 YAML title 或第一个 # heading 提取)",
    )

    __file_path__: Path = None

    @classmethod
    def scheme(cls) -> str:
        return "markdown-kb"

    @classmethod
    def scheme_description(cls) -> str:
        return "Markdown 文档树形知识库, 先序遍历, README.md 为目录自身"

    def as_line(self) -> str:
        """单行摘要, 铺平后给 agent 做上下文."""
        return f"- {self.locator}: {self.description}"


# -- Item ---------------------------------------------------------------


class MarkdownItem(ResourceItem[MarkdownInfo, str]):
    """Markdown 文档资源项. meta 立即可用, get() 读取文件内容."""

    def __init__(self, meta: MarkdownInfo) -> None:
        self._meta = meta

    @classmethod
    def meta_type(cls) -> type[MarkdownInfo]:
        return MarkdownInfo

    @property
    def info(self) -> MarkdownInfo:
        return self._meta

    async def get(self) -> str:
        return self._meta.__file_path__.read_text(encoding="utf-8")


# -- Storage ------------------------------------------------------------


class MarkdownKnowledgeBase(ResourceStorage[MarkdownInfo, str]):
    """
    Markdown 文件树知识库.

    usage:
        kb = MarkdownKnowledgeBase(
            name="moss-howto",
            root=Path("cli/how_to"),
        )
        kb.scan()
        metas = await kb.list_metas()
        item = await kb.get("how-to-make-how-to.md")
        content = await item.get()
    """

    def __init__(self, host: str, root: str | Path) -> None:
        self._host = host
        self._root = Path(root)
        self._metas: list[MarkdownInfo] = []
        self._by_path: dict[str, MarkdownInfo] = {}

    # -- class-level ----------------------------------------------------

    @classmethod
    def scheme(cls) -> str:
        return MarkdownInfo.scheme()

    @classmethod
    def scheme_description(cls) -> str:
        return MarkdownInfo.scheme_description()

    # -- instance-level --------------------------------------------------

    @property
    def host(self) -> str:
        return self._host

    # -- self-describing -------------------------------------------------

    def usage(self) -> str:
        return (
                f"Knowledge base '{self._host}' ({len(self._metas)} documents)\n"
                f"Root: {self._root}\n\n"
                "Documents (pre-order traversal):\n"
                + "\n".join(m.as_line() for m in self._metas)
        )

    async def help(self, question: str | None = None) -> str:
        if question is None:
            return self.usage()
        q = question.lower()
        if "locator" in q or "地址" in q or "寻址" in q:
            return (
                f"完整句柄格式: markdown-kb://{self._host}/<path>\n"
                f"示例: markdown-kb://{self._host}/how-to-make-how-to.md"
            )
        if "描述" in q or "description" in q:
            return (
                "description 优先取 YAML frontmatter 里的 description 字段, "
                "否则取正文第一个非空非标题行的前 120 字符."
            )
        if "结构" in q or "树" in q or "tree" in q:
            return "目录树先序遍历, 每个目录的 README.md 视为目录自身, 其他 .md 文件为子资源."
        return f"[{self._host} help] 此问题无预设答案. 概览:\n{self.usage()}"

    # -- CRUD ------------------------------------------------------------

    async def list_infos(
            self, query: str | None = None, limit: int = -1
    ) -> Sequence[MarkdownInfo]:
        if query is None:
            if limit < 0 or limit >= len(self._metas):
                return list(self._metas)
            return self._metas[:limit]

        # simple keyword match on description + title + path
        result: list[MarkdownInfo] = []
        q = query.lower()
        for m in self._metas:
            if q in m.description.lower() or q in m.title.lower() or q in m.path.lower():
                result.append(m)
                if len(result) >= limit:
                    break
        return result

    async def get(self, path: str) -> MarkdownItem | None:
        meta = self._by_path.get(path)
        if meta is None:
            return None
        return MarkdownItem(meta)

    async def put(
            self, item: ResourceItem[MarkdownInfo, str]
    ) -> str:
        raise NotImplementedError("MarkdownKnowledgeBase is read-only")

    async def delete(self, path: str) -> bool:
        raise NotImplementedError("MarkdownKnowledgeBase is read-only")

    # -- scan ------------------------------------------------------------

    def scan(self) -> None:
        """扫描根目录, 构建先序 meta 列表并缓存."""
        self._metas.clear()
        self._by_path.clear()
        self._scan_dir(self._root)

    def refresh(self) -> None:
        """重新扫描 (scan 的别名)."""
        self.scan()

    @property
    def metas(self) -> list[MarkdownInfo]:
        """先序遍历的全量 meta 列表 (只读视图)."""
        return list(self._metas)

    def _scan_dir(self, current: Path) -> None:
        # 1. README.md 优先 (目录自身)
        readme = current / "README.md"
        if readme.exists():
            self._add_meta(readme)

        # 2. 其他 .md 文件
        for md in sorted(current.glob("*.md"), key=lambda p: p.name):
            if md.name == "README.md":
                continue
            self._add_meta(md)

        # 3. 子目录递归
        dirs = sorted(
            [d for d in current.iterdir()
             if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("_")],
            key=lambda d: d.name,
        )
        for d in dirs:
            self._scan_dir(d)

    def _add_meta(self, file_path: Path) -> None:
        path = str(file_path.relative_to(self._root))
        title, description = _extract_meta(file_path)

        meta = MarkdownInfo(
            host=self._host,
            path=path,
            description=description,
            title=title,
        )
        # 把文件路径挂在私有字段上给 MarkdownItem.get() 用
        meta.__file_path__ = file_path

        self._metas.append(meta)
        self._by_path[path] = meta


# -- helpers ------------------------------------------------------------


def _extract_meta(file_path: Path) -> tuple[str, str]:
    """返回 (title, description)."""
    text = file_path.read_text(encoding="utf-8")
    post = frontmatter.loads(text)

    # title: 优先 frontmatter, 否则第一个 # heading
    title = str(post.metadata.get("title", ""))
    if not title:
        h1_match = re.search(r'^#\s+(.+)$', post.content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

    # description: 优先 frontmatter, 否则正文第一句
    description = post.metadata.get("description", "")
    if not description:
        description = _first_sentence(post.content)

    if not title:
        title = str(file_path.stem)
    return title, description


def _first_sentence(text: str) -> str:
    """提取正文第一句. 跳过空行, 标题行, 分隔线."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return stripped
    return "(no content)"
