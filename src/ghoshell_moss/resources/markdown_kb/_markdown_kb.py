"""
MarkdownKnowledgeBase — glob+frontmatter 参数化的 Markdown 资源知识库.

K3 正规化 (2026-08-12):
  - 扫描从手写递归改为 ground 的 glob_limited (路径发现) + extract_meta (frontmatter 提取)
  - __init__ 参数化: pattern / keys / limit / max_depth — 与 FrontmatterArguments 同构
  - MarkdownKnowledgeBaseMeta 承载配置 (实例化前可配置, ResourceStorageMeta 注册面)
  - recall 语义召回, LLMFuncs 取不到则 NotImplementedError (依赖宽容)

语义:
  - 默认 pattern "**/*.md": README.md = 目录自身 (排最前), 其他 .md = 子资源
  - skills 模式 pattern "*/SKILL.md": 每个 SKILL.md 是同级技能
  - keys=None 时 description 优先 YAML frontmatter, 否则正文第一句

scheme = "markdown-kb" (类级别, 稳定)
host   = name (实例级, 如 "moss-howto")
path   = 相对于 root 的路径, 如 "subdir/doc.md"
locator = "markdown-kb://moss-howto/subdir/doc.md" (计算属性)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from pydantic import BaseModel, Field
import frontmatter

from ghoshell_moss.contracts.resource import (
    ResourceInfo,
    ResourceItem,
    ResourceStorage,
    ResourceStorageMeta,
    Recollection,
    Query,
    unpack_query,
)
from ghoshell_container import IoCContainer, INSTANCE

if TYPE_CHECKING:
    from ghoshell_moss.contracts.llms import LLMFuncs

__all__ = [
    "MarkdownInfo",
    "MarkdownItem",
    "MarkdownKnowledgeBase",
    "MarkdownKnowledgeBaseMeta",
    "extract_meta",
]


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

    def __init__(
            self,
            host: str,
            root: str | Path,
            pattern: str = "**/*.md",
            keys: list[str] | None = None,
            limit: int | None = None,
            max_depth: int | None = None,
            llm_funcs: "LLMFuncs | None" = None,
    ) -> None:
        self._host = host
        self._root = Path(root)
        self._pattern = pattern
        self._keys = keys
        self._limit = limit
        self._max_depth = max_depth
        self._llm_funcs = llm_funcs
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

    async def recall(self, query: Query) -> Recollection:
        """语义召回 — LLMFuncs 对候选 meta 多标签分类, 返回匹配 locators.

        LLMFuncs 取不到 (构造未注入) 或模型解析失败时回落 NotImplementedError,
        不硬依赖 LLM 基础设施.
        """
        if self._llm_funcs is None:
            raise NotImplementedError(
                f"{self.scheme()}://{self.host} recall requires LLMFuncs; "
                "inject llm_funcs= or construct via MarkdownKnowledgeBaseMeta factory"
            )
        text, _session = unpack_query(query)
        try:
            from ghoshell_moss.contracts.llms import LLMConfig
            resolved = LLMConfig().resolve().get_model(tag="small_fast_model")
        except Exception as e:
            raise NotImplementedError(
                f"recall model resolution failed (no LLM config?): {e}"
            ) from e

        candidates = "\n".join(m.as_line() for m in self._metas)
        result = await self._llm_funcs.call(
            instruction=(
                "你是一个技能召回系统. 每个候选技能有一个 locator 和一段 description "
                "描述它解决的问题. 根据用户的任务描述, 选出最匹配的技能 locator. "
                "只返回候选清单中存在的 locator, 不要编造. 如果没有匹配的, 返回空列表."
            ),
            prompt=(
                f"候选技能 (locator: description):\n{candidates}\n\n"
                f"任务: {text}\n\n"
                f"选出与上面任务最相关的技能 locator."
            ),
            result_type=_RecallResult,
            model=resolved,
        )
        return Recollection(
            locators=list(result.result.locators),
            reasoning=result.result.reasoning,
        )

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
        """扫描 root: glob_limited 发现 + frontmatter 提取, 构建 meta 列表并缓存."""
        self._metas.clear()
        self._by_path.clear()

        from ghoshell_moss.ground._hash import glob_limited
        hits = glob_limited(self._root, self._pattern, recursion=self._max_depth)
        files = [h for h in hits if h.is_file()]
        # README.md 排最前 (目录自身语义, 默认 .md 模式); skills 模式无 README, 不受影响
        files.sort(key=lambda p: (p.name != "README.md", str(p)))
        if self._limit is not None:
            files = files[: self._limit]
        for f in files:
            self._add_meta(f)

    def refresh(self) -> None:
        """重新扫描 (scan 的别名)."""
        self.scan()

    @property
    def metas(self) -> list[MarkdownInfo]:
        """glob 命中的全量 meta 列表 (只读视图)."""
        return list(self._metas)

    def _add_meta(self, file_path: Path) -> None:
        path = str(file_path.relative_to(self._root))
        meta_data = extract_meta(file_path, keys=self._keys)
        if self._keys is None:
            title, description = meta_data["title"], meta_data["description"]
        else:
            # keys 指定时: title 优先 "title" 键, 其次 "name" 键 (SKILL.md), 最后 stem
            title = meta_data.get("title") or meta_data.get("name") or file_path.stem
            description = meta_data.get("description", "")

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


class _RecallResult(BaseModel):
    """recall 的多标签分类结构化输出."""

    locators: list[str] = Field(
        default_factory=list,
        description="命中的 locator 列表",
    )
    reasoning: str = Field(
        default="",
        description="命中理由 (为什么匹配这些资源)",
    )


def extract_meta(file_path: Path, keys: list[str] | None = None) -> dict[str, str]:
    """从 Markdown 文件提取元信息 (public 导出).

    keys=None (默认): 派生 title 与 description —
      title: frontmatter title | 第一个 # heading | 文件 stem
      description: frontmatter description | 正文第一句
    keys 指定: 从 frontmatter 按 key 提取, 缺失值为空串.
    """
    text = file_path.read_text(encoding="utf-8")
    post = frontmatter.loads(text)

    if keys is None:
        title = str(post.metadata.get("title", ""))
        if not title:
            h1_match = re.search(r'^#\s+(.+)$', post.content, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()
        if not title:
            title = str(file_path.stem)

        description = post.metadata.get("description", "")
        if not description:
            description = _first_sentence(post.content)
        return {"title": title, "description": description}

    result: dict[str, str] = {}
    for key in keys:
        value = post.metadata.get(key)
        result[key] = str(value) if value is not None else ""
    return result


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


# -- Meta (manifests 注册配置项) ---------------------------------------


class MarkdownKnowledgeBaseMeta(ResourceStorageMeta):
    """MarkdownKnowledgeBase 的 manifests 注册配置项.

    __init__ 即配置面 (与 FrontmatterArguments 同构): host/root/pattern/keys/limit/max_depth.
    factory(container) 实例化 KnowledgeBase, 尝试从 container 取 LLMFuncs 注入
    (recall 依赖), 取不到则 None (recall 回落 NotImplementedError).
    """

    def __init__(
            self,
            host: str,
            root: str | Path,
            pattern: str = "**/*.md",
            keys: list[str] | None = None,
            limit: int | None = None,
            max_depth: int | None = None,
    ) -> None:
        self._host = host
        self._root = str(root)
        self._pattern = pattern
        self._keys = keys
        self._limit = limit
        self._max_depth = max_depth

    def factory(self, container: IoCContainer) -> INSTANCE:
        llm_funcs = None
        try:
            from ghoshell_moss.contracts.llms import LLMFuncs
            llm_funcs = container.force_fetch(LLMFuncs)
        except Exception:
            llm_funcs = None
        return MarkdownKnowledgeBase(
            host=self._host,
            root=self._root,
            pattern=self._pattern,
            keys=self._keys,
            limit=self._limit,
            max_depth=self._max_depth,
            llm_funcs=llm_funcs,
        )

    @classmethod
    def scheme(cls) -> str:
        return MarkdownKnowledgeBase.scheme()

    @property
    def host(self) -> str:
        return self._host

    def description(self) -> str:
        return "Markdown 文档树知识库 (glob+frontmatter 参数化扫描)"
