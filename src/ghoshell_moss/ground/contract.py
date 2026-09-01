"""Ground contract — Ghost 的认知场.

Ground 一句话承诺: "在 context 表面钉住一组注视目标, 每帧重绘."

结构:
1. GroundSet — 容器. open/close 多个 Ground, CTML 接触面
2. Ground   — 一个打开的场. 绑定目录 root, 持有 pin 集, 承担 frame 渲染
3. Pin      — 一枚注视声明. 具体子类携带 verb + typed arguments (K55 envelope)

不承担: 子进程执行 / 周期性 fold / 持久记忆 — 各自由独立 contract 负责.

SPEC: ``ghoshell_moss.ground.SPECIFICATION.md`` (draft 0.2.0).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = [
    "GroundSet",
    "Ground",
    "Pin",
    "FilePin",
    "GlobPin",
    "FrontmatterPin",
    "LsPin",
    "ExecPin",
    "LawPin",
    "FileArguments",
    "GlobArguments",
    "FrontmatterArguments",
    "LsArguments",
    "ExecArguments",
    "LawArguments",
    "GroundConvention",
    "TemplateInfo",
    "GroundError",
    "PathOutsideRootError",
    "ViewHeader",
    "ViewBlock",
    "RenderedView",
    "Snapshot",
]

# -- constants ----------------------------------------------------------------

# 1-byte length prefix + 64-char namespace convention.
PIN_LABEL_MAX_LEN = 63
_PIN_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,%d}$" % PIN_LABEL_MAX_LEN)


# -- frontmatter -------------------------------------------------------------


class GroundConvention(BaseModel):
    """L0 frontmatter — 场的身份声明 + pins 清单 + 场级 ignore 规则.

    K56: frontmatter 是 MOSS 唯一的机器发明域. pins 作为机器声明的注视
    列表驻留在 frontmatter 中, body 保持纯粹的人/模型叙事域.
    未知 key 保留不拒 (extra="allow").

    ignore / ignore_file 是场级规则 — 所有发现型 pin (glob, frontmatter
    pattern, ls) 自动受约束, 无需 pin 级 opt-in. inline list 与文件引用
    合并为最终规则集.
    """

    id: str | None = Field(default=None, alias="$id")
    name: str | None = None
    description: str | None = None
    pins: list[dict] = Field(default_factory=list)
    ignore: list[str] | None = Field(
        default=None,
        description="场级 ignore 规则清单 — .gitignore 语义, 相对场根.",
    )
    ignore_file: str | None = Field(
        default=None,
        description="场根下的 ignore 规则文件路径 (.gitignore / .groundignore).",
    )

    model_config = {"extra": "allow"}


# -- pin: verb arguments models (K55) ----------------------------------------


class FileArguments(BaseModel):
    """file verb 的 arguments."""
    path: str = Field(description="文件路径, 锚点语法允许.")
    range: str | None = Field(
        default=None,
        pattern=r"^\d+(-\d+)?$",
        description="行区间: 'N' (单行) 或 'N-M' (1-indexed).",
    )
    budget: int | None = Field(
        default=None,
        ge=1,
        description="内容字符数上限, 超限 truncate.",
    )

    model_config = {"extra": "allow"}


class GlobArguments(BaseModel):
    """glob verb 的 arguments."""
    path: str = Field(description="glob 路径 (*, **, ? 标准语义).")
    limit: int | None = Field(default=None, ge=1, description="命中路径数上限.")
    max_depth: int | None = Field(default=None, ge=1, description="** 递归深度上限.")

    model_config = {"extra": "allow"}


class FrontmatterArguments(BaseModel):
    """frontmatter verb 的 arguments.

    Single-file mode (path 不含 glob 字符): 提取该文件 frontmatter.
    Pattern mode (path 含 *?[): 匹配多文件, 每个文件的 frontmatter
    作为独立结果块. 这是渐进式披露的核心 — 一个 pin 看全部子场身份.
    """
    path: str = Field(description="文件路径, 或 glob pattern 匹配多文件.")
    keys: list[str] | None = Field(
        default=None,
        description="只提取指定 frontmatter key. None = 全块.",
    )
    budget: int | None = Field(default=None, ge=1, description="内容字符数上限.")
    limit: int | None = Field(default=None, ge=1, description="多文件模式命中数上限.")
    max_depth: int | None = Field(default=None, ge=1, description="递归发现深度上限.")

    model_config = {"extra": "allow"}


class LsArguments(BaseModel):
    """ls verb 的 arguments."""
    path: str = Field(description="目录路径.")
    depth: int = Field(default=2, ge=1, le=8, description="遍历深度. 默认 2.")
    limit: int | None = Field(default=None, ge=1, description="输出条目数上限.")
    max_depth: int | None = Field(
        default=None,
        ge=1,
        description="递归深度上限. 与 depth 取较小者.",
    )

    model_config = {"extra": "allow"}


class ExecArguments(BaseModel):
    """exec verb 的 arguments.

    授权模型 = Makefile 级信任: ref 指向场根子树内可执行文件, 场作者背书.
    协议禁止内联 shell 字符串 (授权泄漏), 禁止跨场引用 (../, 绝对路径).

    mode 决定解释器: shebang(默认, 需 +x) / python(sys.executable) /
    shell(sh). 非 shebang 模式用解释器显式执行, 不要求脚本 +x.
    """
    ref: str = Field(
        description="场根子树内的可执行文件相对路径. 不允许 ../, 不允许绝对路径.",
    )
    mode: str = Field(
        default="shebang",
        description="解释器模式: shebang(默认, 需 +x) | python(sys.executable) | shell(sh). 非 shebang 不要求 +x.",
    )
    timeout: float = Field(
        default=10.0, gt=0, le=60,
        description="秒. 超时渲染 [timeout] 标记, 不静默.",
    )
    budget: int | None = Field(default=None, ge=1, description="stdout 字符数上限.")

    model_config = {"extra": "allow"}


class LawArguments(BaseModel):
    """law verb 的 arguments — 约定文件法链.

    参数是文件名而非路径: 从 cwd 向上逐层收集该文件 (CLAUDE.md /
    AGENT.md 等约定文件), 到场根为止. 收集到的是每个祖先目录里的
    body 内容, 父级向下展示.
    """
    filename: str = Field(
        description="约定文件名 (CLAUDE.md, AGENT.md...). 从 cwd 向上逐层收集.",
    )
    budget: int | None = Field(
        default=None,
        ge=1,
        description="总字符数上限, 超限 truncate.",
    )
    lines: int | None = Field(
        default=None,
        ge=1,
        description="总行数上限, 超限 truncate.",
    )

    model_config = {"extra": "allow"}


# -- pin verb registry -------------------------------------------------------

_VERB_CLASSES: dict[str, type[Pin]] = {}
"""verb → Pin subclass dispatch.  Populated below after class definitions."""


def _register(verb: str):
    def _deco(cls: type[Pin]) -> type[Pin]:
        _VERB_CLASSES[verb] = cls
        return cls
    return _deco


# -- pins --------------------------------------------------------------------


class Pin(BaseModel):
    """pin 基类 — 场里的一枚注视声明.

    K55 envelope: {label, verb, arguments, description}.  具体子类携带
    typed arguments — verb 是 Literal discriminator, arguments 是多态载体.
    """

    label: str = Field(
        min_length=1,
        max_length=PIN_LABEL_MAX_LEN,
        pattern=_PIN_LABEL_RE.pattern,
        description="ground 内唯一标识, 承担 unpin 定位.",
    )
    description: str = Field(
        default="",
        max_length=280,
        description="短评注 — 一行 '为什么盯这个'. 长解说走 body.",
    )
    always_show: bool = Field(
        default=False,
        description="walk / --template 模式下也不折叠 — 永远展开内容. 默认折叠.",
    )

    model_config = {"extra": "ignore"}

    @property
    def is_cwd_anchored(self) -> bool:
        """是否随 $CWD 移动 — walk 时展开, 场根时也渲染. 默认 False."""
        return False


@_register("file")
class FilePin(Pin):
    """单文件注视, 可选行区间."""

    verb: Literal["file"] = "file"
    arguments: FileArguments


@_register("glob")
class GlobPin(Pin):
    """glob 注视 — 命中路径清单, 不出内容."""

    verb: Literal["glob"] = "glob"
    arguments: GlobArguments


@_register("frontmatter")
class FrontmatterPin(Pin):
    """单文件 frontmatter 注视 — 只出 YAML frontmatter, 不出 body."""

    verb: Literal["frontmatter"] = "frontmatter"
    arguments: FrontmatterArguments


@_register("ls")
class LsPin(Pin):
    """目录列表注视 — 结构视图, 不出内容."""

    verb: Literal["ls"] = "ls"
    arguments: LsArguments


@_register("exec")
class ExecPin(Pin):
    """可执行文件注视 — observe 即执行, 对账 = hash(stdout).

    授权模型: ref 指向场作者背书的场内可执行文件, 类比 Makefile target.
    不允许内联 shell 字符串, 不允许跨场引用.
    """

    verb: Literal["exec"] = "exec"
    arguments: ExecArguments


@_register("law")
class LawPin(Pin):
    """约定文件法链注视 — 兼容外部项目约定 (CLAUDE.md / AGENT.md).

    拉的是文档, 参数是文件名而非路径: 从 cwd 向上逐层收集该文件,
    到场根为止 (边界 = ground root). 父级向下展示, 最多一层 @ 解析,
    有 budget/lines 截断语义. 位置依赖 cwd — walk 时随站立位置变化.
    """

    verb: Literal["law"] = "law"
    arguments: LawArguments

    @property
    def is_cwd_anchored(self) -> bool:
        return True


# -- errors ------------------------------------------------------------------


class GroundError(Exception):
    """Ground 契约层所有异常的基类."""


class PathOutsideRootError(GroundError):
    """路径逃逸出锚点子树. SPEC §8 per-anchor confinement."""


# -- template info -----------------------------------------------------------


class TemplateInfo(BaseModel):
    """.grounds/ 中的一枚模板."""

    name: str = Field(description="模板名 — .grounds/ 下相对路径去掉 .md 后缀.")
    source: str = Field(description="发现源: project / user / ghost.")
    path: Path = Field(description="模板文件绝对路径.")
    description: str = Field(default="", description="模板 frontmatter 的 description, 或 body 首行.")


# -- Ground ------------------------------------------------------------------


class Ground(ABC):
    """一个打开的场.

    绑定目录 root, 持有 pin 集合, 承担 frame 渲染与 load/sediment 生命周期.
    由 GroundSet.open() 创建, 生命周期由 GroundSet 管.
    """

    # -- 元信息 ---------------------------------------------------------------

    @property
    @abstractmethod
    def label(self) -> str:
        """GroundSet 内唯一标识, 缺省 = dir basename + 冲突后缀."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """场根目录 (pin 锚点). 绝对路径."""

    @property
    @abstractmethod
    def doc_path(self) -> Path:
        """GROU.md 路径 (法锚点). 通常 = root/GROUND.md; doc= 参数可指到别处."""

    @property
    @abstractmethod
    def convention(self) -> GroundConvention:
        """加载时的 frontmatter 快照. open 后不可变."""

    # -- pin 管理 -------------------------------------------------------------

    @abstractmethod
    def pins(self) -> list[Pin]:
        """当前 pin 集. 最新 pin/update 在前."""

    @abstractmethod
    def pin(self, pin: Pin) -> Pin:
        """添加或覆盖一枚 pin. 同 label 覆盖 (SPEC §4.3 幂等覆写).

        Raises:
            PathOutsideRootError: pin 目标逃逸 anchor 子树.
        """

    @abstractmethod
    def unpin(self, label: str) -> None:
        """撤掉一枚 pin. label 不存在抛 KeyError."""

    # -- 渲染 -----------------------------------------------------------------

    @abstractmethod
    async def render(self, *, cwd: Path | None = None) -> RenderedView:
        """渲染场 — 返回结构化 RenderedView.

        ``str(view)`` = self-explanatory markdown; ``view.model_dump_json()`` = JSON.
        cwd=None 时用场根 (field-root 模式), 否则 walk 模式.
        """

    @abstractmethod
    async def snapshot(
        self,
        *,
        ack_hash: str | None = None,
        cwd: Path | None = None,
    ) -> Snapshot:
        """渲染 + 感知对账 — 返回渲染对象与其全量 digest.

        render() 保持纯内容; snapshot 是 render + digest + 变更标记.
        对账目标是渲染文本全量 (view.to_markdown() 的 sha256), 不是源文件.

        ack_hash: 调用方声明的已承认基线 (如 channel 持久化的旧值).
        缺省用内部缓存的上一帧 hash. 调用后内部缓存推进到新 hash;
        changed 相对基线计算 — 首次无基线为 False.

        进程内运行时侧影, 不落盘 (seen_* 语义). 须单 owner: 缓存写入
        不并发安全, channel 会话是唯一消费者.
        """

    async def context(self) -> str:
        """渲染当前帧 — 消费给 virtual channel 的 context_messages.

        向后兼容委托到 ``render()``. 新调用方建议直接用 ``render()``.
        """
        return str(await self.render())

    # -- 生命周期 -------------------------------------------------------------

    @property
    @abstractmethod
    def dirty(self) -> bool:
        """是否有未落盘的变更 (pin/unpin/update/模板注入).

        只读消费 (frame/meta/observe) 不置 dirty — close 时跳过 sediment,
        保证只读操作永不改写 GROUND.md.
        """

    @abstractmethod
    async def load(self) -> None:
        """从 GROUND.md 恢复 pin 集 + body. 无 L0 文件 = 空集. K14 startup 消费."""

    @abstractmethod
    async def sediment(self) -> None:
        """把当前 pin 集写回 GROUND.md 的 pin 段. 不动 frontmatter 和 body.

        显式调用无条件写盘; dirty 检查是 close 的职责."""

    # -- 法链 -----------------------------------------------------------------

    @property
    @abstractmethod
    def ignore_spec(self) -> object | None:
        """场级 ignore 规则 (pathspec.PathSpec | None).

        所有发现型 pin 自动受约束. 由 GROUND.md frontmatter 的
        ``ignore`` + ``ignore_file`` 合并生成.
        """

    @abstractmethod
    async def chain_text(self) -> str:
        """返回本场的 body (法) — 单层, 不向上合并祖先."""


# -- GroundSet --------------------------------------------------------------


class GroundSet(ABC):
    """一组 Ground 的容器.

    每个 GroundSet 有自己的 label 空间 — 两个 GroundSet 可各自 open 同目录
    而互不干扰. 多实例, 非单例 — 不同 channel 各自创建自己的 GroundSet.

    CTML 接触面: 父 channel 上的 open/close/pin/unpin/update/frame 收 ground
    参数, 转发到对应 Ground.
    """

    # -- open/close -----------------------------------------------------------

    @abstractmethod
    async def open(
        self,
        dir: str | Path,
        *,
        label: str | None = None,
        doc: str | Path | None = None,
        template: str | None = None,
        override: bool = False,
    ) -> Ground:
        """打开一个场.

        - dir: 场根目录 (pin 锚点). 相对路径按 workspace_root 解析.
        - label: 本 GroundSet 内唯一标识. None = dir basename, 冲突加 -2/-3.
        - doc: 显式 GROU.md 路径 (法锚点). None = dir/GROUND.md.
          doc ≠ dir/GROUND.md 时, law anchor 与 pin anchor 解耦 (K35 携带/属地).
        - template: .grounds/ 中的模板名. 指定时用模板的 body + pins 初始化
          Ground. 模板内容复制, 非引用.
        - override: template 定义全权接管 (body + pins), 忽略现有 GROUND.md
          内容. 预览场景 (`frame --template`) 用 — 模板是镜头, 不是补丁.

        同目录幂等 (按 dir.resolve()): 返回已 active 的 Ground, 忽略传入参数.
        """

    @abstractmethod
    async def close(self, label: str) -> None:
        """关掉场 — 触发 sediment 落盘, 从 active 移除. label 不存在抛 KeyError."""

    # -- 查询 -----------------------------------------------------------------

    @abstractmethod
    def active(self) -> dict[str, Ground]:
        """当前打开的全部场."""

    @abstractmethod
    def get(self, label: str) -> Ground | None:
        """按 label 取场. 不存在返回 None (查询语义, 不抛)."""

    @abstractmethod
    def templates(self) -> list[TemplateInfo]:
        """返回可用模板清单, 按 name 排序."""

    # -- 转发 (CTML 接触面) ---------------------------------------------------

    def pin(self, ground: str, pin: Pin) -> Pin:
        return self._must_get(ground).pin(pin)

    def unpin(self, ground: str, label: str) -> None:
        self._must_get(ground).unpin(label)

    async def frame(self, ground: str) -> str:
        return await self._must_get(ground).context()

    def _must_get(self, label: str) -> Ground:
        g = self.get(label)
        if g is None:
            raise KeyError(label)
        return g

    # -- 生命周期 -------------------------------------------------------------

    @abstractmethod
    async def __aenter__(self) -> Self:
        """进入 GroundSet. 空操作或初始化资源."""

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出 GroundSet. 对全部 active 逐个 sediment, best-effort."""


# -- RenderedView (新渲染数据模型) -------------------------------------------


class ViewHeader(BaseModel):
    """渲染视图的头部 — GROUND.md 身份 + 站立位置."""

    id: str | None = Field(default=None, alias="$id", description="$id 身份声明, 存在才渲染.")
    name: str | None = Field(default=None, description="场名, 来自 GROUND.md name 或目录 basename.")
    description: str | None = Field(default=None, description="场描述, 来自 GROUND.md description.")
    ground_path: str = Field(description="$GROUND — 场根绝对路径.")
    cwd: str | None = Field(default=None, description="$CWD — walk 时的站立位置, field-root 时不出现.")


class ViewBlock(BaseModel):
    """渲染视图里的一个内容块 — body / pin / @-reference."""

    kind: Literal["body", "pin", "at", "folded"] = Field(
        description="块类型: body=场正文, pin=注视结果, at=@-引用展开, folded=walk 时折叠的 pin TOC."
    )
    label: str = Field(description="块标识: body / pin label / @文件名.")
    verb: str | None = Field(
        default=None,
        description="pin 动词 (file|glob|frontmatter|ls|exec|law). kind=pin 时必选, 其余为 None.",
    )
    description: str | None = Field(
        default=None,
        description="pin 的一句话说明, 来自 GROUND.md pin description. 渲染为 markdown 注释的一部分.",
    )
    content: str = Field(default="", description="块内容. body 是 GROUND.md body 原文, pin 是观察结果.")
    meta: dict | None = Field(default=None, description="附加上下文 (files count, budget 等).")


class RenderedView(BaseModel):
    """Ground.render() 的返回值 — header + blocks.

    既可序列化 (``-j`` / ``--json``) 供程序消费, 也可 ``str()`` →
    ``to_markdown()`` 供模型 / 人类直接阅读.
    """

    header: ViewHeader
    blocks: list[ViewBlock] = Field(default_factory=list, description="渲染内容块, 按出现顺序.")

    def to_markdown(self) -> str:
        """序列化为自解释 markdown — HTML 注释承载语义标记, 纯文本可读.

        头部是 YAML frontmatter, 正文直接承接, pin 结果用
        ``<!-- verb-label: description -->`` 标记分隔.
        """
        lines: list[str] = []

        # --- header (YAML frontmatter) ---
        lines.append("---")
        if self.header.id:
            lines.append(f"$id: {self.header.id}")
        if self.header.name:
            lines.append(f"name: {self.header.name}")
        if self.header.description:
            lines.append(f"description: {self.header.description}")
        lines.append(f"$GROUND: {self.header.ground_path}")
        if self.header.cwd:
            lines.append(f"$CWD: {self.header.cwd}")
        lines.append("---")

        # --- blocks ---
        for block in self.blocks:
            lines.append("")
            if block.kind == "body":
                lines.append(block.content.rstrip())
            elif block.kind == "at":
                lines.append("---")
                lines.append(f"<!-- at: {block.label} -->")
                lines.append(block.content.rstrip())
            elif block.kind == "folded":
                desc = f" — {block.description}" if block.description else ""
                lines.append("---")
                lines.append(f"<!-- pins{desc} -->")
                lines.append(block.content.rstrip())
            elif block.kind == "pin":
                desc = f": {block.description}" if block.description else ""
                lines.append("---")
                lines.append(f"<!-- {block.verb}-{block.label}{desc} -->")
                lines.append(block.content.rstrip())

        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()


class Snapshot(BaseModel):
    """一次渲染的感知快照 — 渲染对象 + 全量 digest + 相对基线是否变化.

    Ground.snapshot() 的返回值. hash 覆盖 ``view.to_markdown()`` 全量
    文本, 使 "channel 递给模型的那份文本" 与对账信号闭合.
    """

    view: RenderedView = Field(description="渲染对象, 与 render() 逐字等价.")
    hash: str = Field(description="渲染文本全量的 sha256 digest.")
    changed: bool = Field(
        description="相对基线 (ack_hash 或内部缓存) 是否变化. 首次无基线为 False."
    )
