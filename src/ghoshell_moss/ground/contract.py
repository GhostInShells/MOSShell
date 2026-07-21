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
from typing import Any, Literal

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
    "FileArguments",
    "GlobArguments",
    "FrontmatterArguments",
    "LsArguments",
    "GroundConvention",
    "UpdateResult",
    "GroundError",
    "PathOutsideRootError",
]

# -- constants (K54: every magic number has a name + rationale) ---------------

# 1-byte length prefix + 64-char namespace convention; "看着够用" (K54).
PIN_LABEL_MAX_LEN = 63
_PIN_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,%d}$" % PIN_LABEL_MAX_LEN)

# @-expansion: max nesting depth. 深了模型读不动, 浅了引用链断裂.
AT_MAX_DEPTH = 3
# @-expansion: total char budget. 24k = ~8k tokens, 留 75% 给 pins + 对话.
AT_BUDGET = 24_000


# -- frontmatter -------------------------------------------------------------


class GroundConvention(BaseModel):
    """L0 frontmatter — 场的身份声明.

    K56: frontmatter 只有两个保留 key; 旧 GroundConvention 的 8 字段全部撤除.
    未知 key 保留不拒 (与 SPEC §3 一致).
    """

    id: str | None = Field(default=None, alias="$id")
    label: str | None = None

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

    model_config = {"extra": "allow"}


class GlobArguments(BaseModel):
    """glob verb 的 arguments."""
    pattern: str = Field(description="glob pattern (*, **, ? 标准语义).")

    model_config = {"extra": "allow"}


class FrontmatterArguments(BaseModel):
    """frontmatter verb 的 arguments."""
    path: str = Field(description="markdown 文件路径.")

    model_config = {"extra": "allow"}


class LsArguments(BaseModel):
    """ls verb 的 arguments."""
    path: str = Field(description="目录路径.")
    depth: int = Field(default=2, ge=1, le=8, description="遍历深度. 默认 2.")

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

    model_config = {"extra": "ignore"}


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


# -- errors ------------------------------------------------------------------


class GroundError(Exception):
    """Ground 契约层所有异常的基类."""


class PathOutsideRootError(GroundError):
    """路径逃逸出锚点子树. SPEC §8 per-anchor confinement."""


# -- update result -----------------------------------------------------------


class UpdateResult(BaseModel):
    """update(label) 的返回 — 变更摘要.

    通过 CTML ``<result>`` 机制入对话历史. diff_preview 有界, 避免历史洪泛.
    """

    label: str = Field(description="被 update 的 pin label.")
    changed: bool = Field(description="内容是否变化 (hash 判定).")
    old_hash: str | None = Field(default=None, description="update 前的 seen_hash.")
    new_hash: str | None = Field(default=None, description="update 后的 seen_hash.")
    summary: str = Field(default="", description="变更摘要, 有界: 'lines +N -M', 'glob: +2 -1', etc.")


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

    # -- 对账 -----------------------------------------------------------------

    @abstractmethod
    async def update(self, label: str) -> UpdateResult:
        """承认这枚 pin 的当前世界状态 — 重新观察, 推进 seen_* 基线.

        update 不是"检查变更" (每帧 context() 自动做), 是 "我承认了" 的
        第一人称动词. 承认后下一帧不再标 stale.
        """

    # -- 渲染 -----------------------------------------------------------------

    @abstractmethod
    async def context(self) -> str:
        """渲染当前帧 — 消费给 virtual channel 的 context_messages.

        SPEC §6: body verbatim + pin result blocks delimited by HTML comments.
        async: 并行观察所有 pin + 读文件内容.
        """

    # -- 生命周期 -------------------------------------------------------------

    @abstractmethod
    async def load(self) -> None:
        """从 GROUND.md 恢复 pin 集 + body. 无 L0 文件 = 空集. K14 startup 消费."""

    @abstractmethod
    async def sediment(self) -> None:
        """把当前 pin 集写回 GROUND.md 的 pin 段. 不动 frontmatter 和 body."""

    # -- 法链 -----------------------------------------------------------------

    @abstractmethod
    async def chain_text(self) -> str:
        """返回法链 body — 祖先 GROUND.md body 的 root-first 收集."""


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
    ) -> Ground:
        """打开一个场.

        - dir: 场根目录 (pin 锚点). 相对路径按 workspace_root 解析.
        - label: 本 GroundSet 内唯一标识. None = dir basename, 冲突加 -2/-3.
        - doc: 显式 GROU.md 路径 (法锚点). None = dir/GROUND.md.
          doc ≠ dir/GROUND.md 时, law anchor 与 pin anchor 解耦 (K35 携带/属地).

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

    # -- 转发 (CTML 接触面) ---------------------------------------------------

    def pin(self, ground: str, pin: Pin) -> Pin:
        return self._must_get(ground).pin(pin)

    def unpin(self, ground: str, label: str) -> None:
        self._must_get(ground).unpin(label)

    async def update(self, ground: str, label: str) -> UpdateResult:
        return await self._must_get(ground).update(label)

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
