"""
Memento 契约层 — 轨迹第一公民的认知基建.

核心假设:
- commit 是 fork 与重绘的不可变锚点. commit_id 是 stable id (非 content-addressable).
- 成员 (commit + moments) 冻结后不可变; 释义 (title/body/threads) 追加新版本, last-wins.
- payload 不透明: memento 原样透传, 不解析不改写. 本模块不 import 任何 payload schema.
- owner 隔离: 只有 owner 可写自己的 staging/commits/释义. 跨 owner 只读.
- 化身只能从 commit 出生, 永不从 staging. overlay 在 owner meta.json, 不进 staging.
- branch = 时间线: name + ref (BranchRef) + staging. 无 ULID id. 只有开线/延线/弃线.
- commit 自治目录, 出生即冻结, 懒创建. Y-m 分桶从 ULID 时间戳纯函数解出, 严格 UTC.
- merge 不存在: 单父链钉死.

磁盘格式见同目录 FORMAT.md, 设计上下文见 workstreams/2026/06/momento-mori/FEATURE.md.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

from pydantic import AwareDatetime, BaseModel, Field
from ulid import ULID

__all__ = [
    # trailer
    "TRAILER_THREAD",
    "TRAILER_RESUMES",
    "TRAILER_SUSPENDS",
    "TRAILER_KIND",
    "TRAILER_MEMENTO_REF",
    "split_trailers",
    "join_trailers",
    "trailer_values",
    # id
    "new_commit_id",
    # 数据模型
    "MomentRecord",
    "Commit",
    "CommitNote",
    "CommitView",
    "BranchRef",
    "CommitRef",
    "BranchWindow",
    "CommitDetail",
    # hook
    "MementoHooks",
    "NullHooks",
    # Protocol / ABC
    "Line",
    "Memento",
    # 异常
    "MementoError",
    "ReadonlyLineError",
    "LineNotFoundError",
    "CommitNotFoundError",
    "MomentFrozenError",
    "MomentNotInCommitError",
    "EmptyStagingError",
]


def _now() -> datetime:
    return datetime.now().astimezone()


def new_commit_id() -> str:
    """生成 commit id: 'cmt_' + ULID. 前缀保证 grep 可反查."""
    return f"cmt_{ULID()}"


# ── trailer 工具 (FORMAT.md §6) ─────────────────────────────────────────────

TRAILER_THREAD = "Thread"
TRAILER_RESUMES = "Resumes"
TRAILER_SUSPENDS = "Suspends"
TRAILER_KIND = "Kind"
TRAILER_MEMENTO_REF = "Memento-Ref"

_TRAILER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*): .+$")


def split_trailers(body: str) -> tuple[str, list[tuple[str, str]]]:
    """拆分 body 为 (正文, [(key, value)...]). trailer 块 = body 末尾连续的 Key: Value 行."""
    if not body:
        return "", []
    lines = body.split("\n")
    split_at = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if _TRAILER_RE.match(lines[i]):
            split_at = i
        else:
            break
    text_lines = lines[:split_at]
    trailer_lines = lines[split_at:]
    if text_lines and text_lines[-1] == "":
        text_lines.pop()
    trailers: list[tuple[str, str]] = []
    for line in trailer_lines:
        m = _TRAILER_RE.match(line)
        if m:
            key = line[: m.end(1)]
            value = line[m.end(1) + 2 :]
            trailers.append((key, value))
    return ("\n".join(text_lines), trailers)


def join_trailers(text: str, trailers: Iterable[tuple[str, str]]) -> str:
    """组装 body = 正文 + 空行 + trailer 块. 只有 trailer 无正文时不加空行."""
    lines = text.split("\n") if text else []
    for k, v in trailers:
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def trailer_values(trailers: Iterable[tuple[str, str]], key: str) -> list[str]:
    """取 trailer 块中指定 key 的所有值, 按出现序."""
    return [v for k, v in trailers if k == key]


# ── 数据模型 ─────────────────────────────────────────────────────────────────


class MomentRecord(BaseModel):
    """
    Moment 信封. payload 对 memento 不透明 — 原样透传, 字节不动.

    可变性: 同 id 在 staging 内覆盖写 (last-wins); 一旦冻结进 commit 目录,
    再向 staging 写同 id 即 MomentFrozenError. threads 是释义, 冻结后仍可
    经 annotate_moment 整体替换.
    """

    id: str = Field(description="生产者 id. 非空, [A-Za-z0-9._\\-]{1,128}, branch 可写范围唯一.")
    created: AwareDatetime = Field(default_factory=_now)
    type: str = Field(description="payload schema 标识 (如 'moss.moment/v1').")
    payload: dict[str, Any] = Field(description="任意 JSON object. memento 原样透传.")
    threads: list[str] = Field(default_factory=list, description="线索标签. 写入时标注, 可经 moment_note 更新.")


class BranchRef(BaseModel):
    """
    指向某个 commit (可选: commit 内某 moment 为止的前缀切片).

    origin: 目标 commit 所在的 owner. 同 owner 时 = 当前 owner; 跨 owner checkout
    时 != 当前 owner. moment_id 缺省 = 整个 commit; 给定时 commit 只贡献
    [首个 moment ... moment_id] 前缀 (含). 空前缀在类型层不可构造.
    """

    origin: str = Field(description="目标 commit 所属的 owner.")
    commit_id: str = Field(description="目标 commit id.")
    moment_id: str | None = Field(default=None, description="切片截止 moment id (含). None = 整个 commit.")


class Commit(BaseModel):
    """冻结的认知锚点. 成员不可变 — 动它所有子 branch 的 parent 链集体失效."""

    id: str = Field(default_factory=new_commit_id, description="全局稳定 id ('cmt_' ULID).")
    created: AwareDatetime = Field(default_factory=_now)


class CommitNote(BaseModel):
    """
    commit 释义. 追加式多版本, last-wins. 历史版本永远可寻址.

    title: 一行摘要, 用于窗口渲染和搜索.
    body: 正文 + trailer 块, 整体替换语义.
    """

    ref: str = Field(description="所释义的 commit id.")
    title: str = Field(default="", description="一行摘要. 窗口渲染和搜索用.")
    body: str = Field(default="", description="正文 + trailer 块.")
    ts: AwareDatetime = Field(default_factory=_now, description="展示/诊断用, 不参与 last-wins 定序.")
    by: str = Field(default="", description="释义写入者.")

    def text(self) -> str:
        """body 的正文部分 (剥离 trailer)."""
        return split_trailers(self.body)[0]

    def trailers(self) -> list[tuple[str, str]]:
        return split_trailers(self.body)[1]

    def threads(self) -> list[str]:
        return trailer_values(self.trailers(), TRAILER_THREAD)


class CommitView(BaseModel):
    """Commit + 当前释义 (last-wins) 的读取视图. note_seq 是渲染打戳 (0-based 版本号)."""

    commit: Commit
    note: CommitNote = Field(description="当前 (最新) 释义.")
    note_seq: int = Field(description="释义版本号, 0-based. 渲染方持有 (commit_id, note_seq) 即可复原当时视图.")

    @property
    def id(self) -> str:
        return self.commit.id

    def summary(self) -> str:
        return self.note.title or self.note.text()


class CommitRef(BaseModel):
    """commits.jsonl 行. owner 级 append-only — 行序 = 物理时序."""

    commit_id: str
    branch: str = Field(description="冻结时所在 line name (诊断用).")
    parent: BranchRef | None = Field(default=None, description="父 commit. root 为 None.")
    ts: AwareDatetime = Field(default_factory=_now, description="冻结时间, 展示用.")
    kind: Literal["semantic", "mechanical"] = Field(default="semantic")


class BranchWindow(BaseModel):
    """快路径窗口渲染: 折叠区摘要 + 明细区最近 N 帧."""

    summaries: list[CommitView] = Field(description="折叠区: 之前 M 个 commit 的当前释义.")
    details: list[MomentRecord] = Field(description="明细区: 最近 N 帧 (含 staging).")


class CommitDetail(BaseModel):
    """show <commit_id> 的完整返回."""

    commit: Commit
    moments: list[MomentRecord] = Field(description="冻结成员全文, 与 commit 成员一一对应.")
    notes: list[CommitNote] = Field(description="全部释义版本, 行序 (版本序).")


# ── Hook 协议 ────────────────────────────────────────────────────────────────


@runtime_checkable
class MementoHooks(Protocol):
    """fire-and-forget 事件回调. 抛错不得影响核心写入路径."""

    def on_record_staged(self, line: str, record: MomentRecord) -> None:
        """record() 追加 moment 到 staging 后触发."""
        ...

    def on_commit(self, line: str, view: CommitView) -> None:
        """commit() 完成后触发. view 含初始释义."""
        ...

    def on_reinterpreted(self, commit_id: str, view: CommitView) -> None:
        """annotate() 追加释义后触发. view 为新版本."""
        ...

    def on_line_created(self, name: str, from_ref: BranchRef | None) -> None:
        """create_line() 后触发."""
        ...

    def on_line_deleted(self, name: str) -> None:
        """delete_line() 后触发."""
        ...


class NullHooks:
    """全 no-op 默认实现."""

    def on_record_staged(self, line: str, record: MomentRecord) -> None:
        pass

    def on_commit(self, line: str, view: CommitView) -> None:
        pass

    def on_reinterpreted(self, commit_id: str, view: CommitView) -> None:
        pass

    def on_line_created(self, name: str, from_ref: BranchRef | None) -> None:
        pass

    def on_line_deleted(self, name: str) -> None:
        pass


# ── 异常 ─────────────────────────────────────────────────────────────────────


class MementoError(Exception):
    """memento 体系异常基类."""


class ReadonlyLineError(MementoError):
    """对只读 line handle 调用了写操作."""


class LineNotFoundError(MementoError):
    """line name 不存在."""


class CommitNotFoundError(MementoError):
    """commit_id 不存在."""


class MomentFrozenError(MementoError):
    """moment 已随 commit 搬入 commit 目录, staging 无该 id 的可写槽位."""


class MomentNotInCommitError(MementoError):
    """BranchRef.moment_id 不在目标 commit 的成员内."""


class EmptyStagingError(MementoError):
    """staging 为空, 禁止 commit."""


# ── Line (时间线 handle, Protocol) ───────────────────────────────────────────


@runtime_checkable
class Line(Protocol):
    """
    绑定到一条 branch (时间线) 的操作句柄.

    branch = name + BranchRef + staging. 获取方式: memento.get_line(name).
    跨 owner: get_line(name, origin=other) 返回 readonly handle.
    """

    @property
    def name(self) -> str:
        """线名."""
        ...

    @property
    def ref(self) -> BranchRef | None:
        """当前指向. None = root line 从未 commit."""
        ...

    @property
    def readonly(self) -> bool:
        """只读 handle 上写操作 raise ReadonlyLineError."""
        ...

    # ── 延线 ──

    def record(self, record: MomentRecord) -> None:
        """
        写一条 moment 到 staging. 同 id 覆盖直接追加, 读者取 last-wins.

        :raise ReadonlyLineError:
        :raise MomentFrozenError: 该 id 已冻结在 commit 目录.
        """
        ...

    def commit(
        self,
        text: str = "",
        *,
        kind: Literal["semantic", "mechanical"] = "semantic",
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        boundary_moment_id: str | None = None,
        by: str = "",
    ) -> CommitView:
        """
        冻结 staging → 新 commit.

        :param kind: 'semantic' (模型自宣) 或 'mechanical' (规则自动).
        :param boundary_moment_id: 只冻结 staging 中首次出现序 ≤ 此 id 的前缀 (含),
            剩余留在 staging. 缺省 = 冻结全部.
        :param resumes: 回归的线索的 commit id.
        :param suspends: 挂起的线索名.
        :raise ReadonlyLineError:
        :raise EmptyStagingError:
        """
        ...

    # ── 读 ──

    def staging(self) -> list[MomentRecord]:
        """未冻结的 moments, 按首次写入序."""
        ...

    def log(self) -> list[CommitView]:
        """本线历史 (沿 parent 链回溯)."""
        ...

    def window(self, *, detail_n: int = 10, summary_m: int = -1) -> BranchWindow:
        """
        滑动窗口快路径. detail_n: 最近 N 帧 (含 staging).
        summary_m: 明细区之前的释义摘要数, -1 = 全量.
        """
        ...


# ── Memento (owner facade, ABC) ──────────────────────────────────────────────


class Memento(ABC):
    """
    owner facade. 一个实例绑定一个 owner.

    跨 owner 只读: get_line(name, origin=other) 返回 readonly handle.
    跨 owner 写不存在 — 新思考空间 = 新 owner 的新 Memento 实例.

    退化态: 单 line + 自动 commit 的用例只接触 get_line("main") +
    record/commit/staging/log — fork 相关词汇完全不出现.
    """

    @property
    @abstractmethod
    def owner(self) -> str:
        pass

    # ── line 管理 ──

    @abstractmethod
    def create_line(
        self,
        name: str,
        *,
        from_ref: BranchRef | None = None,
        overlay: dict[str, Any] | None = None,
    ) -> Line:
        """
        开线.

        :param from_ref: fork 起点. 跨 owner 时 from_ref.origin != self.owner.
            None = root line (无前驱, 首次 commit 自本 owner 开始).
        :param overlay: 化身出生注入物. 仅在 from_ref 跨 owner 时有意义.
            落 owner meta.json, 创建后不可变.
        """
        pass

    @abstractmethod
    def get_line(self, name: str, *, origin: str | None = None) -> Line:
        """
        取 line handle. origin=None 表示本 owner; origin != self.owner 时
        返回 readonly handle.
        :raise LineNotFoundError:
        """
        pass

    @abstractmethod
    def list_lines(self) -> list[str]:
        """本 owner 全部 line name."""
        pass

    @abstractmethod
    def delete_line(self, name: str) -> None:
        """
        弃线. ref + staging 随线死, 冻结 commit 永存.
        :raise LineNotFoundError:
        """
        pass

    @abstractmethod
    def reset_line(self, name: str, to: BranchRef) -> None:
        """
        移 ref 不改历史. staging 非空时先自动机械 commit 落锚, 再移 ref.
        :raise LineNotFoundError:
        """
        pass

    # ── commit 读 & 释义 ──

    @abstractmethod
    def show(self, commit_id: str) -> CommitDetail:
        """展开一个 commit 的完整成员 + 全部释义版本. :raise CommitNotFoundError:"""
        pass

    @abstractmethod
    def notes(self, commit_id: str) -> list[CommitNote]:
        """某 commit 的全部释义版本 (行序). 取证接口. :raise CommitNotFoundError:"""
        pass

    @abstractmethod
    def annotate(self, commit_id: str, title: str = "", body: str = "", *, by: str = "") -> CommitView:
        """
        孔径二: 追加一条 commit 释义. 整体替换语义, 原版本永远可寻址.
        :raise CommitNotFoundError:
        """
        pass

    @abstractmethod
    def annotate_moment(
        self, commit_id: str, moment_id: str, threads: Sequence[str], *, by: str = ""
    ) -> None:
        """
        moment 级释义 — 整体替换 threads. 冻结后仍合法 (threads 是释义不是成员).
        跨 owner 只读: 他 owner 的 commit 只经其 owner 实例落盘.
        :raise CommitNotFoundError:
        :raise MomentNotInCommitError:
        """
        pass

    # ── owner 级 ──

    @abstractmethod
    def log(self) -> list[CommitRef]:
        """commits.jsonl 物理时序 — 全部 line 的 commit 按 append 序."""
        pass

    @abstractmethod
    def commit_space(self, commit_id: str) -> str:
        """
        commit 自治目录的绝对路径. 运行时解析, 不进入持久化结构.
        :raise CommitNotFoundError:
        """
        pass
