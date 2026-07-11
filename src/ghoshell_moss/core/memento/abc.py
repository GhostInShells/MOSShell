"""
Memento — 轨迹第一公民的认知基建. 契约层 ABC.

四层模型 (FORMAT.md §1-§5):
    MomentRecord (信封) -> Commit (冻结快照) -> MementoBranch (生产序列) -> Memento (owner 命名空间).

一句话核心: **身份和成员冻结, 释义永远开放.**

- commit 是 fork 与重绘的锚点. commit_id 是 stable id (非 content-addressable —
  hash(content) 会让释义后补变成不可能, 不要实现 hash-based id).
- 成员 (commit_id + moment_ids + base) 不可变; 释义 (body/threads) 追加新版本,
  同 id, last-wins (定序规则钉死在 FORMAT.md §2.2).
- 信封模型: memento 不理解 payload. 本模块 MUST NOT import 任何 payload schema
  (含 ghoshell_moss 的 Moment / Message). 强类型编解码在信封之上 (codec 模块).
- owner-isolated writing: 只有 owner 可写自己的池分片 / staging / commits / 释义.
  跨 owner 只读. 旁路释义改写经由 owner 实例落盘 (孔径二, FEATURE.md §3.4).
- 化身只能从 commit 出生, 永不从 staging. divergence prompt 的家在
  BranchMeta.overlay, 不属于对话历史, 禁止进 staging.

磁盘格式契约见同目录 FORMAT.md — 实现与其冲突时以 FORMAT.md 为准.
设计上下文见 workstreams/2026/06/momento-mori/FEATURE.md.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

from pydantic import AwareDatetime, BaseModel, Field
from ulid import ULID

__all__ = [
    # trailer 规范 (FORMAT.md §6)
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
    "new_branch_id",
    # 数据模型
    "CommitKind",
    "MomentRecord",
    "Commit",
    "CommitNote",
    "CommitView",
    "BasePointer",
    "BranchMeta",
    "BranchWindow",
    # hook
    "MementoHooks",
    "NullHooks",
    # ABC
    "MomentPool",
    "MementoBranch",
    "Memento",
    # 异常
    "MementoError",
    "ReadonlyBranchError",
    "BranchNotFoundError",
    "CommitNotFoundError",
    "MomentFrozenError",
    "EmptyStagingError",
]


def _now() -> datetime:
    """FORMAT.md §2: 时间必须 timezone-aware. naive datetime 非法."""
    return datetime.now().astimezone()


def new_commit_id() -> str:
    """FORMAT.md §2.1: commit id = 'cmt_' + ULID. 前缀保证 grep 可反查."""
    return f"cmt_{ULID()}"


def new_branch_id() -> str:
    """FORMAT.md §2.1: branch id = 'brn_' + ULID."""
    return f"brn_{ULID()}"


# ────────────────────────────────────────────────────────────────────────────
# Trailer 规范 — FORMAT.md §6 的唯一 canonical 实现
# (该节同时是释义生成旁路的 prompt 约束, 规范与 prompt 不写两份)
# ────────────────────────────────────────────────────────────────────────────

TRAILER_THREAD = "Thread"
TRAILER_RESUMES = "Resumes"
TRAILER_SUSPENDS = "Suspends"
TRAILER_KIND = "Kind"
TRAILER_MEMENTO_REF = "Memento-Ref"

_TRAILER_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*): (.+)$")

CommitKind = Literal["semantic", "mechanical"]
"""
commit 的双重身份 (FEATURE.md §3.2):
- semantic: 语义锚点, 模型自宣 (如 CTML `<memento:commit summary="..."/>`).
- mechanical: 机械快照, 规则触发, 为 fork 服务, 正文可空.
共存, 规则是主力, 模型自宣是加分项. 以初始释义行的 `Kind:` trailer 区分.
"""


def split_trailers(body: str) -> tuple[str, list[tuple[str, str]]]:
    """
    解析 body -> (正文, trailer 列表). FORMAT.md §6.

    trailer 块 = body 末尾连续的 `Key: value` 行, 与正文以空行分隔
    (全文只有 trailer 无正文时不需要空行). 未知 key 原样保留, 不报错.
    同 key 可重复, 保持出现顺序.
    """
    lines = body.splitlines()
    # 从尾部收集连续的 trailer 行
    idx = len(lines)
    while idx > 0 and _TRAILER_LINE.match(lines[idx - 1]):
        idx -= 1
    tail = lines[idx:]
    head = lines[:idx]
    if not tail:
        return body.rstrip("\n"), []
    if head and any(line.strip() for line in head):
        # 有正文时, trailer 块前必须有空行, 否则视作正文的一部分
        if head[-1].strip():
            return body.rstrip("\n"), []
    trailers: list[tuple[str, str]] = []
    for line in tail:
        m = _TRAILER_LINE.match(line)
        assert m is not None
        trailers.append((m.group(1), m.group(2)))
    text = "\n".join(head).rstrip("\n")
    return text, trailers


def join_trailers(text: str, trailers: Iterable[tuple[str, str]]) -> str:
    """组装 body = 正文 + 空行 + trailer 块. split_trailers 的逆操作."""
    trailer_lines = [f"{key}: {value}" for key, value in trailers]
    text = text.rstrip("\n")
    if not trailer_lines:
        return text
    if not text:
        return "\n".join(trailer_lines)
    return text + "\n\n" + "\n".join(trailer_lines)


def trailer_values(trailers: Iterable[tuple[str, str]], key: str) -> list[str]:
    """取某个 key 的全部值 (同 key 可重复, 如多条 Thread:). key 大小写敏感."""
    return [v for k, v in trailers if k == key]


# ────────────────────────────────────────────────────────────────────────────
# 数据模型 — 与 FORMAT.md 的行格式一一对应
# ────────────────────────────────────────────────────────────────────────────


class MomentRecord(BaseModel):
    """
    Moment 池的信封 (FORMAT.md §3.1). memento 唯一认识的 moment 形态.

    payload 对 memento 不透明: 原样透传, 字节不动, 不解析不改写.
    `type` 标识 payload schema, schema 归生产者所有 (如 'moss.moment/v1',
    其编解码见 codec 模块, 不在契约层).

    可变性 (FORMAT.md §3.2): 同 id 覆盖写 = 新对象共享 id, last-wins;
    一旦成为任何 commit 的成员即冻结, 再写 raise MomentFrozenError.
    threads 是释义层, 冻结后仍可经 annotate 更新 (整体替换).
    """

    id: str = Field(description="生产者 id, 原样透传. 非空, [A-Za-z0-9._\\-]{1,128}, owner 池内唯一.")
    created: AwareDatetime = Field(default_factory=_now, description="信封创建时间.")
    type: str = Field(description="payload schema 标识, 如 'moss.moment/v1'.")
    payload: dict[str, Any] = Field(description="任意 JSON object. memento 原样透传.")
    threads: list[str] = Field(default_factory=list, description="线索标签, 写入时标注 (FEATURE.md §3.6), 可经释义更新.")
    by: str = Field(default="", description="写入者标识 (模型名/规则 id), 诊断用.")


class Commit(BaseModel):
    """
    commit 成员行 (FORMAT.md §5.1). 写入后不可变 — 成员一变,
    所有子 branch 冻结的祖先链集体失效.

    seq 是 branch 内本地序号 (从 1 起连续), 给目录列表语义用;
    id 是全局稳定锚点, 一切外部引用用它.
    释义 (summary/threads/kind) 不在此 — 在 CommitNote, 可追加新版.
    """

    id: str = Field(default_factory=new_commit_id, description="全局稳定 id, 'cmt_' 前缀. 写入后不变.")
    seq: int = Field(description="branch 内序号, 从 1 起连续递增.")
    moment_ids: list[str] = Field(description="staging 冻结时刻的去重有序 moment id 列表. 非空.")
    created: AwareDatetime = Field(default_factory=_now)


class CommitNote(BaseModel):
    """
    commit 释义行 (FORMAT.md §5.2). 追加式多版本, last-wins.
    body = 正文 + trailer 块 (FORMAT.md §6), 整体替换语义.
    历史版本永远可寻址 — "再巩固 + 取证"双得.
    """

    ref: str = Field(description="所释义的 commit id.")
    body: str = Field(default="", description="正文 + trailer 块. 机械 commit 可只有 trailer.")
    ts: AwareDatetime = Field(default_factory=_now, description="仅展示与诊断. 不参与 last-wins 定序.")
    by: str = Field(default="", description="释义写入者.")

    def text(self) -> str:
        """正文部分 (剥离 trailer 块)."""
        return split_trailers(self.body)[0]

    def trailers(self) -> list[tuple[str, str]]:
        return split_trailers(self.body)[1]

    def threads(self) -> list[str]:
        return trailer_values(self.trailers(), TRAILER_THREAD)

    def kind(self) -> str:
        values = trailer_values(self.trailers(), TRAILER_KIND)
        return values[-1] if values else ""


class CommitView(BaseModel):
    """
    Commit + 当前释义 (last-wins 解析结果) 的读取视图.

    note_seq 是渲染打戳 (FORMAT.md §5.2): 所读释义是第几个版本 (0-based 行序).
    记忆可变的系统里 "当时模型看见什么" 必须可重建, 否则行为不可归因 —
    渲染方持有 (commit_id, note_seq) 即可复原当时视图.
    """

    commit: Commit
    note: CommitNote = Field(description="当前 (最新) 释义.")
    note_seq: int = Field(description="释义版本序号, 0-based. 渲染打戳用.")

    @property
    def id(self) -> str:
        return self.commit.id

    @property
    def seq(self) -> int:
        return self.commit.seq

    def summary(self) -> str:
        """当前释义的正文部分. 折叠展示时的代表内容."""
        return self.note.text()


class BasePointer(BaseModel):
    """
    branch 的 fork 起点 (FORMAT.md §4.1). 创建后永不变 —
    这是祖先链可以在 fork 时刻冻结的前提.
    """

    fork: str = Field(description="源 branch 的 owner 命名空间.")
    branch_id: str = Field(description="源 branch id.")
    commit_id: str = Field(description="fork 起点 commit id.")
    commit_seq: int = Field(description="起点 commit 在源 branch 的 seq. 目录截断用.")


class BranchMeta(BaseModel):
    """
    branch 身份卡 — meta.json (FORMAT.md §4.1).

    ancestry: fork 时刻一次性冻结的展平祖先链 (自最老祖先到直接 base).
    O(d) 链上溯坍缩为 O(1) — 可变父链系统里这是危险反规范化,
    这里因成员不可变而免费. 读者发现 ancestry 与实际 base 链不一致 MUST 抛错.

    overlay: 化身出生注入物的家 (divergence prompt 等).
    不属于对话历史, MUST NOT 进 staging. 创建后不可变.
    """

    branch_id: str = Field(default_factory=new_branch_id, description="全局稳定 id, 'brn_' 前缀. 不可变.")
    fork: str = Field(description="owner 命名空间. memento 不解释其语义. 不可变.")
    name: str = Field(default="", description="语义标签 (如 'main'). 可改, name 不是身份.")
    title: str = Field(default="", description="可读标题. 释义性, 可改.")
    description: str = Field(default="", description="可读描述. 释义性, 可改.")
    base: BasePointer | None = Field(default=None, description="fork 起点. None = root branch. 不可变.")
    ancestry: list[BasePointer] = Field(
        default_factory=list,
        description="冻结的展平祖先链, 最老在前. root 为空. 最后一项等于 base. 不可变.",
    )
    overlay: dict[str, Any] = Field(
        default_factory=dict,
        description="化身出生注入物 (divergence prompt 等). 创建后不可变.",
    )
    created: AwareDatetime = Field(default_factory=_now)
    updated: AwareDatetime = Field(default_factory=_now)


class BranchWindow(BaseModel):
    """
    每轮渲染的快路径视图 (FEATURE.md §3.5): O(K+m), 不走 base 链.
    化身出生成本 = 一次窗口渲染 — 出生必须便宜, 否则并行扇出在延迟上不成立.

    渲染为模型可读消息 (Message) 是 porcelain 层的事, 契约层只给结构.
    """

    summaries: list[CommitView] = Field(description="折叠区: 之前 M 个 commit 的当前释义, 时间升序.")
    details: list[MomentRecord] = Field(description="明细区: 最近 N 帧 (含 staging), 时间升序.")


# ────────────────────────────────────────────────────────────────────────────
# Hook 协议 — 事件外溢的唯一出口, wire 阶段才接到 session/matrix 总线
# ────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class MementoHooks(Protocol):
    """
    fire-and-forget 事件回调. memento 实现不依赖 Session / Topic / IoC.
    hook 抛错不得影响核心写入路径 (实现侧 try/except 兜底).
    单测里 hook 直接用 list collector.
    """

    def on_record_staged(self, branch_id: str, record: MomentRecord) -> None:
        """Branch.update() 写入池 + staging 后触发."""
        ...

    def on_commit(self, branch_id: str, view: CommitView) -> None:
        """Branch.commit() 完成后触发. view 含初始释义."""
        ...

    def on_reinterpreted(self, branch_id: str, view: CommitView) -> None:
        """Branch.reinterpret() 追加释义后触发 (孔径二). view 为新版本."""
        ...

    def on_branch_created(self, meta: BranchMeta) -> None:
        """root 或 fork branch 创建后触发."""
        ...

    def on_branch_switched(self, branch_id: str) -> None:
        """Memento.switch() 之后触发."""
        ...


class NullHooks:
    """全 no-op 默认实现."""

    def on_record_staged(self, branch_id: str, record: MomentRecord) -> None:
        pass

    def on_commit(self, branch_id: str, view: CommitView) -> None:
        pass

    def on_reinterpreted(self, branch_id: str, view: CommitView) -> None:
        pass

    def on_branch_created(self, meta: BranchMeta) -> None:
        pass

    def on_branch_switched(self, branch_id: str) -> None:
        pass


# ────────────────────────────────────────────────────────────────────────────
# 异常
# ────────────────────────────────────────────────────────────────────────────


class MementoError(Exception):
    """memento 体系异常基类."""


class ReadonlyBranchError(MementoError):
    """对只读 branch handle 调用了写操作."""


class BranchNotFoundError(MementoError):
    """branch_id 不存在."""


class CommitNotFoundError(MementoError):
    """commit_id 不存在."""


class MomentFrozenError(MementoError):
    """moment 已被 commit 冻结, 拒绝覆盖写 (FORMAT.md §3.2)."""


class EmptyStagingError(MementoError):
    """staging 为空, 禁止 commit (FORMAT.md §5.1)."""


# ────────────────────────────────────────────────────────────────────────────
# MomentPool — per-owner 分片的信封池
# ────────────────────────────────────────────────────────────────────────────


class MomentPool(ABC):
    """
    MomentRecord 的持久化池 (FORMAT.md §3). 按 record.id 寻址.

    写面按 owner 分片 (moments/{owner}/...), 单写者 append-only, 无锁;
    读全局. owner 是分片键, 不是 record 字段.
    """

    @abstractmethod
    def put(self, record: MomentRecord, *, owner: str) -> None:
        """
        追加一条记录行. 同 id 重复 put = 覆盖写 (last-wins),
        覆盖行落在该 id 首次出现的同一文件 (FORMAT.md §3.2/§3.4).

        冻结检查在 Branch 层做 (池不知道 commit 的存在).
        """
        pass

    @abstractmethod
    def annotate(self, moment_id: str, threads: Sequence[str], *, owner: str, by: str = "") -> None:
        """
        追加 moment 级释义行: 整体替换 threads (FORMAT.md §3.3).
        payload 永远不可经此改写.
        :raise MementoError: moment_id 不在该 owner 池中.
        """
        pass

    @abstractmethod
    def get(self, moment_id: str) -> MomentRecord | None:
        """按 id 取当前视图 (记录 last-wins + threads 释义 last-wins 已解析). 不存在返回 None."""
        pass

    @abstractmethod
    def get_many(self, moment_ids: Iterable[str]) -> dict[str, MomentRecord]:
        """批量取. 缺失 id 不在返回字典中."""
        pass

    @abstractmethod
    def close(self) -> None:
        pass


# ────────────────────────────────────────────────────────────────────────────
# MementoBranch — Commit 有序序列 + staging 活跃写面
# ────────────────────────────────────────────────────────────────────────────


class MementoBranch(ABC):
    """
    一条 branch = 冻结的 commit 序列 + 一段 staging (FORMAT.md §4/§5).

    owner-only 写: readonly handle 上调用写方法 raise ReadonlyBranchError.
    完整历史 = 冻结祖先链 (meta.ancestry) + own commits + staging.
    """

    @property
    @abstractmethod
    def meta(self) -> BranchMeta:
        """身份卡, 实时反映 meta.json."""
        pass

    @property
    @abstractmethod
    def readonly(self) -> bool:
        pass

    # ── 写入路径 (owner-only) ──

    @abstractmethod
    def update(self, record: MomentRecord) -> None:
        """
        写入池 + staging. 同 id 覆盖 (staging 保持首次出现的位置序, FORMAT.md §4.2).

        :raise ReadonlyBranchError:
        :raise MomentFrozenError: 该 id 已被本 branch 历史上的 commit 冻结.
        """
        pass

    @abstractmethod
    def annotate_moment(self, moment_id: str, threads: Sequence[str], *, by: str = "") -> None:
        """moment 级释义 (整体替换 threads). 冻结后仍合法 — threads 是释义不是成员."""
        pass

    @abstractmethod
    def commit(
        self,
        text: str = "",
        *,
        kind: CommitKind,
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        by: str = "",
    ) -> CommitView:
        """
        冻结 staging -> 新 commit. 原子动作 = 成员行 + 初始释义行 + truncate staging.

        body 由参数按 FORMAT.md §6 组装 (Kind trailer 必有 — 用参数强制,
        自由拼 body 导致漏 Kind 是可预见的错误).
        kind='mechanical' 时 text 可空 (机械快照不编造正文);
        kind='semantic' 是模型自宣的语义锚点.

        :param resumes: 重入锚 — 回归了哪些被搁置线索 (值为该线索最后的 commit id).
        :param suspends: 本处挂起了哪些线索 (值为线索名).
        :raise ReadonlyBranchError:
        :raise EmptyStagingError: staging 为空.
        """
        pass

    @abstractmethod
    def reinterpret(self, commit_id: str, body: str, *, by: str = "") -> CommitView:
        """
        追加一条释义新版本 (孔径二: 记忆再巩固). 整体替换语义, 原版本永远可寻址.

        body 是完整的 正文+trailer 文本 — 与 commit() 不同, 改写场景需要
        自由重组 trailer (如修正错标的 Thread). caller 可用 join_trailers 组装.

        :raise ReadonlyBranchError:
        :raise CommitNotFoundError: commit_id 不是本 branch 的 own commit.
            祖先的释义归祖先的 owner — 跨 owner 改写不经过这里 (孔径二经
            owner 实例落盘, FORMAT.md §1).
        """
        pass

    # ── 读取路径 ──

    @abstractmethod
    def head(self) -> CommitView | None:
        """最新 own commit. None = 从未 commit (staging 可能有内容)."""
        pass

    @abstractmethod
    def staging(self) -> list[MomentRecord]:
        """未冻结的 records, 按首次写入序. 从 staging.jsonl + 池实时重建."""
        pass

    @abstractmethod
    def own_commits(self) -> list[CommitView]:
        """本 branch 自己的 commits (当前释义视图), seq 升序. 不含祖先."""
        pass

    @abstractmethod
    def all_commits(self) -> list[CommitView]:
        """
        完整历史: 沿冻结祖先链 (每段截至 commit_seq) + own commits, 时间升序.
        慢路径 — 回溯 API. 每轮渲染走 window().
        """
        pass

    @abstractmethod
    def get_commit(self, commit_id: str) -> CommitView | None:
        """按 id 取 (检索范围 = all_commits). 不存在返回 None."""
        pass

    @abstractmethod
    def commit_records(self, commit_id: str) -> list[MomentRecord]:
        """
        展开一个 commit 的完整成员序列. `show <commit_id>` (缺页中断) 的基石.
        :raise CommitNotFoundError:
        """
        pass

    @abstractmethod
    def notes(self, commit_id: str) -> list[CommitNote]:
        """
        某 commit 的全部释义版本, 行序 (即版本序). 取证接口 —
        "当时模型看见什么" 由 (commit_id, note_seq) 从这里复原.
        :raise CommitNotFoundError:
        """
        pass

    @abstractmethod
    def window(self, *, detail_n: int = 10, summary_m: int = -1) -> BranchWindow:
        """
        滑动窗口快路径 (O(K+m), 不走 base 链回扫).

        :param detail_n: 最近 N 帧全量明细 (含 staging). <=0 表示不要明细.
        :param summary_m: 明细区之前 M 个 commit 的释义摘要. -1 全量, 0 不要.
        """
        pass


# ────────────────────────────────────────────────────────────────────────────
# Memento — owner-scoped 命名空间 facade
# ────────────────────────────────────────────────────────────────────────────


class Memento(ABC):
    """
    顶层 facade. 一个实例绑定一个 owner (构造期参数).

    跨 owner 读: get_branch(branch_id, fork=other) 返回 readonly handle.
    跨 owner 写: 不存在. 要新思考空间 => 新 owner 的新 Memento 实例,
    共享同一存储根即可.
    """

    @property
    @abstractmethod
    def owner(self) -> str:
        pass

    # ── 当前指针 ──

    @abstractmethod
    def current(self) -> MementoBranch:
        """
        本 owner 的 current branch (writeable). HEAD.json 缺失时
        自动创建 root branch (name='main') — 退化态 (蠢记忆) 的全部入口.
        """
        pass

    @abstractmethod
    def switch(self, branch_id: str) -> None:
        """
        切换 current 指针.
        :raise BranchNotFoundError: branch_id 不属于本 owner.
        """
        pass

    # ── fork 边界: 化身只能从 commit 出生, 永不从 staging ──

    @abstractmethod
    def checkout(
        self,
        *,
        base_fork: str,
        base_branch_id: str,
        base_commit_id: str | None = None,
        name: str = "",
        overlay: dict[str, Any] | None = None,
    ) -> MementoBranch:
        """
        从任意 owner 的某个 commit fork 出新 branch (owner = self.owner).
        fork 时刻冻结展平祖先链进 BranchMeta (= 源的 ancestry + 源的截断点).

        :param base_commit_id: fork 起点. None = 源 branch 的 head.
            staging 永远不是合法出生点 — 没有参数能表达它.
        :param overlay: 出生注入物 (divergence prompt 等). 创建后不可变.
        :raise BranchNotFoundError: 源 branch 不存在.
        :raise CommitNotFoundError: base_commit_id 不在源 branch 历史中.
        :raise MementoError: 源 branch 从未 commit 且 base_commit_id=None.
        """
        pass

    @abstractmethod
    def get_branch(self, branch_id: str, *, fork: str | None = None) -> MementoBranch:
        """
        取任意 branch handle. fork 默认 self.owner;
        fork != self.owner 时 handle 必然 readonly.
        :raise BranchNotFoundError:
        """
        pass

    @abstractmethod
    def list_branches(self, fork: str | None = None) -> list[BranchMeta]:
        """列 branch 身份卡. fork=None 表示本 owner."""
        pass

    @abstractmethod
    def list_forks(self) -> list[str]:
        """branches/ 下所有 owner 命名空间. 跨 owner 浏览用."""
        pass

    # ── 生命周期 ──

    @abstractmethod
    def close(self) -> None:
        pass
