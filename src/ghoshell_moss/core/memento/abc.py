"""
Memento — 可折叠、可分叉、可追溯的对话历史.

四层模型: Moment (原子) -> Commit (不可变快照) -> MementoBranch (生产序列) -> Memento (owner-scoped 命名空间).

设计要旨:
- commit_id 是 primary anchor. 存储结构可演进, id 不变. 跨 branch / 跨 session / 跨进程的引用一律走 id.
- owner-isolated writing 杜绝并发: 只有 branch owner 可写自己的 staging 和 commits.
- base pointer 回溯 (不复制): Branch B fork 自 A 时, B 沿 BasePointer 回溯 A.
- staging 显式维护: jsonl 行记录"上次 commit 之后的 moment_ids", commit 时冻结清空.
- merge ≡ 带引用的 Message: 见 Memento.make_merge_message.
- ABC 零外部体系依赖: 不引入 Storage / Session / Cell / IoC. 实现注入构造期参数.

参考 FEATURE.md (workstreams/2026/06/momento-mori) 获取完整设计上下文.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, Field

from ghoshell_moss.core.blueprint.memento import Moment, Reaction
from ghoshell_moss.message import Addition, Message, unique_id

__all__ = [
    "Moment",
    "Reaction",
    "Commit",
    "BasePointer",
    "BranchMeta",
    "MementoHooks",
    "NullHooks",
    "MomentStore",
    "MementoBranch",
    "Memento",
    "MementoRef",
    "MementoError",
    "ReadonlyBranchError",
    "BranchNotFoundError",
    "CommitNotFoundError",
]


# ────────────────────────────────────────────────────────────────────────────
# 协议常量 — 强类型 Addition, 未来模型可通过 moss codex get-interface 反射发现
# ────────────────────────────────────────────────────────────────────────────


class MementoRef(Addition):
    """
    指向一个具体 commit 的强类型引用. 作为 Message.additional 中的一项使用.

    生成方: Memento.make_merge_message / Branch.history (commit-summary 行).
    消费方: 调用 MementoRef.read(message) 拿到强类型实例, 再通过
            Memento.get_branch(branch_id, fork=fork).commit_moments(commit_id)
            展开原文.

    设计意图: 不使用 dict + magic string 的弱类型组合 — 关键内部数据结构必须
    可被未来模型反射发现 (get-interface), 字段演进有 pydantic 校验兜底.
    """

    fork: str
    branch_id: str
    commit_id: str
    commit_seq: int

    @classmethod
    def keyword(cls) -> str:
        return "memento.ref"


# ────────────────────────────────────────────────────────────────────────────
# 数据模型 — 新增的快照层
# ────────────────────────────────────────────────────────────────────────────


class Commit(BaseModel):
    """
    一段 staging 冻结后的不可变快照.

    不可 amend. 写入磁盘后内容不变, seq 不变, id 不变.
    seq 是 owner branch 内的本地序号 (1, 2, 3, ...), 给目录列表语义用.
    id 是全局稳定 anchor — 所有外部引用都用它, 不用 seq 也不用 (branch_id, seq) 组合键.
    """

    id: str = Field(
        default_factory=unique_id,
        description="commit 的全局稳定 id. 跨 branch / 跨 session 引用都用它. 写入后不变.",
    )
    seq: int = Field(
        description="所属 branch 内的有序编号 (从 1 开始). 由 commits/ 目录列表语义提供."
                    "不是身份 — 只是 branch 当前的本地序号.",
    )
    summary: str = Field(
        description="commit 的摘要. merge message 的 content, 折叠展示时的代表内容."
                    "由 caller (人/模型/规则) 决定生成方式, ABC 不预设.",
    )
    moment_ids: list[str] = Field(
        description="冻结进此 commit 的 moment ids 有序列表. 与 staging.jsonl 的内容一致.",
    )
    created: AwareDatetime = Field(
        description="commit 创建时间.",
    )


class BasePointer(BaseModel):
    """
    Branch 的 base pointer — 指向它 fork 自的源 branch 上的某个 commit.

    沿 BasePointer 回溯, 可以重建 branch 的完整历史 (own commits + ancestor commits 截至 commit_seq).
    commit_id 与 commit_seq 同时存. commit_id 是检索身份, commit_seq 给目录截断用.
    """

    fork: str = Field(description="父 branch 的 owner (fork 命名空间).")
    branch_id: str = Field(description="父 branch 的 id.")
    commit_id: str = Field(description="fork 起点 commit 的 id.")
    commit_seq: int = Field(description="fork 起点 commit 在父 branch 中的 seq.")


class BranchMeta(BaseModel):
    """
    Branch 的身份卡 — meta.json 内容. 大多数字段不可变, 少数 (name/title/description/updated) 可改.
    """

    branch_id: str = Field(
        default_factory=unique_id,
        description="branch 的全局稳定 id. 不可变.",
    )
    fork: str = Field(
        description="owner 命名空间字符串. ABC 不知道它是什么 — 可以是 cell address, "
                    "也可以是任意调用方约定. 构造期注入, 不可变.",
    )
    name: str = Field(
        default="",
        description="语义标签 (如 'main', 'side-thinker'). 可以 rename, 数据不动."
                    "name 不是身份, branch_id 才是.",
    )
    title: str = Field(default="", description="可读标题.")
    description: str = Field(default="", description="可读描述.")
    recap: str = Field(
        default="",
        description="前情提要. fork 时由调用方写入, 表达"
                    "'此 branch 从某个 commit 分出来, 之前发生过这些'.",
    )
    base: BasePointer | None = Field(
        default=None,
        description="此 branch 的 fork 起点. None 表示 root branch.",
    )
    created: AwareDatetime
    updated: AwareDatetime


# ────────────────────────────────────────────────────────────────────────────
# Hook 协议 — 屏蔽 session 事件
# ────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class MementoHooks(Protocol):
    """
    事件回调集. memento 实现不依赖 Session / Topic / IoC,
    所有"事件外溢"通过此 hook 暴露. wire 阶段才把 hook 接到 session 总线上.

    所有 hook 都是 fire-and-forget — 抛错不应该影响 memento 的核心写入路径,
    由实现侧 try/except 兜底. ABC 不要求实现做异常隔离, 但建议生产实现做.

    单测里 hook 直接是个 list collector 就够了.
    """

    def on_moment_updated(self, branch_id: str, moment: Moment) -> None:
        """Branch.update(moment) 写入 staging 后触发."""
        ...

    def on_commit(self, branch_id: str, commit: Commit) -> None:
        """Branch.commit() 写入新 commit 后触发."""
        ...

    def on_branch_created(self, meta: BranchMeta) -> None:
        """新 branch (root 或 fork) 创建并写入 meta.json 后触发."""
        ...

    def on_branch_switched(self, branch_id: str) -> None:
        """Memento.switch() 改变 owner 当前 current branch 后触发."""
        ...


class NullHooks:
    """所有 hook 都 no-op 的默认实现."""

    def on_moment_updated(self, branch_id: str, moment: Moment) -> None:
        pass

    def on_commit(self, branch_id: str, commit: Commit) -> None:
        pass

    def on_branch_created(self, meta: BranchMeta) -> None:
        pass

    def on_branch_switched(self, branch_id: str) -> None:
        pass


# ────────────────────────────────────────────────────────────────────────────
# MomentStore — 全局 Moment 池
# ────────────────────────────────────────────────────────────────────────────


class MomentStore(ABC):
    """
    Moment 的持久化池. 按 moment.id 寻址, 全局唯一.

    设计上是"跨 owner 共享, 多 writer", 但因为单一 owner 写自己的 branch staging,
    SQLite WAL 兜底就够 — 写入频次 ≤1Hz, 已被 Cache/Parameter 验证.

    owner 不是 Moment 的字段, 而是 store 单独维护的索引列 (用于诊断/筛选, 非语义关键).
    保持 Moment 类纯净, 未来包剥离无负担.
    """

    @abstractmethod
    def put(self, moment: Moment, *, owner: str) -> None:
        """
        写入 (upsert) 一个 moment. 已存在的 id 覆盖.

        owner 是诊断字段, 表示"是谁写入的". 不影响 get 行为.
        实现侧序列化 moment 时建议丢弃 perspectives (动态快照, 持久化无意义),
        保留 compacted_perspectives. 参考 Moment.to_json(exclude_perspectives=True).
        """
        pass

    @abstractmethod
    def get(self, moment_id: str) -> Moment | None:
        """按 id 取 moment. 不存在返回 None."""
        pass

    @abstractmethod
    def get_many(self, moment_ids: Iterable[str]) -> dict[str, Moment]:
        """批量取. 返回 {moment_id -> Moment} 字典, 缺失 id 不在字典里."""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭底层连接. 用于干净的 shutdown."""
        pass


# ────────────────────────────────────────────────────────────────────────────
# 异常
# ────────────────────────────────────────────────────────────────────────────


class MementoError(Exception):
    """memento 体系异常基类."""


class ReadonlyBranchError(MementoError):
    """对只读 branch 调用了写操作 (非 owner 调 update / commit)."""


class BranchNotFoundError(MementoError):
    """branch_id 不存在."""


class CommitNotFoundError(MementoError):
    """commit_id 不存在."""


# ────────────────────────────────────────────────────────────────────────────
# MementoBranch — 单一生产序列
# ────────────────────────────────────────────────────────────────────────────


class MementoBranch(ABC):
    """
    一条 branch — Commit 有序序列 + 一段 staging.

    owner-only 写: readonly=True 时调用 update/commit 抛 ReadonlyBranchError.
    跨 owner 读: 任何 owner 都能通过 Memento.get_branch(branch_id, fork=...) 获取 readonly handle.

    完整历史 = base chain (沿 BasePointer 回溯) + 本 branch 的 own commits + staging.
    """

    @property
    @abstractmethod
    def meta(self) -> BranchMeta:
        """branch 的身份卡. 实时反映 meta.json 内容."""
        pass

    @property
    @abstractmethod
    def readonly(self) -> bool:
        """此 handle 是否只读. owner != memento.owner 时 True."""
        pass

    # ── 写入路径 (owner-only) ──

    @abstractmethod
    def update(self, moment: Moment) -> None:
        """
        把一个 moment 写入 staging.

        Upsert 语义 — 同 id 覆盖. 实现侧:
        1. moment_store.put(moment, owner=self.meta.fork)
        2. append moment.id to staging.jsonl (若 id 已在, 不重复 append)
        3. hooks.on_moment_updated(branch_id, moment)

        :raise ReadonlyBranchError: 当前 handle 是 readonly.
        """
        pass

    @abstractmethod
    def commit(self, summary: str) -> Commit:
        """
        冻结 staging -> 新 commit. seq = 当前 max_seq + 1.

        实现侧:
        1. 读 staging.jsonl 的 moment_ids
        2. 写 commits/{seq:04d}.json (Commit 对象, id 由 unique_id 生成)
        3. 清空 staging.jsonl
        4. 更新 meta.updated
        5. hooks.on_commit(branch_id, commit)

        :param summary: commit summary. 由 caller 决定生成方式.
        :raise ReadonlyBranchError: 当前 handle 是 readonly.
        :raise MementoError: staging 为空 (没有要 commit 的内容).
        """
        pass

    # ── 读取路径 ──

    @abstractmethod
    def head(self) -> Commit | None:
        """最新 commit. None = 此 branch 从未 commit 过 (staging 可能有内容)."""
        pass

    @abstractmethod
    def staging(self) -> list[Moment]:
        """未 commit 的 moments, 按写入顺序. 实时从 staging.jsonl + moment_store 重建."""
        pass

    @abstractmethod
    def own_commits(self) -> list[Commit]:
        """本 branch 自己持有的 commits, 按 seq 升序. 不含 base chain."""
        pass

    @abstractmethod
    def all_commits(self) -> list[Commit]:
        """
        完整历史 commits, 按时间升序.

        实现 = (沿 BasePointer 回溯, 截至 base.commit_seq 的 ancestor own_commits) + self.own_commits().
        """
        pass

    @abstractmethod
    def get_commit(self, commit_id: str) -> Commit | None:
        """按 id 取 commit. 检索范围 = all_commits(). 不存在返回 None."""
        pass

    @abstractmethod
    def commit_moments(self, commit_id: str) -> list[Moment]:
        """
        展开一个 commit 的完整 moment 序列 (从 moment_store 拉回).

        memento channel 的 `show <commit_id>` 用这个. 折叠语义的"按需展开"基石.
        :raise CommitNotFoundError: commit_id 不在 all_commits 中.
        """
        pass

    @abstractmethod
    def history(
        self,
        *,
        detail_n: int = 10,
        summary_m: int = -1,
    ) -> Iterable[Message]:
        """
        sliding window 上下文窗口 — 默认渲染策略.

        组成 (按时间序):
        1. base chain 的最早 ancestor commit 起, 之前 M 个 commit 的 summary (作为 Message,
           tag='commit-summary', additional 携带 memento.ref).
        2. 最近 N 个 moment 的全量内容 (含 staging 中的, 含已 commit 但在 detail 窗口内的).

        :param detail_n: 最近 N 帧 moment 全量展开. <=0 表示不要明细.
        :param summary_m: 之前 M 个 commit 的 summary. -1 表示全量, 0 表示不要 summary.
        """
        pass


# ────────────────────────────────────────────────────────────────────────────
# Memento — owner-scoped 命名空间 + 顶层 facade
# ────────────────────────────────────────────────────────────────────────────


class Memento(ABC):
    """
    顶层 facade. 一个 Memento 实例绑定一个 owner (构造期参数).

    跨 owner 读: get_branch(branch_id, fork=other_owner) 返回 readonly handle.
    跨 owner 写: 不允许. 想"开新思考空间" => 构造一个新的 Memento (owner 不同),
                共享同一个 moment_store + branches_storage 即可.

    所有方法只对"本 owner"的 current branch 有写副作用.
    """

    @property
    @abstractmethod
    def owner(self) -> str:
        """此 Memento 实例绑定的 owner 命名空间."""
        pass

    # ── 当前指针 ──

    @abstractmethod
    def current(self) -> MementoBranch:
        """
        本 owner 当前 current branch (writeable).

        若从未存在: 自动创建一个 root branch (name='main'), 返回它.
        current 指针存储在 _branches.json 的 'current' 字段.
        """
        pass

    @abstractmethod
    def switch(self, branch_id: str) -> None:
        """
        切换 current 指针. 触发 hooks.on_branch_switched.
        :raise BranchNotFoundError: branch_id 不在本 owner.
        """
        pass

    # ── branch 管理 ──

    @abstractmethod
    def checkout(
        self,
        *,
        base_fork: str,
        base_branch_id: str,
        base_commit_id: str | None = None,
        new_name: str = "",
        recap: str = "",
    ) -> MementoBranch:
        """
        从任意 owner 的某个 commit fork 出新 branch. 新 branch 的 owner = self.owner.

        :param base_fork: 源 branch 的 fork (owner) 命名空间.
        :param base_branch_id: 源 branch id.
        :param base_commit_id: fork 起点 commit. None = 从源 branch 的 head fork.
        :param new_name: 新 branch 的语义标签 (可空).
        :param recap: 写入新 branch 的 meta.recap.
        :raise BranchNotFoundError: 源 branch 不存在.
        :raise CommitNotFoundError: base_commit_id 不在源 branch.
        :raise MementoError: 源 branch 从未 commit (head=None) 且 base_commit_id=None.
        """
        pass

    @abstractmethod
    def get_branch(
        self,
        branch_id: str,
        *,
        fork: str | None = None,
    ) -> MementoBranch:
        """
        获取任意 branch handle.

        :param fork: 默认 self.owner. 跨 owner 时手动指定.
                     fork != self.owner 时返回的 handle 必然 readonly=True.
        :raise BranchNotFoundError.
        """
        pass

    @abstractmethod
    def list_branches(self, fork: str | None = None) -> list[BranchMeta]:
        """
        列 branches.
        :param fork: None = self.owner. 字符串 = 指定 owner.
        """
        pass

    @abstractmethod
    def list_forks(self) -> list[str]:
        """branches/ 目录下所有 owner 命名空间. 跨 owner 浏览用."""
        pass

    # ── merge as message ──

    @abstractmethod
    def make_merge_message(
        self,
        branch_id: str,
        commit_id: str,
        *,
        fork: str | None = None,
    ) -> Message:
        """
        把一个 commit 包装成 Message — merge ≡ 带引用的 message 的显式 API.

        生成的 Message:
          content = commit.summary
          additional 持有一个 MementoRef (强类型, 通过 MementoRef.read 反查).

        调用者把它当普通 Message 喂回主链路 (如 session.output / staging.update 等).
        接收方通过 MementoRef.read(message) 拿到引用, 必要时
        get_branch(...).commit_moments(commit_id) 展开原文.

        :param fork: 默认 self.owner. 引用别人的 commit 时指定.
        :raise BranchNotFoundError / CommitNotFoundError.
        """
        pass

    # ── 生命周期 ──

    @abstractmethod
    def close(self) -> None:
        """关闭底层资源 (主要是 moment_store)."""
        pass
