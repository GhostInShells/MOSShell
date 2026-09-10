"""memento — lean cognitive-trajectory index (RC 版设计).

本文件是契约层 (abstract class + 数据模型). 设计比 v3 (abc.py) 大幅收敛:

- commit 是「可还原一个 session 的锚点」, **不承载 moment**. moment 字节住在 agent
  session 里, memento 只拿 metadata 里的钥匙 (约定 ``session_id`` + ``tail``) 指过去.
  还原逻辑归消费者 (dolores), memento 只存取、不解释.
- branch = 一个目录. 目录下: ``meta.json`` / ``commits.jsonl`` (正序 append-only, 权威) /
  ``commit_notes.jsonl`` (旁路摘要, 可丢可重建).
- fork 是**引用** (读父支 + 子支两个 commits.jsonl 拼一条连续轨迹), 不复制.
- **单一真值 message**: commit 是纯锚点, 不带 message; message 只住 Note (一个家),
  last-wins. ``commit(message=...)`` 是便捷糖 = 锚点 + 种子一条 Note, 但存储上 message
  仍只落在 Note 这一行. title = message 首行 (截断), body = 其余 (对齐 git ``-m``).
- 本版拿掉: confluence / segment 一级公民 / commit 独立目录 / staging / 双源 message.

memento 只绑定一个本地 path, 与存储后端不强相关; 契约面不承接存储实现.
操作面用 ABC 勾勒, 数据模型用 pydantic 具体化. 实现层主权归模型, 契约层人类 review.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, Field
import ulid

__all__ = [
    "CommitRef",
    "Note",
    "BranchRef",
    "ForkRef",
    "BranchMeta",
    "CommitSummary",
    "BranchView",
    "Branch",
    "Memento",
]


def _unique_id() -> str:
    return str(ulid.ULID())


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CommitRef(BaseModel):
    """一个 commit 锚点 — commits.jsonl 的一行. **纯锚点, 不带 message.**

    不承载 moment. metadata 装「足以还原一个 session 的钥匙」, 约定 key:
    ``session_id`` + ``tail``. memento 对 metadata 只 get/set, 不解析.
    该 commit 的 message 住在对应的 Note 里, 单一真值.
    """

    id: str = Field(default_factory=_unique_id)
    metatype: str = Field(default="", description="生产 commit 的类型, 如 'session'.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="不透明扩展. 惯用约定 {'session_id':..., 'tail':...} 作为还原钥匙.",
    )
    created: AwareDatetime = Field(default_factory=_now_utc)


class Note(BaseModel):
    """一个 commit 的唯一摘要 — commit_notes.jsonl 的一行.

    side-channel: 生产者 (也是消费者) 自己写读, 生产它慢一点没关系, 不依赖严格有序.
    同 ``commit_id`` 后写覆盖前写 (last-wins). message 是 commit 的唯一 message;
    title = 首行 (截断), body = 其余. 篇幅由生产者保障 (准入责任在生产者).
    """

    commit_id: str = Field(...)
    message: str = Field(default="", description="commit 的摘要 message (对齐 git -m).")


class BranchRef(BaseModel):
    """指向一个 branch 的当前指针 — ``{branch_name}.ref.json`` 的内容."""

    name: str = Field(...)
    description: str = Field(default="")
    branch_id: str = Field(...)
    created: AwareDatetime = Field(default_factory=_now_utc)


class ForkRef(BaseModel):
    """fork 引用 — 定位父支 + 锚点 commit.

    ``branch_id`` 定位父支的 commits.jsonl (per-branch 存储下, commit_id 本身无法
    定位文件). ``commit_id`` 是父支里 fork 的锚点. 父支 name 反查 branches.jsonl 即可,
    故不在此冗余. 分支里要极窄, 刻意不带 name/description/created.
    """

    branch_id: str = Field(description="父支 branch_id (引用, 非复制).")
    commit_id: str = Field(description="父支里 fork 的锚点 commit id.")


class BranchMeta(BaseModel):
    """branch 创建时的元信息 — ``branches/{branch_id}/meta.json``."""

    branch_id: str = Field(...)
    name: str = Field(...)
    description: str = Field(default="")
    metatype: str = Field(default="", description="branch 生产时的类型.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="可扩展元数据.")
    fork_from: ForkRef | None = Field(
        default=None,
        description="fork 引用 (父支指针 + 锚点 commit). 非 fork 的 root branch 为 None.",
    )
    created: AwareDatetime = Field(default_factory=_now_utc)


class CommitSummary(BaseModel):
    """一个 commit 的读侧投影: id + 来源 Note 的 message + 派生 seq.

    title = message 首行 (截断), body = 其余. 无 Note 时为 message 空. 渲染截断由
    消费者 (render) 负责, 本模型只做确定性派生.

    ``seq`` 是 branch 内派生序列 (1-based), 读时按 commits 顺序算、不落盘. 看位置用 seq,
    引用用 id (全局唯一).
    """

    id: str
    message: str = Field(default="")
    seq: int = Field(description="branch 内派生序列 (1-based), 读时算, 不存; 看用 seq, 引用用 id.")

    @property
    def title(self) -> str:
        return self.message.split("\n", 1)[0]

    @property
    def body(self) -> str:
        lines = self.message.split("\n")
        return "\n".join(lines[1:])


class BranchView(BaseModel):
    """branch 的预算化上下文视图 — 读支的投影.

    折叠策略 (近详远粗, 确定性): ``latest`` 最近 N 条给 detail; ``history`` 更早的
    commits 折叠为摘要 (优先 Note); ``previous`` 是 fork 父支的压实 recap.

    预算保证: 生产者保障 Note message 篇幅 (准入); render 动作由消费者签发. memento
    侧不强制预算, 只做投影.
    """

    name: str
    description: str
    branch_id: str
    created: AwareDatetime
    commit_id: str = Field(description="最近 commit 的 id, 供「跟这个 commit 对话」.")
    previous: "BranchView | None" = Field(default=None, description="fork 父支 (引用).")
    history: list[CommitSummary] = Field(default_factory=list, description="更早 commits, 折叠摘要.")
    latest: list[CommitSummary] = Field(default_factory=list, description="最近 commits, detail.")
    commits_total: int = Field(default=0)


class Branch(ABC):
    """一个指向目录的 branch 对象. 目录下: meta.json / commits.jsonl / commit_notes.jsonl.

    读写两级, 语义是「是否要重新碰磁盘 / 是否要获取写权限」:

    - 同步读 = 缓存的快路径; ``a<name>`` = 真磁盘 IO (aiofiles), docstring 标
      **"IO-costly 性能开销"**. 提供显式 ``a*`` 是为给调用方一个卸载点, 免得它被迫
      ``asyncio.to_thread`` + 手包 task 而遗忘.
    - ``async with branch:`` 是**写权限门控**: 获取 branch 写权限或快速失败. 这是非强制、
      协作式的约定试探, 不是安全/权限边界. 默认实现 = 进程级文件锁, **不防止单个文件的
      准入约束** (越轨者直接改文件防不住, 那是封闭存储系统的职责).
    - 写操作 ``acommit`` / ``afork`` / ``anote`` 声明持有协程锁, 且**互锁**
      (同一时刻一条 branch 只有一个写者).
    """

    @property
    @abstractmethod
    def ref(self) -> BranchRef:
        """当前指针 (name / description / branch_id / created). 构造期即可得."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """branch 独立目录路径."""

    @abstractmethod
    def meta(self) -> BranchMeta:
        """branch 元信息 (metatype / metadata / fork_from). 构造期即可得 (root_path 足够),
        不做异步."""

    @abstractmethod
    def commits(self) -> list[CommitRef]:
        """正序 commits.jsonl (缓存快路径). 同步缓存见 ``acommits``(IO-costly)."""

    @abstractmethod
    async def acommits(self) -> list[CommitRef]:
        """IO-costly 性能开销: 用 aiofiles 从磁盘重读 commits.jsonl."""

    @abstractmethod
    def notes(self) -> dict[str, Note]:
        """commit_id -> Note, 后写覆盖前写 (缓存快路径)."""

    @abstractmethod
    async def anotes(self) -> dict[str, Note]:
        """IO-costly 性能开销: 重读 commit_notes.jsonl."""

    @abstractmethod
    def commit(
        self,
        *,
        message: str = "",
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        """打一个 commit 锚点到 commits.jsonl (append-only). 纯锚点, 不带 message.

        便捷糖: ``message`` 非空时, 内部再种子一条 Note (message 仍只住 Note, 单一真值).
        默认实现 O_APPEND 原子 append, 单写者走此快路径. message 可后补 (``note``).
        """

    @abstractmethod
    async def acommit(
        self,
        *,
        message: str = "",
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        """协程锁写操作, 与 ``afork`` 互锁: 持锁下执行 commit."""

    @abstractmethod
    def note(self, commit_id: str, message: str) -> Note:
        """step 2 (旁路, 可迟): 为已有 commit_id 写唯一摘要 message, last-wins.

        由后台/生产者在关键路径外、稍后调用; 篇幅由生产者保障 (准入). title = 首行派生.
        """

    @abstractmethod
    async def anote(self, commit_id: str, message: str) -> Note:
        """协程锁写操作, 与 ``afork`` 互锁: 持锁下执行 note."""

    @abstractmethod
    def view(self, *, n: int = 10) -> BranchView:
        """最近 n 条 detail + 更早折叠摘要; 含 fork 父支 recap (缓存快路径)."""

    @abstractmethod
    async def aview(self, *, n: int = 10) -> BranchView:
        """IO-costly 性能开销: 聚合多文件构建 view, 最贵."""

    @abstractmethod
    def query_commits(
        self,
        *,
        from_date: datetime | None = None,
        until_date: datetime | None = None,
    ) -> list[CommitRef]:
        """在这条 branch 自己的 commits.jsonl 里, 按 created 时间范围过滤. 不做全量搜索."""

    @abstractmethod
    async def aquery_commits(
        self,
        *,
        from_date: datetime | None = None,
        until_date: datetime | None = None,
    ) -> list[CommitRef]:
        """IO-costly 性能开销: 同上, 但重读磁盘."""

    @abstractmethod
    def fork(self, name: str, description: str = "") -> "Branch":
        """fork 出一个新 branch (引用父支, 非复制). 新支 commits.jsonl 从空开始,
        fork_from 指回父 commit; 读侧拼父支 + 子支双文件成连续轨迹. 单写者同步快路径."""

    @abstractmethod
    async def afork(self, name: str, description: str = "") -> "Branch":
        """协程锁写操作, 与 ``acommit`` 互锁: 建目录 + meta + ref, 多文件写,
        持锁执行."""

    @abstractmethod
    async def __aenter__(self) -> "Branch":
        """写权限门控: 获取 branch 写权限或快速失败. 非强制协作约定, 默认进程级文件锁."""

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """释放写权限."""


class Memento(ABC):
    """owner 级入口. 一个 owner = 一个目录.

    owner/ 下: ``branches.jsonl`` (历史 branch 名单), ``{branch_name}.ref.json`` (当前指针),
    ``branches/{branch_id}/`` (各 branch 目录).
    """

    @property
    @abstractmethod
    def root(self) -> Path:
        """memento 根目录 (只绑定本地 path, 与存储后端不强相关)."""

    @abstractmethod
    def create_branch(
        self,
        name: str,
        description: str = "",
        *,
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Branch:
        """创建新 branch. name 必须不存在, 否则 NameError."""

    @abstractmethod
    def get_branch(self, name: str) -> Branch | None:
        """按 name 取 branch; 不存在返回 None."""

    @abstractmethod
    def list_branches(self) -> list[BranchRef]:
        """活跃 branch 列表 (ref.json glob)."""

    @abstractmethod
    def delete_branch(self, name: str) -> None:
        """删除 name 指针 (移除 ref.json); branch 目录与 commits 保留."""
