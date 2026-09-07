from abc import ABC, abstractmethod
from typing import ClassVar, Iterable, Any, List, Literal
from typing_extensions import Self
from pydantic import BaseModel, Field, AwareDatetime, ValidationError
from datetime import datetime, timezone
from pathlib import Path
import ulid
import asyncio


def unique_id() -> str:
    return str(ulid.ULID())


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CommitRef(BaseModel):
    """指向一个 commit 对象的指针. 属于不可修改数据."""
    id: str = Field(
        default_factory=unique_id,
        description="指向真实的 commit 的 id. 从 id 推导存储位置. "
    )
    branch_id: str = Field(
        description="branch id",
    )
    description: str = Field(
        default='',
        description="创建指针本身的简介.",
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="ref 创建的时间. "
    )


class MomentRecord(BaseModel):
    """append only 可存储的数据记录."""
    id: str = Field(
        default_factory=unique_id,
        description="moment 的 id",
    )
    content: str = Field(
        description="纯文本化的 moment 描述."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="moment 的元数据, 用来记录. ",
    )


class CommitInfo(BaseModel):
    """ commit 的详细信息. 存储在 commit 目录内. 可变更."""
    title: str = Field(
        description="commit 的标题",
    )
    body: str = Field(
        description="commit 的详细描述"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="通过 kv 方式定义的扩展讯息"
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )
    updated: AwareDatetime = Field(
        default_factory=_now_utc,
        description="修改的时间."
    )


class Fork(BaseModel):
    from_branch_id: str = Field(
        description="branch id",
    )
    from_commit_id: str = Field(
        description="commit id",
    )
    fork_branch_id: str = Field(
        description="the fork id"
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )


class Confluence(BaseModel):
    branch_id: str = Field(
        description="branch id",
    )
    commit_id: str = Field(
        description="segment id",
    )
    content: str = Field(
        description="content",
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )


class CommitView(BaseModel):
    """单个 commit 的可读视图. """
    ref: CommitRef = Field(
        description="不可变的信息"
    )
    info: CommitInfo = Field(
        description="当前的详细信息"
    )
    path: str = Field(
        description="存储的绝对路径",
    )
    moments: list[MomentRecord] = Field(
        default_factory=list,
        description="所有存储的数据"
    )
    forks: list[Fork] = Field(
        default_factory=list,
        description="发生过的 forks"
    )
    confluences: list[Confluence] = Field(
        default_factory=list,
        description="发生过的 confluences"
    )
    errors: list[str] = Field(
        default_factory=list,
    )


class BranchRef(BaseModel):
    branch_id: str = Field(
        description="branch id",
    )
    fork_from: Fork | None = Field(
        description="fork from",
    )
    name: str = Field(
        description="branch name",
    )
    description: str = Field(
        description="branch description",
    )
    owner: str = Field(
        description="branch owner",
    )


class Segment(BaseModel):
    id: str = Field(
        default_factory=unique_id,
    )
    start_commit_id: str = Field(
        description="start commit id",
    )
    end_commit_id: str = Field(
        description="segment end commit id",
    )
    summary: str = Field(
        default='',
        description="segment summary",
    )


class BranchInfo(BaseModel):
    """
    Owner 工作区下的分支指针, 每个指针指向一个真实的 branch 工作区, 以 branch id 指向.
    一个 Owner 可以同时拥有很多个 name 命名的分支, 都会重定向到具体的 branch 空间.
    """
    context: str = Field(
        default="",
        description="创建时的上下文"
    )
    status: str = Field(
        default="",
        description="当前状态的整体摘要"
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )
    updated: AwareDatetime = Field(
        default_factory=_now_utc,
        description="修改的时间."
    )


class PreviousCommit(BaseModel):
    ref: CommitRef = Field(
        description=""
    )
    info: CommitInfo = Field(
        description=""
    )


class ForkFromBranch(BaseModel):
    ref: BranchRef = Field(
        description=""
    )
    info: BranchInfo = Field(
        description=""
    )
    commit_ref: CommitRef = Field(
        description=""
    )
    commit_info: CommitInfo = Field(
        description=""
    )


class BranchView(BaseModel):
    ref: BranchRef = Field(
        description="branch id",
    )
    info: BranchInfo = Field(
        description="branch info",
    )
    path: str = Field(
        description="branch absolute path",
    )
    commit: CommitRef | None = Field(
        description="是否最新的数据来自某个 commit"
    )
    moments: list[MomentRecord] = Field(
        default_factory=list,
        description="当前 commit 的 branch moments",
    )
    forks: list[Fork] = Field(
        default_factory=list,
        description="当前未归档的 forks"
    )
    confluences: list[Confluence] = Field(
        default_factory=list,
        description="当前未归档的 confluences"
    )
    previous_commits: list[PreviousCommit] = Field(
        default_factory=list,
        description="未归档的 commits",
    )
    previous_segments: list[Segment] = Field(
        default_factory=list,
        description="未归档的 segments"
    )
    fork_from: ForkFromBranch = Field(
        description="fork from branch",
    )


class Commit(ABC):
    """一个提交节点的独立存储空间. """

    @property
    @abstractmethod
    def ref(self) -> CommitRef:
        """commit 的指针信息, 不可变信息"""
        ...

    @property
    @abstractmethod
    def info(self) -> CommitInfo:
        """commit 的详细数据, 可变信息"""
        ...

    @property
    @abstractmethod
    def path(self) -> Path:
        """commit 的存储区域. """
        ...

    async def view(self) -> CommitView:
        """合并出来的完整视图. 每次调用都会重新获取. """
        moments, forks, confluences = await asyncio.gather(
            self.moments(),
            self.forks(),
            self.confluences(),
            return_exceptions=True,
        )

        view = CommitView(
            ref=self.ref,
            info=self.info,
            path=str(self.path.absolute()),
            moments=moments,
            forks=forks,
            confluences=confluences,
        )
        if not isinstance(moments, Exception):
            view.moments = moments
        else:
            view.errors.append(str(moments))
        if not isinstance(forks, Exception):
            view.forks = forks
        else:
            view.errors.append(str(forks))
        if not isinstance(confluences, Exception):
            view.confluences = confluences
        else:
            view.errors.append(str(confluences))
        return view

    @abstractmethod
    async def forks(self) -> list[Fork]:
        """在这个 commit 周期里发生过的 forks"""
        ...

    @abstractmethod
    async def confluences(self) -> list[Confluence]:
        """在这个 commit 周期发生过的 confluences."""
        ...

    @abstractmethod
    async def update(self, info: CommitInfo) -> None:
        """更新 commit 的数据."""
        ...

    @abstractmethod
    async def moments(self) -> list[MomentRecord]:
        """返回 moments 记录. 对于 commit 而言, moment 记录是不可变的. """
        ...


class Branch(ABC):
    """
    一个工作区的分支.
    """

    @property
    @abstractmethod
    def ref(self) -> BranchRef:
        ...

    @property
    @abstractmethod
    def info(self) -> BranchInfo:
        """branch 的描述信息."""
        ...

    @property
    @abstractmethod
    def path(self) -> Path:
        """branch 的独立工作区"""
        ...

    @abstractmethod
    async def moments(self) -> list[MomentRecord]:
        """当前未被压缩的 moments"""
        ...

    @abstractmethod
    async def commits(self) -> List[CommitRef]:
        """已经生产的 commits"""
        ...

    @abstractmethod
    async def segments(self) -> list[Segment]:
        """已经生产的 segments"""
        ...

    @abstractmethod
    async def forks(self) -> list[Fork]:
        """未被归档到 segments 的 forks"""
        ...

    @abstractmethod
    async def confluences(self) -> list[Confluence]:
        """还未归档到 segment 里的 confluences."""
        ...

    @abstractmethod
    async def append(self, moment: MomentRecord) -> None:
        """添加 moment, append only"""
        ...

    @abstractmethod
    async def commit(self, description: str, *, info: CommitInfo | None = None) -> Commit:
        """基于 description 创建一个 commit """
        # 0. 生成新的 ref
        # 1. 将当前的 moments 迁移到目标 commit 目录. ref 也写入目标目录.
        # 2. ref append 到当前 commits 记录中.
        # 3. commit info 可以事后更新.
        ...

    @abstractmethod
    async def compact(
            self,
            segment_summary: str,
    ) -> Segment:
        """基于最后一个 commit, 生成一个新的 segment 到记录中. """
        ...

    @abstractmethod
    async def fork(
            self,
            name: str,
            *,
            description: str = '',
            context: str = '',
            commit_id: str | None = None,
            force: bool = False,
    ) -> 'Branch':
        """ checkout 一个新的分支."""
        ...

    @abstractmethod
    async def confluent(
            self,
            content: str,
            *,
            description: str = '',
            last_commit_id: str | None = None,
    ) -> Confluence:
        """从当前最后一个 commit 向 fork 的上游回流. """
        ...

    @abstractmethod
    async def view(
            self,
            *,
            backtrack: int = 1,
    ) -> BranchView:
        """
        :param backtrack: 回溯的 branch 信息.
        """
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """锁定生命周期, 可以写, 否则只读"""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出锁."""
        ...


class Repository(ABC):

    @property
    @abstractmethod
    def owner(self) -> str:
        ...

    @property
    @abstractmethod
    def root(self) -> Path:
        ...

    @abstractmethod
    async def heads(self) -> list[BranchRef]:
        """获取所有的活跃的 branch 指针"""
        ...

    @abstractmethod
    async def branches(self) -> list[BranchRef]:
        """所有存储过的 branch"""
        ...

    @abstractmethod
    async def main(self, *, read_only: bool = False) -> Branch:
        """get or create main branch"""
        ...

    @abstractmethod
    async def checkout(
            self,
            ref: str | BranchRef,
            *,
            read_only: bool = False,
    ) -> Branch:
        """进入一个 branch by name or ref"""
        ...

    @abstractmethod
    async def create(
            self,
            name: str,
            description: str = '',
            *,
            fork_from: BranchRef | None = None,
    ) -> Branch:
        """创建一个新的 branch. """
        ...

    @abstractmethod
    async def delete(self, branch_name: str) -> None:
        """删除某个 branch 在工作区的 ref, 实际上不会删除 branch 自己的存储空间. """
        ...

    @abstractmethod
    async def fetch_commit(self, commit_id: str) -> Commit | None:
        """获取一个 commit"""
        ...

    @abstractmethod
    async def list_commits(
            self,
            *,
            from_date: datetime | None = None,
            until_date: datetime | None = None,
            branch_id: str | None = None,
            descending: bool = False,
            limit: int = -1,
    ) -> list[CommitRef]:
        """获取指定时间范围的 commits. """
        ...


class Memento(ABC):
    @property
    @abstractmethod
    def root(self) -> Path:
        ...

    @abstractmethod
    def repository(self, owner: str) -> Repository:
        ...
