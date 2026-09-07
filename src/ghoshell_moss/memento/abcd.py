import shutil
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Callable

from typing_extensions import Self
from pydantic import BaseModel, Field, AwareDatetime
from datetime import datetime, timezone
from pathlib import Path
import ulid
import asyncio

__all__ = [
    'Memento', 'MomentRecord',
    'Repository',
    'Branch', 'BranchId', 'BranchInfo', 'BranchMeta', 'BranchRef', 'BranchView',
    'Commit', 'CommitId', 'CommitInfo', 'CommitMeta', 'CommitRef', 'CommitView',
    'Segment', 'SegmentView',
    'Recap'
]


def _unique_id() -> str:
    return str(ulid.ULID())


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


BranchId = str
CommitId = str


class CommitRef(BaseModel):
    """
    指向一个 commit 对象的指针. 属于不可修改数据. 通过 created + id 指向唯一存储路径.
    默认存储位置在 [owner]/branches/{branch_id}/commits.jsonl
    """
    id: CommitId = Field(
        default_factory=_unique_id,
        description="指向真实的 commit 的 id. 从 id 推导存储位置. "
    )
    description: str = Field(
        default='',
        description="创建指针本身的简介.",
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="ref 创建的时间. "
    )

    def commit_dir(self, root: Path) -> Path:
        if not root.is_dir():
            raise NotADirectoryError(f'{root} is not a directory')
        yyyy = self.created.strftime("%Y")
        mm = self.created.strftime("%m")
        return root / 'commits' / yyyy / mm / f"cmt_{self.id}"

    def moments_file(self, root: Path) -> Path:
        return self.commit_dir(root) / 'moments.jsonl'

    def forks_file(self, root: Path) -> Path:
        return self.commit_dir(root) / 'forks.jsonl'

    def imports_file(self, root: Path) -> Path:
        return self.commit_dir(root) / 'imports.jsonl'

    def meta_file(self, root: Path) -> Path:
        return self.commit_dir(root) / 'meta.json'

    def info_file(self, root: Path) -> Path:
        return self.commit_dir(root) / 'info.json'

    def segment_content_file(self, root: Path) -> Path:
        """由外部的 segment 写入的摘要文件. """
        return self.commit_dir(root) / 'segment.md'


class MomentRecord(BaseModel):
    """
    append only 可存储的数据记录.
    staging 阶段为 branch 持有, 默认存醋在 [owner]/branches/{branch_id}/moments.jsonl
    committed 阶段为 commit 持有, 默认存储在 [owner]/commits/[yyyy]/[mm]/{commit_id}/moments.jsonl
    append only 数据.
    """
    id: str = Field(
        default_factory=_unique_id,
        description="moment 的 id",
    )
    content: str = Field(
        description="纯文本化的 moment 描述."
    )
    metatype: str = Field(
        default="",
        description="可选字段, 记录生产 moment record 的类型",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="moment 的元数据, 用来记录. ",
    )


class Recap(BaseModel):
    id: str = Field(
        description="recap id",
    )
    kind: Literal['branch', 'segment', 'commit'] = Field(
        description="recap kind",
    )
    content: str = Field(
        description="recap content in nature language",
    )

    def to_xml(self) -> str:
        return (f"<recap kind={self.kind} from={self.id}>\n"
                f"{self.content}"
                "\n</recap>"
                )


class Confluence(BaseModel):
    """
    记录分支之间创建, 汇流的关键点. 分为 Fork 与 Merge 两种情况.
    当一个 A 分支, fork 出一个新分支 B 时, A 会存储在:
    - staging 阶段: [owner]/branches/{branch_id}/forks.jsonl
    - commited 阶段: [owner]/commits/yyyy/mm/{commit_id}/forks.jsonl
    B 则存储在 CommitMeta 里.

    当一个 B 分支, 将自己的一个节点回调给主分支或其它分支 A 时, 只记录在 A:
    - staging 阶段记录在: [owner]/branches/{branch_id}/imports.jsonl
    - commited 阶段, 记录在: [owner]/commits/yyyy/mm/cmt_[commit_id]/imports.jsonl
    """
    id: str = Field(
        default_factory=_unique_id,
        description="自身的唯一 id",
    )
    from_branch_id: BranchId = Field(
        description="the branch id fork from",
    )
    from_commit_id: CommitId = Field(
        description="commit id fork from",
    )
    to_branch_id: BranchId = Field(
        description="the new branch id"
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )

    def to_xml(self) -> str:
        return ("<confluence>\n"
                f"from_branch: {self.from_branch_id}\n"
                f"from_commit: {self.from_commit_id}\n"
                "</confluence>")


class CommitMeta(BaseModel):
    """
    Commit 不可变更的元数据.
    默认存储在 [owner]/commits/[yyyy]/[mm]/cmt_{commit_id}/meta.json, 可以 glob 查找.
    """
    ref: CommitRef = Field(
        description=""
    )
    branch_id: BranchId = Field(
        description="branch id, 用来反查 commit 生产时所属的 branch. ",
    )
    metatype: str = Field(
        default="",
        description="可选字段, 记录生产 commit 的数据类型",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="通过 kv 方式定义的扩展讯息"
    )


class CommitInfo(BaseModel):
    """
    commit 的详细信息, 可变更.
    默认存储在 [owner]/commits/yyyy/mm/cmt_{commit_id}/commit.json, 可以 glob 查找.
    """
    title: str = Field(
        default='',
        description="commit 的标题",
    )
    body: str = Field(
        default="",
        description="commit 的详细描述"
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )
    updated: AwareDatetime = Field(
        default_factory=_now_utc,
        description="修改的时间."
    )

    @property
    def content(self) -> str:
        lines = []
        if self.title:
            lines.append(self.title)
        if self.body:
            lines.append(self.body)
        return "\n\n".join(lines)


class CommitView(BaseModel):
    """单个 commit 的可读视图. """
    meta: CommitMeta = Field(
        description="commit 不可变的创建信息"
    )
    info: CommitInfo = Field(
        description="当前的详细信息"
    )
    path: str = Field(
        description="存储的绝对路径, commit 目录可以存储额外的数据文件. ",
    )
    moments: list[MomentRecord] = Field(
        default_factory=list,
        description="所有存储的数据"
    )
    forks: list[Confluence] = Field(
        default_factory=list,
        description="发生过的 forks"
    )
    imports: list[Confluence] = Field(
        default_factory=list,
        description="得到过的 confluences"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="生成 view 时发现并忽略的异常. ",
    )


class BranchRef(BaseModel):
    """
    branch 的指针数据. 由 owner 持有指向活跃的 branch 和创建过的 branches.

    创建过的 branches 默认存储在 [owner]/branches.jsonl . name 可能过期.
    活跃的 branch ref 存醋在 [owner]/{branch_name}.head.json, 禁止重名.
    """
    id: BranchId = Field(
        description="branch id",
    )
    description: str = Field(
        default='',
        description="branch description when created",
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )

    def branch_dir(self, root: Path) -> Path:
        return root / 'branches' / f'br_{self.id}'

    def staging_forks_file(self, root: Path) -> Path:
        return self.branch_dir(root) / 'forks.jsonl'

    def staging_moments_file(self, root: Path) -> Path:
        return self.branch_dir(root) / 'moments.jsonl'

    def staging_imports_file(self, root: Path) -> Path:
        return self.branch_dir(root) / 'imports.jsonl'

    def commits_file(self, root: Path) -> Path:
        return self.branch_dir(root) / 'commits.jsonl'

    def segments_file(self, root: Path) -> Path:
        return self.branch_dir(root) / 'segments.jsonl'

    def new_commit(
            self,
            root: Path,
            description: str,
            *,
            metatype: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        """
        基于文件系统的核心示例. 在实际的使用中, 应该基于 branch 实现, 线性写入.
        """
        ref = CommitRef(
            description=description,
        )
        meta = CommitMeta(
            ref=ref,
            metadata=metadata or {},
            metatype=metatype or '',
            branch_id=self.id,
        )
        info = CommitInfo()
        ref.commit_dir(root).mkdir(parents=True, exist_ok=True)
        ref.meta_file(root).write_text(meta.model_dump_json())
        ref.info_file(root).write_text(info.model_dump_json())
        unlink = []
        if file := self.staging_imports_file(root):
            if file.exists():
                unlink.append(file)
                shutil.copy(file, ref.imports_file(root))
        if file := self.staging_forks_file(root):
            if file.exists():
                unlink.append(file)
                shutil.copy(file, ref.forks_file(root))
        if file := self.staging_moments_file(root):
            if file.exists():
                unlink.append(file)
                shutil.copy(file, ref.moments_file(root))
        if len(unlink) > 0:
            for file in unlink:
                file.unlink()
        with self.commits_file(root).open('a', encoding='utf-8') as f:
            f.write(ref.model_dump_json(indent=0, ensure_ascii=False) + '\n')
        return ref


class BranchMeta(BaseModel):
    """
    branch 创建时生产的元信息, 不可变.
    存储在 [owner]/branches/{branch_id}/meta.json
    """
    ref: BranchRef = Field(
        description="branch ref",
    )
    name: str = Field(
        description="branch name when created",
    )
    owner: str = Field(
        description="branch owner",
    )
    metatype: str = Field(
        default="",
        description="记录 branch 生产时的数据类型.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="可以扩展的元数据字段.",
    )
    fork_from: Confluence | None = Field(
        default=None,
        description="fork from branch",
    )
    context: str = Field(
        default="",
        description="创建时的上下文"
    )
    lineage: list[BranchId] = Field(
        default_factory=list,
        description="自身的祖先 branch 节点. 会逐层继承, 线性增长."
    )


class Segment(BaseModel):
    """
    在一个 Branch 的生命周期中, 可能产出过 N 个 commit, 周期性元数据变更或上下文重组时, 可以生产 segment.
    举例使用 agent 作为 branch 的载体, 运行中周期性切换 agent session id, 则每个 session 可以作为一个 segment.
    它包含多个 commit, 完成一个可提示摘要.

    存储在 [owner]/branches/{branch_id}/segments.jsonl 中, 可视作对多个 commit 的压缩.
    """
    id: str = Field(
        default_factory=_unique_id,
        description="segment id",
    )
    start_commit_id: str = Field(
        description="segment start_commit_id",
    )
    end_commit_id: str = Field(
        description="segment end_commit_id",
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="segment created time",
    )
    metatype: str = Field(
        default="",
        description="记录 segment 被生产的类型. ",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="segment metadata",
    )


class BranchInfo(BaseModel):
    """
    一个 Branch 分支的可变讯息.
    存储在 [owner]/branches/{branch_id}/branch.json
    """
    title: str = Field(
        default='',
        description="branch title, 正式有语义的内容描述",
    )
    status: str = Field(
        default="",
        description="当前运行状态的讯息. 重写方便读取 Branch 状态."
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="创建的时间. "
    )
    updated: AwareDatetime = Field(
        default_factory=_now_utc,
        description="修改的时间."
    )


class Staging(BaseModel):
    """Branch 里 staging 阶段的数据."""

    moments: list[MomentRecord] = Field(
        default_factory=list,
        description="当前 staging 的 moments",
    )
    forks: list[Confluence] = Field(
        default_factory=list,
        description="当前 staging 的 forks"
    )
    imports: list[Confluence] = Field(
        default_factory=list,
        description="当前未归档的 imports"
    )


class Previous(BaseModel):
    branches: list[Recap] = Field(
        default_factory=list,
        description="previous branches",
    )
    segments: list[Recap] = Field(
        default_factory=list,
        description="previous segments",
    )
    commits: list[Recap] = Field(
        default_factory=list,
        description="previous commits",
    )


class SegmentView(BaseModel):
    """
    segment 的视图.
    """
    branch_ref: BranchRef = Field(
        description="branch ref",
    )
    segment: Segment = Field(
        description="segment data",
    )
    content: str = Field(
        description="segment content",
    )
    commits: list[CommitRef] = Field(
        default_factory=list,
        description="commits recaps",
    )


class BranchView(BaseModel):
    """
    Branch 的视图, 被构建出来的结构化数据.
    """
    meta: BranchMeta = Field(
        description="branch 的元信息",
    )
    info: BranchInfo = Field(
        description="branch info",
    )
    path: str = Field(
        description="branch absolute path",
    )
    commit_id: CommitId | None = Field(
        default="",
        description="是否最新的数据来自某个 commit, 否则来自当前的 staging 区. "
    )
    previous: Previous = Field(
        default_factory=Previous,
        description="branch 的前置讯息. ",
    )
    staging: Staging = Field(
        default_factory=Staging,
        description="branch 的当前讯息. "
    )


class Commit(ABC):
    """
    一个提交节点的独立存储空间.
    本质上是一个路径的指针.
    """

    @property
    @abstractmethod
    def path(self) -> Path:
        """commit 的存储区域. """
        ...

    @abstractmethod
    async def meta(self) -> CommitMeta:
        """
        commit 的元信息, 属于不可变信息
        默认存储在 [commit_dir]/meta.json
        """
        ...

    @abstractmethod
    async def info(self) -> CommitInfo:
        """
        commit 的详细数据, 可变信息
        默认存储在 [commit_dir]/commit.json
        """
        ...

    @abstractmethod
    async def forks(self, *, on_error: Callable[[Exception], None] | None = None) -> list[Confluence]:
        """
        在这个 commit 周期里发生过的 forks
        默认存储在 [commit_dir]/forks.jsonl
        首次存储才创建文件. 
        """
        ...

    @abstractmethod
    async def imports(self, *, on_error: Callable[[Exception], None] | None = None) -> list[Confluence]:
        """
        在这个 commit 周期发生过的 imports.
        首次存储才创建文件. 
        """
        ...

    @abstractmethod
    async def moments(self, *, on_error: Callable[[Exception], None] | None = None) -> list[MomentRecord]:
        """
        返回 moments 记录. 对于 commit 而言, moment 记录是不可变的. 
        """
        ...

    async def view(self, *, on_error: Callable[[Exception], None] | None = None) -> CommitView:
        """合并出来的完整视图. 每次调用都会重新获取. 此处是示例. """

        meta, info, moments, forks, imports = await asyncio.gather(
            self.meta(),
            self.info(),
            self.moments(on_error=on_error),
            self.forks(on_error=on_error),
            self.imports(on_error=on_error),
            # 内部数据结构转换异常应该跳过, 通过回调通知.
            return_exceptions=False,
        )

        for data in [meta, info]:
            if isinstance(data, Exception):
                raise ValueError(f"Invalid commit data. meta: {meta}; info: {info}")

        view = CommitView(
            meta=meta,
            info=info,
            path=str(self.path.absolute()),
        )
        view.forks = forks
        view.imports = imports
        view.moments = moments
        return view

    @abstractmethod
    async def update(self, info: CommitInfo) -> None:
        """更新 commit 的数据."""
        ...

    @property
    @abstractmethod
    def read_only(self) -> bool:
        """是否是只读状态. """
        ...

    @property
    @abstractmethod
    def writable(self) -> bool:
        """是否可以写. 当 readonly = False, 同时没有进入 async with statement 时, writable 也是 False. """
        ...

    async def recap(self) -> Recap:
        meta, info = await asyncio.gather(self.meta(), self.info())
        return Recap(
            id=meta.ref.id,
            content=info.content,
            kind='commit',
        )

    @abstractmethod
    async def __aenter__(self) -> Self:
        """开启锁, 获取写权限. 如果一开始就是 readonly 的, 则无权限获取锁. """
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """关闭锁. """
        ...


class Branch(ABC):
    """
    工作区里的分支, 记录了可追溯的 moment 轨迹, 并且可以在 commit / branch 目录里存放各种 memento (facts, anchors...).
    是一种面向模型的数据记录方式. 
    
    它考虑到实时的上下文压缩和切换, 约定支持旁路的 commit 技术: 
    1) 较快地生成 commit. 
    2) 旁路更新 commit info. 
    来生成分段的摘要. 
    
    若干个 commit 可以被压缩成一个 segment,

    写权限的 branch 实例在读动作后, 会镜像缓存到内存中.
    """

    @property
    @abstractmethod
    def path(self) -> Path:
        """branch 的独立工作区"""
        ...

    @abstractmethod
    async def fork_from(self) -> 'Branch | None':
        """获取父 branch 的信息. """
        ...

    @abstractmethod
    async def meta(self) -> BranchMeta:
        """获取 branch 的 meta 数据. """
        ...

    @abstractmethod
    async def ref(self) -> BranchRef:
        return (await self.meta()).ref

    @abstractmethod
    async def info(self) -> BranchInfo:
        """读取 branch 的可变信息."""
        ...

    @abstractmethod
    async def moments(self) -> list[MomentRecord]:
        """
        当前未被压缩的 moments
        """
        ...

    @abstractmethod
    async def add_moment(self, moment: MomentRecord) -> None:
        """添加 moment, append only"""
        ...

    @abstractmethod
    async def commits(self) -> List[CommitRef]:
        """
        已经生产的 commits
        默认存储在 [branch]/commits.jsonl, append only
        """
        ...

    @abstractmethod
    async def segments(self, *, on_error: Callable[[Exception], None] | None = None) -> list[Segment]:
        """
        已经生产的 segments
        默认存储在 [branch]/segments.jsonl, append only
        """
        ...

    @abstractmethod
    async def forks(self, *, on_error: Callable[[Exception], None] | None = None) -> list[Confluence]:
        """未被归档到 segments 的 forks"""
        ...

    @abstractmethod
    async def imports(self, *, on_error: Callable[[Exception], None] | None = None) -> list[Confluence]:
        """未被归档的 imports"""
        ...

    @abstractmethod
    async def update(
            self,
            title: str,
            status: str,
    ) -> BranchInfo:
        """更新 branch 讯息. """
        ...

    @abstractmethod
    async def commit(
            self,
            description: str,
            *,
            metatype: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> Commit:
        """基于 description 创建一个 commit."""
        # 0. 生成新的 ref
        # 1. 将当前的 moments 迁移到目标 commit 目录. ref 也写入目标目录.
        # 2. ref append 到当前 commits 记录中.
        # 3. commit info 可以事后更新.
        ...

    @abstractmethod
    async def get_commit(self, commit_id: CommitId) -> Commit | None:
        """
        获取当前 branch id 创建过的 commit. 不存在则抛出文件不存在.
        当前 branch id 的 commit 是允许写的, 否则是只读的.
        """
        ...

    @abstractmethod
    async def import_from(self, branch_id: str, *, from_commit_id: str | None = None) -> Confluence:
        """
        尝试获取一个 commit. 可添加到当前上下文中.
        """
        ...

    @abstractmethod
    async def add_import(self, confluence: Confluence) -> None:
        """添加一个合法的 import 讯息. """
        ...

    @abstractmethod
    async def export_to(self, branch_id: str | None = None, *, commit_id: str | None = None) -> Confluence:
        """尝试导出一个 commit 信息给目标 branch. 为空的话, 则是自己 fork from 的 branch. 否则会抛出异常. """
        ...

    async def view_commit(
            self,
            commit_id: CommitId,
            *,
            on_error: Callable[[Exception], None] | None = None,
    ) -> CommitView:
        commit = await self.get_commit(commit_id)
        if commit is not None:
            return await commit.view(on_error=on_error)
        raise FileNotFoundError(f"Commit {commit_id} not found")

    @abstractmethod
    async def fork(
            self,
            name: str,
            description: str,
            context: str,
            *,
            commit_id: str | None = None,
            metatype: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> 'Branch':
        """
        从当前上下文中 fork 一个新的 branch
        1. 目标 branch name 必须不存在. 否则不生效.
        2. 当前 branch 持有写权限.
        """
        ...

    @abstractmethod
    async def slice(
            self,
            end_commit_id: str | None = None,
            content: str | None = None,
            *,
            metatype: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> Segment:
        """
        基于指定或最后一个 commit, 生成一个新的 segment
        content 可以后补.
        """
        ...

    @abstractmethod
    async def update_segment(
            self,
            seg_id: str,
            content: str,
    ) -> Path:
        """
        在 segment 的 end commit 里写入 content, 会记录到对应 commit 的 segment file 里.
        """
        ...

    @abstractmethod
    async def view_segment(self, seg_id: str, *, on_error: Callable[[Exception], None] | None = None) -> SegmentView:
        """获取当前 branch 下指定 segment 的讯息."""
        ...

    @abstractmethod
    async def view(
            self,
            *,
            on_error: Callable[[Exception], None] | None = None,
    ) -> BranchView:
        """
        :param on_error: 注册错误回调的函数.
        """
        ...

    @abstractmethod
    async def recap_commits(self, *commit_ids: str, on_error: Callable[[Exception], None] | None = None) -> list[Recap]:
        """
        根据 commit id, 顺序生成 commit recap.
        commit_id 为空时, recap 全部 commit.
        """
        ...

    @abstractmethod
    async def recap_segments(
            self,
            *segment_ids: str,
            on_error: Callable[[Exception], None] | None = None,
    ) -> list[Recap]:
        """
        根据 segment_id, 顺序生成 segment_recap.
        segment_id 为空时, recap 全部 segment.
        """
        ...

    @abstractmethod
    async def recap_linage(
            self,
            *branch_ids: str,
            on_error: Callable[[Exception], None] | None = None,
    ) -> list[Recap]:
        """
        生成历史 branch 的 recap.
        """
        ...

    @property
    @abstractmethod
    def read_only(self) -> bool:
        """是否是只读状态. """
        ...

    @property
    @abstractmethod
    def writable(self) -> bool:
        """是否可以写. 当 readonly = False, 同时没有进入 async with statement 时, writable 也是 False. """
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """锁定生命周期 (文件系统通常是 进程+文件锁), 可以写, 否则只读. 如果一开始就没有权限, 则此处会抛出异常. """
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出锁."""
        ...


class Repository(ABC):
    """
    隶属于单个 owner 的 branch 空间.
    """

    @property
    @abstractmethod
    def owner(self) -> str:
        """当前 repository 的所有者. """
        ...

    @property
    @abstractmethod
    def root(self) -> Path:
        """当前 repository 的根目录. """
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
            name: str,
            *,
            read_only: bool = False,
    ) -> Branch:
        """进入一个 branch by name"""
        ...

    @abstractmethod
    async def get(self, branch_id: str) -> Branch | None:
        """尝试获取一个 branch. """
        ...

    @abstractmethod
    async def create(
            self,
            name: str,
            description: str = '',
            *,
            fork_from: BranchRef | None = None,
    ) -> Branch:
        """
        创建一个新的 branch. 允许 fork. 不是 read only
        :raise NameError: 如果目标 name 存在的话.
        """
        ...

    @abstractmethod
    async def delete(self, branch_name: str) -> None:
        """
        删除某个 branch 在工作区的 ref, 实际上不会删除 branch 自己的存储空间.
        :raise FileNotFoundError: 如果 branch name 不存在.
        """
        ...

    @abstractmethod
    async def fetch_commit(self, commit_id: str) -> Commit | None:
        """获取一个 commit, readonly 一定为 false. """
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
    ) -> list[Commit]:
        """获取指定时间范围的 commits. """
        ...

    @abstractmethod
    async def get_commit(self, commit_id: str, *, read_only: bool = True) -> Commit | None:
        """获取 commit"""
        ...


class Memento(ABC):

    @property
    @abstractmethod
    def root(self) -> Path:
        """memento 根目录的位置. """
        ...

    @abstractmethod
    def repository(self, owner: str) -> Repository:
        """根据 owner 名称, 获取 repository 实例. """
        ...
