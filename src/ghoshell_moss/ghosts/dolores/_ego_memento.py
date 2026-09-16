"""EgoMementoManager — ghost 级的 memento 旁路服务 (由 ghost 持有, 注入 ego).

职责边界 (决策落点):

- **锚点**: ``commit()`` 组 ``DshSessionRef`` + 写 memento. metadata = ``{ref, prev_turn}``,
  memento 只存不解析. 区间取「已完成的 turn」(追认): 签发时 turn/end 已知, usage 齐整.
- **view / 切点**: ``view_message()`` 给 ghost 的 memories; ``resume_ref()`` 给 ego 重建的切点.
- **阈值**: ``evaluate()`` 是纯算法. **窗口状态 (window_base / warned) 在 ego**, 不在这里 ——
  manager 不托管 ego 的运行时状态. 每个 commit 重置滑动窗口.
- **旁路 note 生产**: ``schedule_note()`` 为每个锚点排一个旁路任务 (身份旁路 session, seed = 源
  session 的 verbatim 前缀, 吃满前缀缓存), 取回 plain text 写进 note. prompt 轨迹接续 (前驱坐标
  + 最近 N 条已就绪 message 作前文), 约束显式进载荷 (低思考 + maxTokens 硬 cap). **这些任务的
  生命周期与状态归 manager** (``bypass``): 关停时取消在飞的、不等 —— 缺 note 的 commit 就空着,
  内容仍可 read.
  **不做重启补漏**: 空的就空着.
- 未落: ``read`` (commit 区间 → 文本)、``chat_commit``, 以及 ego 的 inflight 替换 (当前只在 ego
  开启/关闭时用 ``resume_ref()`` 重建).

dep 只有不可变项 (connection / memento / config / logger) —— 这些是它干活的工具, 不是 ego 状态.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.deepseek_harness.trajectory import render_transcript
from ghoshell_moss.deepseek_harness.types.refs import DshSessionRef
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, TokenUsage
from ghoshell_moss.memento.abcd import Branch, BranchView, CommitRef, CommitView, Memento
from ghoshell_moss.message import Message

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshConnection

__all__ = [
    "BranchInfo", "BrokenCommitError", "BypassCommit", "BypassState",
    "CommitDecision", "CommitDigest", "EgoMementoConfig", "EgoMementoManager",
]

# metadata 约定 (memento 只存不解析).
_REF_KEY = "ref"
_PREV_TURN_KEY = "prev_turn"

# 旁路原语路由 (dsh plugin 上开): 收 {ref, prompt}, 源 session 冷 seed 跑一轮回 {message}.
# note / chat 都是它的调用方 —— prompt 语义留在 MOSS 侧, plugin 只做一轮运行.
_BYPASS_RUN_ROUTE = "/moss-api/ghost/dolores/bypass/run"
# read 路由: 收 {ref} 回 {events}, 源 log 的 turn 区间原始切片 (live-or-cold).
_READ_ROUTE = "/moss-api/ghost/dolores/read"

# note 旁路的固定约束 (强制, 不可调): 摘要压低思考模式 —— 压缩不该深想, 省 token 不阻塞主路.
_NOTE_EFFORT = "low"
# 旁路摘要前置上下文的窗口: 取本 commit 之前最近 N 条已就绪的 message 作前文.
_NOTE_PRIOR_COUNT = 2


def _note_prompt(
    *,
    latest_coord: str | None,
    prior: list[tuple[str, str]],
) -> str:
    """note 摘要 prompt —— 轨迹接续 (英文, 面向模型).

    ``latest_coord`` = 本 commit 的前驱坐标 (None = 首条, 覆盖完整上下文); ``prior`` =
    最近已就绪 commit 的 ``(coord, message)``, 老→新, 作前文.
    """
    lines = [
        "You are writing the next commit in your memento trajectory — a line of commits that records",
        "your past. The conversation above is the span this commit covers.",
    ]
    if latest_coord is None:
        lines.append("")
        lines.append("This is the first commit: it covers the whole conversation so far.")
    else:
        lines.append("")
        lines.append(
            f"This commit continues right after {latest_coord} — record only what happened since then."
        )
    if prior:
        lines.append("")
        lines.append("The most recent commits before this one, for continuity:")
        for coord, message in prior:
            lines.append(f"- {coord}: {message}")
    lines.extend([
        "",
        "Write the summary itself and nothing else, in this structure:",
        "1. What happened — continuing from the previous commit.",
        "2. Points worth noting.",
        "3. Resources involved (files, etc.) — names and connections only, no detail.",
        "4. Thoughts or feelings you want to record for your future self.",
        "",
        "Keep it short. Your persistent state (identity, ground) stays continuous — do not re-state",
        "it; record only what changed in this span.",
    ])
    return "\n".join(lines)


# chat 的 prompt 前缀 —— 必须点破"对话对象是上下文而不是 commit 本身".
_CHAT_PREAMBLE = (
    "You are not talking to a single commit next; you are talking to the context that commit "
    "belongs to. The conversation history you see ends where that commit was made, and anything "
    "after it is outside your view. Answer the question below directly, without pleasantries, and "
    "do not pretend you know what came later."
)


class BrokenCommitError(RuntimeError):
    """chat 的目标 commit 的 message 是坏占位 (非真摘要), 不可对话."""


class EgoMementoConfig(BaseModel):
    """``.dolores.yml`` 的 ``memento:`` 段. Field 默认即兜底."""

    branch_name: str = Field(default="main", description="ego 提交的 memento branch (ghost 启动时取其/建).")
    view_limit: int = Field(default=10, description="Branch.view(n=..) — 最近 n 条进 memory view.")
    warn_tokens: int = Field(
        default=50_000,
        description="K: 距上次 commit 的窗口增量 token 达此值, 插**一次** commit 提醒 (每窗口).",
    )
    force_tokens: int = Field(
        default=100_000,
        description="T: 达此值强制建锚点. 置 0 = 每 turn commit (验证期用).",
    )
    note_max_tokens: int = Field(
        default=800,
        description="note 摘要的输出上限 (maxTokens 硬 cap, 旁路单轮请求侧). 判定区间 500-1000, 取 800.",
    )


class BypassState(str, Enum):
    """旁路任务的运行状态 (manager 持有的治理状态; note 本身的真值在 memento, 不在这里)."""

    RUNNING = "running"  # 旁路任务在飞
    READY = "ready"      # note 已落
    FAILED = "failed"    # 空 message / 传输失败 —— 终态留空, 不自动重试 (内容仍可 read)


@dataclasses.dataclass
class BypassCommit:
    """一个被旁路治理的 commit: 切点 ref + 状态 + 在飞任务."""

    commit_id: str
    ref: DshSessionRef
    state: BypassState = BypassState.RUNNING
    task: asyncio.Task | None = None


class BranchInfo(BaseModel):
    """一个 branch 的概览 —— 先看有哪些 branch, 再用 ``branch_view(name)`` 看内容.

    坐标 ``latest_coord`` 是给模型引用的地址 (形如 ``1-27``); 空 branch 没有坐标.
    """

    name: str = Field(description="branch 名, 看 view 时用它。")
    index: int = Field(default=0, description="owner 内 branch 序号 — 坐标前半截。")
    description: str = Field(default="")
    commits_total: int = Field(default=0)
    latest_coord: str = Field(default="", description="最新 commit 的坐标; 空 branch 为空串。")
    latest_title: str = Field(default="", description="最新 commit 的 message 首行。")
    created: datetime | None = Field(default=None)


class CommitDigest(BaseModel):
    """一条 commit 的列表条目 —— ``git log --oneline`` 的 seq + title, 带 body 做详情."""

    coord: str = Field(description="坐标 ``{branch_index}-{seq}``, read/chat 都用它。")
    seq: int = Field(default=0, description="branch 内 commit 序号。")
    title: str = Field(default="", description="message 首行。")
    body: str = Field(default="", description="message 其余部分 (详情)。")
    created: datetime | None = Field(default=None)
    broken: bool = Field(default=False, description="message 是坏占位 (chat 不可用)。")


class CommitDecision(str, Enum):
    """turn/end 一次阈值判定的结果 (纯算法输出, 由 ego 执行)."""

    NONE = "none"
    WARN = "warn"
    FORCE = "force"


def _input_size(usage: TokenUsage) -> int:
    """一次调用的窗口输入大小 = 未缓存 input + cache read (两者都进了 prompt)."""
    return usage.inputTokens + (usage.cacheReadTokens or 0)


class EgoMementoManager:
    """ghost 级的 memento 旁路服务. 见模块 docstring."""

    def __init__(
            self,
            *,
            connection: DshConnection,
            memento: Memento,
            config: EgoMementoConfig | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self._connection = connection
        self._memento = memento
        self._config = config or EgoMementoConfig()
        self._logger = logger or get_moss_logger()
        # 旁路治理状态: 每个排过旁路的 commit → 它的切点 / 状态 / 在飞任务 (见 BypassCommit).
        self._bypass: dict[str, BypassCommit] = {}

    @property
    def config(self) -> EgoMementoConfig:
        return self._config

    # ── 生命周期 (由 ghost 开启/关闭; 旁路任务归它治理) ───────────────

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """关停: 取消在飞的旁路任务, 不等它们收尾.

        旁路 note 是 best-effort —— 缺 note 的 commit 就空着 (内容仍可 read), 为一个 LLM 调用把
        ghost 关闭拖住几秒不值得. 运行期排的旁路早就跑完了, 通常只有退出前刚排的那个会丢.
        """
        running = [run for run in self._bypass.values() if run.task is not None and not run.task.done()]
        for run in running:
            run.task.cancel()
        if running:
            await asyncio.gather(*(run.task for run in running), return_exceptions=True)
            self._logger.info("memento bypass: %d in-flight note run(s) dropped at shutdown", len(running))

    @property
    def bypass(self) -> dict[str, BypassCommit]:
        """旁路治理状态的只读面 (观测/测试); note 本身的真值在 memento, 不在这里."""
        return self._bypass

    # ── 锚点 (写 memento) ────────────────────────────────────────

    def commit(
            self,
            *,
            session_id: str,
            start_turn: int,
            end_turn: int,
            message: str = "",
    ) -> CommitRef | None:
        """落一个锚点. 区间 = ``(start_turn, end_turn]`` —— **下界开、上界闭**, start = 上个 commit 的 end_turn.

        下界开意味着 turn ``start_turn`` 归**上一个** commit (它就是上一个的 ``end_turn``), 本段不重复
        覆盖它; 于是相邻 commit 严丝合缝 (``(0,1] (1,2]``), 新区间逐字抄旧的 ``end_turn`` 即可, 不需要 +1.
        ``start_turn == end_turn`` = 空区间 (没有新追认的 turn) → **不落锚点, 返回 None**.
        ``metadata.prev_turn`` 即 ``start_turn``.
        ``message`` 非空时顺便种子一条 note (便捷); 默认只落锚点 —— authoritative message 归 sidecar.
        返回本 commit 的 ``CommitRef`` (``id``/``seq`` 供排 sidecar + 造 notice); 空区间返回 None.
        """
        if start_turn == end_turn:
            return None
        ref = DshSessionRef(session_id=session_id, start_turn=start_turn, end_turn=end_turn)
        return self._branch().commit(
            message=message,
            metatype="session",
            metadata={_REF_KEY: ref.model_dump(mode="json"), _PREV_TURN_KEY: start_turn},
        )

    # ── 旁路 (慢腿: 排任务归 manager, 生命周期同 ghost) ───────────────

    def schedule_note(self, commit_id: str) -> None:
        """commit 后调用: 排一个旁路任务 —— 源 session 冷 seed 跑一轮, 产 message 写回 note.

        同一个 commit 只排一次 (幂等). 找不回切点 ref 的 commit 直接跳过 (note 留空, 可 read).
        """
        if commit_id in self._bypass:
            return
        commit = self._find_commit(commit_id)
        ref = self._ref_of(commit) if commit is not None else None
        if ref is None:
            self._logger.warning("memento bypass: commit %s has no cut ref, note stays empty", commit_id)
            return
        run = BypassCommit(commit_id=commit_id, ref=ref)
        self._bypass[commit_id] = run
        run.task = asyncio.create_task(self._run_bypass(run), name=f"memento-note-{commit_id}")

    async def _run_bypass(self, run: BypassCommit) -> None:
        """跑一轮旁路并写 note. 空 message / 传输失败 → 终态留空, 不自动重试.

        prompt 轨迹接续 (前驱坐标 + 前文); 旁路约束显式进载荷 (低思考 + maxTokens), 不靠
        plugin 的身份判定间接降级.
        """
        try:
            result = await self._connection.call(
                _BYPASS_RUN_ROUTE,
                {
                    "ref": run.ref.model_dump(mode="json"),
                    "prompt": self._note_prompt_for(run.commit_id),
                    "reasoning_effort": _NOTE_EFFORT,
                    "max_tokens": self._config.note_max_tokens,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            run.state = BypassState.FAILED
            self._logger.warning("memento bypass failed, commit %s keeps an empty note: %s", run.commit_id, exc)
            return
        message = str((result or {}).get("message", ""))
        if message == "":
            run.state = BypassState.FAILED
            self._logger.warning("memento bypass returned no text for commit %s", run.commit_id)
            return
        self._branch().note(run.commit_id, message)
        run.state = BypassState.READY

    async def drain_bypass(self) -> None:
        """等所有在飞的旁路任务收尾 (观测/测试用)."""
        while pending := [r.task for r in self._bypass.values() if r.task is not None and not r.task.done()]:
            await asyncio.gather(*pending, return_exceptions=True)

    def _note_prompt_for(self, commit_id: str) -> str:
        """为某 commit 组 note prompt: 前驱坐标 + 最近 N 条已就绪 message 作前文."""
        commit = self._find_commit(commit_id)
        seq = commit.seq if commit is not None else 0
        latest_coord, prior = self._note_context(seq)
        return _note_prompt(latest_coord=latest_coord, prior=prior)

    def _note_context(self, target_seq: int) -> tuple[str | None, list[tuple[str, str]]]:
        """本 commit 的轨迹上下文: 前驱坐标 (None = 首条) + 最近 N 条已就绪 message (老→新).

        branch append-only → 前驱 (seq < target_seq) 永不变, 与旁路异步执行时机无关.
        坏 commit / 空 message 跳过 —— 不是真摘要, 不能当前文.
        """
        branch = self._branch()
        prior_refs = [c for c in branch.commits() if c.seq < target_seq]
        latest = prior_refs[-1] if prior_refs else None
        latest_coord = f"{branch.index}-{latest.seq}" if latest is not None else None
        prior: list[tuple[str, str]] = []
        for commit in prior_refs[-_NOTE_PRIOR_COUNT:]:
            view = branch.get_commit(commit.seq)
            if view is not None and not view.is_broken and view.message.strip():
                prior.append((view.coord, view.message))
        return latest_coord, prior

    # ── read / chat (旁路原语的另两个调用方; 目标用坐标引用) ──────────

    async def read(self, coord: str) -> str:
        """commit 区间 → 可读文本.

        取的是源 session 的**原始 log 区间**, 不是 surface 投影 —— transcript 要 ``tool/call``
        这类只在 log 里的记录. 源 session 可以是冷的 (上次运行留下的), plugin 侧冷读兜底.
        坏 commit 照读: 坏的是摘要, 原文还在.
        """
        view = self._require_commit(coord)
        ref = self._ref_of_view(view)
        raw = await self._connection.call(_READ_ROUTE, {"ref": ref.model_dump(mode="json")})
        events = [SessionEvent.from_dict(event) for event in (raw or {}).get("events", [])]
        return render_transcript(events)

    async def chat(self, coord: str, prompt: str) -> str:
        """和某条 commit **所属的上下文**对话一轮 (走旁路原语).

        坏 commit 直接失败 —— 占位 message 不是真摘要, 拿它当上下文没有意义.
        prompt 前缀点破"对话对象是上下文而不是 commit 本身", 否则模型会以为在和一条记录说话.
        """
        view = self._require_commit(coord)
        if view.is_broken:
            raise BrokenCommitError(f"commit {view.coord} has no context to talk to; read it instead")
        ref = self._ref_of_view(view)
        result = await self._connection.call(
            _BYPASS_RUN_ROUTE,
            {"ref": ref.model_dump(mode="json"), "prompt": f"{_CHAT_PREAMBLE}\n\n{prompt}"},
        )
        text = str((result or {}).get("message", ""))
        if text == "":
            raise RuntimeError(f"chat with commit {view.coord} returned no text")
        return text

    def _require_commit(self, coord: str) -> CommitView:
        view = self.resolve(coord)
        if view is None:
            raise KeyError(f"commit {coord} not found")
        return view

    def _ref_of_view(self, view: CommitView) -> DshSessionRef:
        ref = self._ref_of(view.ref)
        if ref is None:
            raise RuntimeError(f"commit {view.coord} carries no session ref")
        return ref

    # ── notice 构造 (ego 排队列, 在 thinking-enter 注入) ──────────────

    def warn_notice(self, growth: int) -> Message:
        """K 阈值提醒 (每窗口一次): 催模型在话题边界主动 commit."""
        text = (
            f"本段对话已积累约 {growth} tokens 的未提交内容. 若话题已到边界, "
            f"用 commit 提交一个锚点 (带一句 message 说明这段讲了什么)."
        )
        return Message.new(tag="memento_notice", attributes={"kind": "warn"}).with_content(text)

    def committed_notice(self, commit: CommitRef) -> Message:
        """某 commit 已生成 — 告知模型锚点已落, note 稍后旁路补."""
        ref = self._ref_of(commit)
        span = f"{ref.start_turn}-{ref.end_turn}" if ref is not None else str(commit.seq)
        return Message.new(tag="memento_notice", attributes={"kind": "committed"}).with_content(
            f"已生成 commit (turns {span}); 摘要由旁路补上."
        )

    def latest_ref(self) -> DshSessionRef | None:
        """branch tip 的 ref (= 下一个 commit 的区间下界); 无 commit 返回 None."""
        branch = self._memento.get_branch(self._config.branch_name)
        if branch is None:
            return None
        commits = branch.commits()
        return self._ref_of(commits[-1]) if commits else None

    # ── 阈值 (纯算法; 窗口状态由 ego 持) ─────────────────────────

    def window_size(self, usage: TokenUsage) -> int:
        """这次调用的窗口大小 (ego 记下它, 与 window_base 求增量)."""
        return _input_size(usage)

    def evaluate(self, growth: int, warned: bool) -> CommitDecision:
        """按增量 token 判定: FORCE (强制 commit) / WARN (提醒一次) / NONE."""
        if growth >= self._config.force_tokens:
            return CommitDecision.FORCE
        if growth >= self._config.warn_tokens and not warned:
            return CommitDecision.WARN
        return CommitDecision.NONE

    # ── 读侧投影 (branch / commit 都是纯读, 不触 dsh) ───────────────

    def view_message(self, name: str | None = None, *, n: int | None = None) -> Message | None:
        """把 branch view 渲染成 xml-like memory 块 (坏 commit 已由 memento view 折叠).

        ``name`` 缺省 = 当前 branch (ego 的 memory 用这条); 给了名字就读别的 branch.
        """
        branch = self._memento.get_branch(name or self._config.branch_name)
        if branch is None:
            return None
        return self._render_view(branch.view(n=n if n is not None else self._config.view_limit))

    def list_branches(self) -> list[BranchInfo]:
        """所有现存 branch 的概览 —— 先看有哪些, 再用 ``view_message(name)`` 看内容."""
        infos: list[BranchInfo] = []
        for ref in self._memento.list_branches():
            branch = self._memento.get_branch(ref.name)
            if branch is None:
                continue
            commits = branch.commits()
            latest = branch.get_commit(commits[-1].seq) if commits else None
            infos.append(BranchInfo(
                name=ref.name,
                index=branch.index,
                description=ref.description,
                commits_total=len(commits),
                latest_coord=latest.coord if latest is not None else "",
                latest_title=latest.title if latest is not None else "",
                created=ref.created,
            ))
        return infos

    def list_commits(
            self,
            name: str | None = None,
            *,
            from_date: datetime | None = None,
            until_date: datetime | None = None,
    ) -> list[CommitDigest]:
        """按时间区间列 commit (seq + title + body), 等价 ``git log --oneline``; 详情就在同一条里."""
        branch = self._memento.get_branch(name or self._config.branch_name)
        if branch is None:
            return []
        digests: list[CommitDigest] = []
        for ref in branch.query_commits(from_date=from_date, until_date=until_date):
            view = branch.get_commit(ref.seq)
            if view is None:
                continue
            digests.append(CommitDigest(
                coord=view.coord, seq=view.seq, title=view.title, body=view.body,
                created=view.created, broken=view.is_broken,
            ))
        return digests

    def resolve(self, coord: str) -> CommitView | None:
        """坐标 (如 ``1-27``) → CommitView; 格式错 / 不存在返回 None. read / chat 的入口."""
        return self._memento.resolve_commit(coord)

    def resume_ref(self) -> DshSessionRef | None:
        """ego 重建的切点: 最后一个**摘要已就绪**的 commit 的 ref; 没有则 None (= 全新 session).

        只认 `note.message` 非空的 commit —— 摘要区靠它, 没有它就退化成"整段原文再带一遍" (无效还原).
        更晚的 commit 若 note 还没生产出来, 它到切点之间的原文就作为 raw 尾巴带过去, 不丢内容.
        """
        branch = self._memento.get_branch(self._config.branch_name)
        if branch is None:
            return None
        notes = branch.notes()
        cut: DshSessionRef | None = None
        for commit in branch.commits():
            note = notes.get(commit.id)
            if note is not None and note.message:
                cut = self._ref_of(commit)
        return cut

    # ── 渲染 ─────────────────────────────────────────────────────

    def _render_view(self, view: BranchView) -> Message:
        container = Message.new(tag="branch", attributes={"name": view.name, "index": str(view.index)})
        commits = view.history + view.latest
        if not commits:
            return container.with_content("no commit")
        return container.with_messages(*[self._render_commit(cv) for cv in commits])

    def _render_commit(self, cv: CommitView) -> Message:
        attributes = {"seq": cv.coord, "created": cv.created.isoformat()}
        if cv.is_broken:
            attributes["error"] = "1"
        return Message.new(tag="commit", attributes=attributes).with_content(cv.message)

    # ── 内部读取 ─────────────────────────────────────────────────

    def _branch(self) -> Branch:
        branch = self._memento.get_branch(self._config.branch_name)
        if branch is None:
            raise RuntimeError(f"memento branch '{self._config.branch_name}' not found")
        return branch

    def _ref_of(self, commit: CommitRef) -> DshSessionRef | None:
        raw = commit.metadata.get(_REF_KEY)
        return DshSessionRef(**raw) if raw else None

    def _find_commit(self, commit_id: str) -> CommitRef | None:
        for commit in self._branch().commits():
            if commit.id == commit_id:
                return commit
        return None
