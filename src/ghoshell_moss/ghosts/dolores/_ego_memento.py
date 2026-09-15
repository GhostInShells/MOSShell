"""EgoMementoManager — ghost 级的 memento 旁路服务 (由 ghost 持有, 注入 ego).

职责边界 (决策落点):

- **锚点**: ``commit()`` 组 ``DshSessionRef`` + 写 memento. metadata = ``{ref, prev_turn}``,
  memento 只存不解析. 区间取「已完成的 turn」(追认): 签发时 turn/end 已知, usage 齐整.
- **view / 切点**: ``view_message()`` 给 ghost 的 memories; ``resume_ref()`` 给 ego 重建的切点.
- **阈值**: ``evaluate()`` 是纯算法. **窗口状态 (window_base / warned) 在 ego**, 不在这里 ——
  manager 不托管 ego 的运行时状态. 每个 commit 重置滑动窗口.
- **旁路 note 生产**: ``schedule_note()`` 为每个锚点排一个旁路任务 (身份旁路 session, seed = 源
  session 的 verbatim 前缀, 吃满前缀缓存), 取回 plain text 写进 note. **这些任务的生命周期与状态
  归 manager** (``bypass``): 关停时取消在飞的、不等 —— 缺 note 的 commit 就空着, 内容仍可 read.
  **不做重启补漏**: 空的就空着.
- 未落: ``read`` (commit 区间 → 文本)、``chat_commit``, 以及 ego 的 inflight 替换 (当前只在 ego
  开启/关闭时用 ``resume_ref()`` 重建).

dep 只有不可变项 (connection / memento / config / logger) —— 这些是它干活的工具, 不是 ego 状态.
"""

from __future__ import annotations

import asyncio
import dataclasses
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.deepseek_harness.types.refs import DshSessionRef
from ghoshell_moss.deepseek_harness.types.session_events import TokenUsage
from ghoshell_moss.memento.abcd import Branch, BranchView, CommitRef, CommitView, Memento
from ghoshell_moss.message import Message

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshConnection

__all__ = ["BypassCommit", "BypassState", "CommitDecision", "EgoMementoConfig", "EgoMementoManager"]

# metadata 约定 (memento 只存不解析).
_REF_KEY = "ref"
_PREV_TURN_KEY = "prev_turn"

# 旁路 note 生产路由 (dsh plugin 上开): 收 {ref, prompt?}, 冷 seed 跑一轮回 {message}.
_NOTE_RUN_ROUTE = "/moss-api/ghost/dolores/note/run"


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
    ) -> CommitRef:
        """落一个锚点. 区间 = ``[start_turn, end_turn]`` (**含端**), start = 上个 commit 的 end_turn.

        相邻 commit **共享边界 turn**: ``[0,1] [1,2]`` —— 边界 turn 既是上一段的收尾, 也是本段的下界.
        这是「追认」语义: 每个 commit 覆盖到「上个 commit 那一刻」为止 (不是从它之后开始).
        ``metadata.prev_turn`` 即 ``start_turn``.
        ``message`` 非空时顺便种子一条 note (便捷); 默认只落锚点 —— authoritative message 归 sidecar.
        返回本 commit 的 ``CommitRef`` (``id``/``seq`` 供排 sidecar + 造 notice).
        """
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
        """跑一轮旁路并写 note. 空 message / 传输失败 → 终态留空, 不自动重试."""
        try:
            result = await self._connection.call(_NOTE_RUN_ROUTE, {"ref": run.ref.model_dump(mode="json")})
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

    # ── 读侧投影 ─────────────────────────────────────────────────

    def view_message(self, *, n: int | None = None) -> Message | None:
        """把 branch view 渲染成 xml-like memory 块 (坏 commit 已由 memento view 折叠)."""
        branch = self._memento.get_branch(self._config.branch_name)
        if branch is None:
            return None
        return self._render_view(branch.view(n=n if n is not None else self._config.view_limit))

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
