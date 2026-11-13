"""EgoMementoManager — ghost 级的 memento 旁路服务 (由 ghost 持有, 注入 ego).

职责边界 (决策落点):

- **锚点**: ``commit()`` 组 ``DshSessionRef`` + 写 memento. metadata = ``{ref, prev_turn}``,
  memento 只存不解析. 区间取「已完成的 turn」(追认): 签发时 turn/end 已知, usage 齐整.
- **view / 切点**: ``view_message()`` 给 ghost 的 memories; ``compact_cut_turn()`` 给 compact 的
  安全切点 (SCS = tip commit 的 start_turn).
- **阈值**: ``evaluate()`` 是纯算法. **窗口状态 (window_base / warned) 在 ego**, 不在这里 ——
  manager 不托管 ego 的运行时状态. 每个 commit 重置滑动窗口.
- **旁路 note 生产**: ``schedule_note()`` 为每个锚点排一个一次性 ``asyncio.Task``; task 在
  commit 的 ``end_seq`` 上跑一轮 ``chat_off_record`` (effort off; 保留会话自己的 tools —— 清空
  tools 会另起一条前缀缓存族, 整段上下文重读), 消息体是 ``<mindfork task="memento.note">``
  容器: 静态格式规则在 memento channel 的 instruction 里, 应身身份与输出契约在容器里.
  取回 plain text 写进 note. 重试 / 失败 / fallback 全在 task 内: 模型不可用按 ``note_retry_timeout``
  退避重试, 到 deadline 仍失败就留空 + named notice; 内容失败写 terminal error note.
  **没有第二次补写机制** —— task 是唯一写入者; 关停时在飞的 task 取消, 空 note 就空着.
- **补漏**: 退役. 空 note 由 named notice 暴露; 不再有开机回扫 / 第二套派发.

dep 只有不可变项 (connection / memento / config / logger / prompt_manager) —— 这些是它干活的
工具, 不是 ego 状态.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.deepseek_harness.v0_2_0 import (
    ContentBlock,
    DshSessionAnchor,
    DshSessionRef,
    TokenUsage,
    event_from_dict,
    render_transcript,
)
from ghoshell_moss.ghosts.dolores.constants import MINDFORK, NIRMANAKAYA
from ghoshell_moss.ghosts.dolores.mindset.prompter import (
    DefaultPrompterKeys,
    PromptManager,
)
from ghoshell_moss.ghosts.dolores.mindset.seeds import MEMENTO_CHAT_SEED
from ghoshell_moss.memento.abcd import Branch, BranchView, CommitRef, CommitView, Memento, Note
from ghoshell_moss.message import Message

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.v0_2_0 import DshConnection

__all__ = [
    "BranchInfo", "BrokenCommitError", "BypassCommit", "BypassState",
    "CommitDecision", "CommitDigest", "EgoMementoConfig", "EgoMementoManager",
    "SourceUnreadableError",
]

# metadata 约定 (memento 只存不解析).
_REF_KEY = "ref"
_PREV_TURN_KEY = "prev_turn"

# read 路由: 收 {ref} 回 {events}, 源 log 的 turn 区间原始切片 (live-or-cold).
_READ_ROUTE = "/moss-api/ghost/dolores/read"

# 旁路的思考强度. 不是"越省越好": `off` 会把 reasoning 块从 prompt 里拿掉, 请求因此换一条前缀
# 缓存族 (实测同一会话: 179k → 141k, 主回合的热一点吃不到). `low` 与主回合 (max) 同族, 又只付
# 少量思考 token —— 在 deepseek 家族上, 这是唯一"快、且命中"的档.
_BYPASS_EFFORT = "low"
# 旁路摘要前置上下文: 只展开最近 1 条已就绪的 commit message 作前文.
_NOTE_PRIOR_COUNT = 1


def _local(created: datetime) -> datetime:
    """The record's instant on the body's clock — the OS zone, read at render time.

    Records store an absolute instant; which clock a reader sees is a rendering question,
    and the answer is always "the body's current clock". Nothing here freezes a frame.
    """
    return created.astimezone()


def stamp_zone(created: datetime) -> str:
    """``+8`` / ``-5`` / ``+5:30`` — the offset of the clock being displayed; empty when naive."""
    offset = created.utcoffset()
    if offset is None:
        return ""
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{sign}{hours}" if minutes == 0 else f"{sign}{hours}:{minutes:02d}"


def stamp_day(created: datetime) -> str:
    """``2026-11-13+8`` — the calendar day on the body's clock, zone attached.

    Absolute on purpose: the branch spans months, and a bare ``D13`` cannot say which
    month it belongs to. The zone rides along so a bare clock underneath it stays
    unambiguous.
    """
    local = _local(created)
    return f"{local.strftime('%Y-%m-%d')}{stamp_zone(local)}"


def stamp_clock(created: datetime) -> str:
    """``15:09`` — time of day on the body's clock; the day above already carries the zone."""
    return _local(created).strftime("%H:%M")


def stamp(created: datetime) -> str:
    """``2026-11-13 15:09+8`` — the whole stamp, for things that sit outside a ``<day>``."""
    local = _local(created)
    return f"{local.strftime('%Y-%m-%d %H:%M')}{stamp_zone(local)}"


def log_emanation_usage(logger: LoggerItf, *, kind: str, subject: str, usage: TokenUsage) -> None:
    """One INFO line per model call — the ghost's token accounting, both paths.

    ``kind=note`` / ``kind=chat`` are the bypass calls: an off-record call never enters the
    session log, so without this line the ghost's parallel thinking is invisible to cost
    statistics. ``kind=ego`` is the main path, logged per step by the run loop — without it a
    cache question could only be answered by subtracting two bypass notes.

    The format stays flat key=value pairs: a future aggregator greps one prefix instead of
    parsing prose.
    """
    logger.info(
        "emanation usage kind=%s %s input=%s cache_read=%s cache_write=%s output=%s reasoning=%s",
        kind, subject,
        usage.inputTokens, usage.cacheReadTokens, usage.cacheWriteTokens,
        usage.outputTokens, usage.reasoningTokens,
    )


class BrokenCommitError(RuntimeError):
    """chat 的目标 commit 的 message 是坏占位 (非真摘要), 不可对话."""


class SourceUnreadableError(RuntimeError):
    """源 session 读不出来: 当前 dsh 拒绝这份存档, 且上游没有兼容接口.

    不是 commit 内容坏, 是**存储格式与当前 harness 不兼容** —— 典型是 0.1.5 时代写的 ego
    session 在 0.2.0 的 v4 不变式 (surface 首节点必须是受保护的 system/message) 下不合法,
    冷读时被 dsh 的格式校验拒掉 (``stored log is corrupt: … requires a protected first
    surface head`` / ``migration refuses the transformed artifact``).

    处置: **显式抛出, 不回落不猜**. 每读一次就报一次 —— 让调用方看见"这里确实没了", 而不是
    拿到一段像正文的半成品. 上游没有兼容读接口 (0.2.1-rc.2 里这条规则一字未改), 所以这是
    终态, 不是待修项.
    """


class EgoMementoConfig(BaseModel):
    """``.dolores.yml`` 的 ``memento:`` 段. Field 默认即兜底."""

    branch_name: str = Field(default="main", description="ego 提交的 memento branch (ghost 启动时取其/建).")
    view_limit: int = Field(
        default=6,
        description="Branch.view(n=..) — 最近 n 条进 memory view (详展窗口).",
    )
    digest_titles: int = Field(
        default=20,
        description=(
            "折叠段 (earlier) 保留多少条 title. 其余只有 count + 坐标区间 —— 这是个预算, "
            "不是渲染全部 commit 的承诺."
        ),
    )
    warn_tokens: int = Field(
        default=50_000,
        description="K: 距上次 commit 的窗口增量 token 达此值, 插**一次** commit 提醒 (每窗口).",
    )
    force_tokens: int = Field(
        default=100_000,
        description="T: 达此值强制建锚点. 置 0 = 每 turn commit (验证期用).",
    )
    compact_tokens: int = Field(
        default=600_000,
        description=(
            "C: 上下文**绝对**窗口上限 (最近一次调用的 prompt 大小). 达此值且上一 run 有效时, "
            "下一个 run 启动前做 compact (原地 surface replace, 替换内容 = memory 视图). "
            "应设成实际模型 context window 的某比例 (架构师量级 400k-800k, 取中位). 0 = 关闭自动 compact."
        ),
    )
    note_max_tokens: int = Field(
        default=4096,
        description=(
            "note 请求的输出上限 (thinking + title + body). low 档下思考与正文共享这个天花板; "
            "正文预算另见 note_body_tokens."
        ),
    )
    note_body_tokens: int = Field(
        default=1500,
        description=(
            "note 正文的预算 —— 写进 commit note 规则 (channel instruction 与旁路请求共用). "
            "title 另计. 与 note_max_tokens 分开: 前者是给模型看的约束, 后者是请求的硬天花板."
        ),
    )
    note_retry_timeout: float = Field(
        default=180.0,
        description="note task 对模型不可用的总重试窗口 (秒); 到点仍失败 → 留空 + named notice.",
    )
    note_backoff_base: float = Field(
        default=5.0,
        description="note task 退避基数 (秒), 指数增长, 单次等待封顶 60s.",
    )


class BypassState(str, Enum):
    """旁路任务的运行状态 (manager 持有的治理状态; note 本身的真值在 memento, 不在这里)."""

    RUNNING = "running"  # 旁路任务在飞
    READY = "ready"      # note 已落
    FAILED = "failed"    # 内容失败已写 error note; 模型不可用到 deadline 则留空 (内容仍可 read)


@dataclasses.dataclass
class BypassCommit:
    """一个被旁路治理的 commit: 切点 ref + 状态 + 在飞任务."""

    commit_id: str
    ref: DshSessionRef
    state: BypassState = BypassState.RUNNING
    task: asyncio.Task | None = None


class BranchInfo(BaseModel):
    """一个 branch 的概览 —— 先看有哪些 branch, 再用 ``view_message(name)`` 看内容.

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


# 源存档读不出来时, dsh 的两类错误文本 (错误码不可控, 只认稳定关键字):
# 1. 格式代际不支持 —— 迁移/解码层直接 refuse;
# 2. 存档不满足当前格式不变式 —— 校验层判它 corrupt, 但字节没坏.
_UNREADABLE_MARKERS = (
    "SessionFormatUnsupported",
    "reads only v3",
    "log format v4",
    "stored log is corrupt",
    "requires a protected first surface head",
    "refuses the transformed artifact",
)


def _is_unreadable_source(exc: Exception) -> bool:
    """这个错误是不是"源存档读不出来" (格式不兼容), 而不是"旁路本身失败".

    读不了 ≠ commit 内容坏: 前者是版本漂移, 后者是摘要没产出. 读侧要显式抛
    (:class:`SourceUnreadableError`), 旁路则走 ``_skip`` —— 不写 error note, 不标 broken.
    """
    text = str(exc)
    return any(marker in text for marker in _UNREADABLE_MARKERS)


def _unreadable_reason(exc: Exception) -> str:
    """压一行可读原因: 去掉 dsh 的 (raw log: /abs/path) 尾巴 —— 路径不进模型上下文."""
    text = str(exc)
    cut = text.find("(raw log:")
    if cut >= 0:
        text = text[:cut].rstrip(" ;:")
    return text


class EgoMementoManager:
    """ghost 级的 memento 旁路服务. 见模块 docstring."""

    def __init__(
            self,
            *,
            connection: DshConnection,
            memento: Memento,
            config: EgoMementoConfig | None = None,
            prompt_manager: PromptManager | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self._connection = connection
        self._memento = memento
        self._config = config or EgoMementoConfig()
        self._prompt_manager = prompt_manager
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
        丢下的空 note 不是永久空洞: named notice 会把它暴露给 ghost —— 没有第二次补写机制.
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
        commit = self._branch().commit(
            message=message,
            metatype="session",
            metadata={_REF_KEY: ref.model_dump(mode="json"), _PREV_TURN_KEY: start_turn},
        )
        # 旁路: 锚点已落地. hook 机制已整体移除 (文件即契约无法随 dolores 升级) ——
        # 先在代码侧重做, 完成后再把进程侧作为薄适配层接回:
        # self.fire_hook("on_commit", commit)
        return commit

    # ── 节点空间 (写者自己的地盘) ─────────────────────────────────

    async def open_node(self, coord: str) -> tuple[str, list[tuple[str, int, str]]]:
        """打开某条 commit 自己的节点空间 —— get-or-create, 再列一层内容.

        返回 ``(节点目录绝对路径, [(名字, 字节数, 类型)])``; 字节数 ``-1`` = 读不到 (调用侧
        按 ``?`` 显示, 不当 0). 排列与 ``core/file_editor`` 的 ``ls`` 同形 (目录先、名字序).

        **一碰就建**: 调一次就 ensure 目录 + 种子索引, 于是 ``memento=`` 从此挂在那条锚点上.
        契约上"节点存在"因此只说明"有人开过这个空间", 不说明里面有东西 —— 别拿存在当内容.

        **磁盘活整体丢进线程**: 一次 CTML 调用不该把 shell 的 event loop 压在几次 stat 上.
        坐标不存在抛 ``KeyError`` (命令侧转文本提示, 与 read/chat 同一条约定).
        """

        def _work() -> tuple[str, list[tuple[str, int, str]]]:
            view = self._require_commit(coord)
            index = self._branch().ensure_memento(view.seq)  # 已在则绝不覆盖
            node_dir = index.parent
            entries: list[tuple[str, int, str]] = []
            for entry in sorted(node_dir.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = -1
                kind = "dir" if entry.is_dir() else "symlink" if entry.is_symlink() else "file"
                entries.append((entry.name, size, kind))
            return str(node_dir), entries

        return await asyncio.to_thread(_work)

    # ── 旁路 (慢腿: 排任务归 manager, 生命周期同 ghost) ───────────────

    def schedule_note(self, commit_id: str) -> None:
        """commit 后调用: 排一个一次性 note task —— 在 commit 的 end_seq 上跑一轮
        ``chat_off_record``, 产 message 写回 note.

        同一个 commit 只排一次 (幂等). **已经有 note 的不排** —— 旁路只补空白: ghost 自己
        写了 note (``commit(message=...)``), 那条 commit 就已经有作者了. 找不回切点 ref 的
        commit 跳过 (note 留空, 可 read).
        """
        if commit_id in self._bypass or self._note_of(commit_id) is not None:
            return
        commit = self._find_commit(commit_id)
        ref = self._ref_of(commit) if commit is not None else None
        if ref is None:
            self._logger.warning("memento bypass: commit %s has no cut ref, note stays empty", commit_id)
            return
        run = BypassCommit(commit_id=commit_id, ref=ref)
        self._bypass[commit_id] = run
        run.task = asyncio.create_task(self._guarded_note(run), name=f"memento-note-{commit_id}")

    async def _guarded_note(self, run: BypassCommit) -> None:
        """The note task's last resort — every escape from ``_run_bypass`` ends here.

        ``_run_bypass`` paves its own failures (model unavailable / unreadable source / empty
        reply), but a tail of it runs **outside** that paving: the usage log, the shape of the
        reply, and the note write itself. A death there used to vanish. The task is fire-and-
        forget, and the shutdown path only gathers tasks that are *still running* — so a task
        that had already died with an exception was never retrieved: one
        ``Task exception was never retrieved`` line at GC time, no commit id, not in the log.

        So the task carries its own catch. Nothing here changes the outcome — the note stays
        empty and the commit is still readable — it only makes the outcome visible and settles
        the run state instead of leaving it "running" forever.
        """
        try:
            await self._run_bypass(run)
        except asyncio.CancelledError:
            # 关停取消不算失败: 不写 error note, 不改状态 —— 直接穿透.
            raise
        except Exception:
            self._logger.exception("memento note task died: commit=%s", run.commit_id)
            run.state = BypassState.FAILED

    async def _run_bypass(self, run: BypassCommit) -> None:
        """跑一轮 note task 并写 note.

        成功 → 写 message. 内容失败 (空文本) → 写一条 terminal error note (``_fail``).
        模型不可用 (传输/API) → 在 ``note_retry_timeout`` 内退避重试; 到点仍失败就留空 +
        日志 —— named notice 是唯一外部记录. 源不可读 → ``_skip``.
        **取消不算失败** —— 关停时 CancelledError 直接穿透, 不写 error.
        """
        # follow=False: 这是一次性把手, 不开 `session/follow`. 这个 ref 常常就是 ego 自己的会话,
        # 而 ego 早已 follow 着它 —— 再开一条同 streamId 的流, 宿主判违规并关掉整条 mux (1008).
        session = self._connection.attach_session(run.ref.session_id, self._logger, follow=False)
        try:
            try:
                reply = await self._note_reply_with_retry(session, run)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if _is_unreadable_source(exc):
                    self._skip(run, str(exc))
                    return
                self._logger.warning(
                    "memento note: model unavailable for commit %s; note stays empty: %s",
                    run.commit_id, exc,
                )
                run.state = BypassState.FAILED
                return
        finally:
            with contextlib.suppress(Exception):
                await session.close()
        log_emanation_usage(
            self._logger, kind="note", subject=f"commit={run.commit_id}", usage=reply.usage,
        )
        message = reply.text.strip()
        if message == "":
            self._fail(run, "run returned no text")
            return
        self._branch().note(run.commit_id, message)
        run.state = BypassState.READY

    async def _note_reply_with_retry(self, session, run: BypassCommit):
        """One note call, retried while the model is unavailable.

        Content failures (no text) are not retried here — they are the caller's ``_fail``.
        The retry window is bounded by ``note_retry_timeout``; backoff is exponential and
        capped at 60s. CancelledError always propagates.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.note_retry_timeout
        attempt = 0
        while True:
            try:
                return await self._note_call(session, run)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if _is_unreadable_source(exc):
                    raise
                now = loop.time()
                if now >= deadline:
                    raise
                delay = min(
                    self._config.note_backoff_base * (2 ** attempt),
                    60.0,
                    deadline - now,
                )
                attempt += 1
                await asyncio.sleep(delay)

    async def _note_call(self, session, run: BypassCommit):
        anchor = DshSessionAnchor(
            session_id=run.ref.session_id, seq=run.ref.end_seq, turn=run.ref.end_turn,
        )
        # No `readonly`: the flag empties the tools menu, which starts a *new* prefix family and
        # costs the whole conversation cache (measured: 34,432 → 6,912 cacheReadTokens on the
        # same session) — and a model without its tool declarations emits tool-call syntax as
        # plain text. An off-record call cannot execute a tool anyway (the plugin's one-shot is
        # a bare llm.stream), so keeping the session's tools costs nothing and keeps the cache.
        return await session.chat_off_record(
            [[ContentBlock(type="text", text=self._note_message(run.commit_id))]],
            anchor=anchor,
            llm={"reasoningEffort": _BYPASS_EFFORT},
            max_tokens=self._config.note_max_tokens,
        )

    def _fail(self, run: BypassCommit, reason: str) -> None:
        """把一次内容失败写成终态: 落一条 error note (不是留空), 并记日志.

        error note 是终态 (memento 契约: never retried), 同时让读侧 ``CommitView.is_broken``
        成立, view 会把连续的 broken commit 折起来. 没有第二次补写机制 —— 写盘本身再失败就
        只能记日志了, 这已经是最后一道.
        """
        run.state = BypassState.FAILED
        try:
            self._branch().note(run.commit_id, message="", error=reason)
        except Exception:
            self._logger.exception("memento bypass: cannot mark commit %s broken", run.commit_id)
        self._logger.warning("memento bypass failed, commit %s marked broken: %s", run.commit_id, reason)

    def _skip(self, run: BypassCommit, reason: str) -> None:
        """源 session 读不出来 (格式代际不兼容) —— 不是旁路失败, 是版本漂移.

        两个方向都算: 源比 harness 新 (v4 vs v3), 或源比当前不变式旧 (0.1.5 的 ego session 在
        0.2.0 的 protected head 规则下不合法). 写一条空 note (无 error), 但**不标 broken**:
        坏的是"读不了", 不是"commit 内容坏了". 读侧不折它, 只当它没摘要 (``view.message``
        空), 真去 read 时抛 :class:`SourceUnreadableError`.
        """
        run.state = BypassState.FAILED
        try:
            self._branch().note(run.commit_id, message="", error="")
        except Exception:
            self._logger.exception("memento bypass: cannot write skip note for commit %s", run.commit_id)
        self._logger.info("memento bypass skipped (source session unreadable under the current format): %s", reason)

    async def drain_bypass(self) -> None:
        """等所有在飞的旁路任务收尾 (观测/测试用)."""
        while pending := [r.task for r in self._bypass.values() if r.task is not None and not r.task.done()]:
            await asyncio.gather(*pending, return_exceptions=True)

    def _note_message(self, commit_id: str) -> str:
        """Build the ``<mindfork>`` container for one commit note task.

        The static rules live in the memento channel instruction (read from the active
        ``memento.note`` prompt); this message carries what only the request knows: that the
        reader is a 應身 (the same mind, forked for one task), which span to record, and that the
        reply *is* the note — nothing is executed, so a tool call would be an empty answer.
        """
        commit = self._find_commit(commit_id)
        seq = commit.seq if commit is not None else 0
        latest_coord, prior = self._note_context(seq)
        if latest_coord is None:
            connection = "This is the first commit: it covers the whole conversation so far."
        else:
            connection = (
                f"This commit continues right after {latest_coord} — record only what happened since then."
            )
        prior_lines: list[str] = []
        if prior:
            prior_lines.append("The most recent commit before this one, for continuity:")
            for coord, message in prior:
                prior_lines.append(f"- {coord}: {message}")
        prior_block = "\n".join(prior_lines)
        return (
            f"<{MINDFORK} task=\"{DefaultPrompterKeys.MEMENTO_NOTE}\">\n"
            f"<connection>{escape(connection)}</connection>\n"
            f"<prior>{escape(prior_block)}</prior>\n"
            f"You are a concurrent {NIRMANAKAYA} of yourself above. This is a one-shot task: write\n"
            "the note for the commit that ends this span. Your final output is the note itself:\n"
            "first line the title, then the body. Nothing you write is executed; do not call tools.\n"
            f"The body budget is {self._config.note_body_tokens} tokens; the title line is extra.\n"
            f"</{MINDFORK}>"
        )

    def _chat_message(self, question: str) -> str:
        """Build the ``<mindfork>`` container for one chat round: instruction + question."""
        preamble = MEMENTO_CHAT_SEED
        if self._prompt_manager is not None:
            preamble = self._prompt_manager.render(DefaultPrompterKeys.MEMENTO_CHAT)
        return (
            f"<{MINDFORK} task=\"{DefaultPrompterKeys.MEMENTO_CHAT}\">\n"
            f"<instruction>{escape(preamble)}</instruction>\n"
            f"<question>{escape(question)}</question>\n"
            f"</{MINDFORK}>"
        )

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

        源存档读不出来 (格式代际不兼容) 时抛 :class:`SourceUnreadableError` —— 不回落成空串、
        不返回一段看起来像正文的东西: 读不到就报读不到.
        """
        view = self._require_commit(coord)
        ref = self._ref_of_view(view)
        try:
            raw = await self._connection.call(_READ_ROUTE, {"ref": ref.model_dump(mode="json")})
        except Exception as exc:
            raise self._read_failure(view, ref, exc) from exc
        events = [event_from_dict(event) for event in (raw or {}).get("events", [])]
        text = render_transcript(events)
        # commit_view was a process-hook slot; gone with the mechanism. The raw span is the read.
        # ground = await self._commit_ground(view)
        # return f"{text}\n\n{ground}" if ground else text
        return text

    async def chat(self, coord: str, prompt: str) -> str:
        """和某条 commit **所属的上下文**对话一轮 (chat off record).

        坏 commit 直接失败 —— 占位 message 不是真摘要, 拿它当上下文没有意义.
        处境说明与问题装进一个 ``<mindfork task="memento.chat">`` 容器: 同一个 mindfork
        容器, instruction 是 ``memento.chat`` asset, question 是动态输入. 只发一条 user
        message; 插件按容器原样说, 不建 session、不落盘.
        """
        view = self._require_commit(coord)
        if view.is_broken:
            raise BrokenCommitError(f"commit {view.coord} has no context to talk to; read it instead")
        ref = self._ref_of_view(view)
        anchor = DshSessionAnchor(session_id=ref.session_id, seq=ref.end_seq, turn=ref.end_turn)
        # follow=False: chat 只借上下文, 不参与这个会话的运行态; 目标会话可能正被 ego follow 着.
        session = self._connection.attach_session(ref.session_id, self._logger, follow=False)
        try:
            # No `readonly`: see ``_note_call`` — emptying the tools menu costs the prefix cache.
            reply = await session.chat_off_record(
                [[ContentBlock(type="text", text=self._chat_message(prompt))]],
                anchor=anchor,
                llm={"reasoningEffort": _BYPASS_EFFORT},
            )
        except Exception as exc:
            raise self._read_failure(view, ref, exc) from exc
        finally:
            with contextlib.suppress(Exception):
                await session.close()
        log_emanation_usage(
            self._logger, kind="chat", subject=f"coord={view.coord}", usage=reply.usage,
        )
        text = reply.text
        if text == "":
            if reply.tool_calls:
                asked = "; ".join(f"{call.name}({call.arguments[:120]})" for call in reply.tool_calls)
                raise RuntimeError(
                    f"chat with commit {view.coord} answered with a tool call instead of prose: {asked}"
                )
            raise RuntimeError(f"chat with commit {view.coord} returned no text")
        return text

    def _read_failure(self, view: CommitView, ref: DshSessionRef, exc: Exception) -> Exception:
        """读源 session 失败 → 该抛什么.

        格式不兼容映射成 :class:`SourceUnreadableError` (说明"这里确实没了, 别再试"), 其它异常
        原样返回 (传输/服务端问题是真的可重试, 不该被伪装成终态).
        """
        if not _is_unreadable_source(exc):
            return exc
        return SourceUnreadableError(
            f"memento {view.coord}: the source session {ref.session_id} cannot be read by the "
            f"current dsh — the stored log does not satisfy the current format invariants, and "
            f"there is no compatible read path. The original transcript is unavailable; the "
            f"commit's note (if any) is still in view/history. dsh said: {_unreadable_reason(exc)}"
        )

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
            f"About {growth} tokens of uncommitted conversation have accumulated. If you are at a "
            f"topic boundary, commit an anchor now."
        )
        return Message.new(tag="memento_notice", attributes={"kind": "warn"}).with_content(text)

    def coord(self, commit: CommitRef) -> str:
        """commit → 模型侧的坐标 ``{branch_index}-{seq}`` (读面 / 回执 / 落锚命令共用一种写法)."""
        return f"{self._branch().index}-{commit.seq}"

    def committed_notice(self, commit: CommitRef) -> Message:
        """某 commit 已生成 — 回执锚点坐标 (模型可观测的游标); turn 区间不可观测, 不透露."""
        coord = self.coord(commit)
        return Message.new(tag="memento_notice", attributes={"kind": "committed"}).with_content(
            f"a commit was made at {coord}."
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

    def view_message(
            self,
            name: str | None = None,
            *,
            n: int | None = None,
            until: str | None = None,
    ) -> Message | None:
        """把 branch view 渲染成 xml-like memory 块 (坏 commit 已由 memento view 折叠).

        ``name`` 缺省 = 当前 branch (ego 的 memory 用这条); 给了名字就读别的 branch.
        ``until`` 是一条坐标: 渲染"那条 commit 之前"的轨迹 —— compact 之后重渲染 memory
        就走这条(切点那条的原文 tail 还留在 surface 里, 不该再被摘要一遍).
        """
        branch = self._memento.get_branch(name or self._config.branch_name)
        if branch is None:
            return None
        return self._render_view(branch.view(
            n=n if n is not None else self._config.view_limit,
            until=self._seq_before(branch, until),
        ))

    async def aview_message(
            self,
            name: str | None = None,
            *,
            n: int | None = None,
            until: str | None = None,
    ) -> Message | None:
        """``view_message`` 的异步入口 —— 开机 memory / compact 重渲染都走这条.

        ``branch_view`` 槽位随进程 hook 机制一起移除（文件即契约无法随 dolores 升级）；
        代码侧 hook 重做后，这里再接回。现在两条入口渲染的是同一份投影。
        """
        branch = self._memento.get_branch(name or self._config.branch_name)
        if branch is None:
            return None
        view = await branch.aview(
            n=n if n is not None else self._config.view_limit,
            until=self._seq_before(branch, until),
        )
        return self._render_view(view)

    def _seq_before(self, branch: Branch, coord: str | None) -> int | None:
        """坐标 → 该 branch 上的 seq; 渲染时用它做**排他**上界.

        坐标属于别的 branch 时返回 None (整条线渲染) —— 一条线的 snapshot 不该被另一条线的
        坐标悄悄截断; 传错坐标是调用方的问题, 这里只保证不会切出半个视图.
        """
        if not coord:
            return None
        view = self._memento.resolve_commit(coord)
        if view is None or view.branch_index != branch.index:
            return None
        return view.seq

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

    def compact_cut_turn(self) -> int | None:
        """安全 compact 位点 (SCS): 倒数第二个 commit 的 end_turn, 即 tip commit 的 start_turn.

        tip commit ``n`` 的 ``start_turn`` = ``end_turn(n-1)`` (半开区间 (start, end]), 所以切到它
        意味着「压到 n-1、保留最新一个 commit 的 span 作原文 tail」——「不丢最近几轮」是结构保证,
        不靠算。返回 ``None`` 表示无可压 (没有 tip / tip 就是第一个 commit, 无 n-1)。
        """
        cut = self.compact_cut()
        return None if cut is None else cut[0]

    def compact_cut(self) -> tuple[int, str] | None:
        """``(cut_turn, tip_coord)`` —— 压缩点, 外加"哪一条 commit 留在 surface 里".

        两者必须一起给: commit 的 turn 决定 surface 从哪儿切, 坐标决定 memory 重渲染到哪儿停。
        只给 turn 的话, 重渲染的人还得自己反查是哪一条 —— 而那正好是最容易错的地方。
        """
        ref = self.latest_ref()
        if ref is None or ref.start_turn <= 0:
            return None
        commits = self._branch().commits()
        if not commits:
            return None
        return ref.start_turn, self.coord(commits[-1])

    # ── 渲染 ─────────────────────────────────────────────────────

    def _render_view(self, view: BranchView) -> Message:
        """Two densities, and the count says which window this is.

        ``latest`` is the detail window (the contract's ``n``); ``history`` is
        everything older and is *folded* — one entry naming the range and the few
        titles closest to the window, never one entry per commit. The fold is what
        the contract always claimed and the renderer never did: it concatenated
        the two lists and drew every commit in full, so a 292-commit branch put
        282 of them into memory while the budget only existed in a signature.
        """
        container = Message.new(
            tag="memento",
            attributes={
                "branch": view.name,
                "index": str(view.index),
                "total": str(view.commits_total),
            },
        )
        children: list[Message] = []
        fork = self._render_fork(view)
        if fork is not None:
            children.append(fork)
        if view.history:
            children.append(self._render_digest(view))
        children.extend(self._render_days(view.latest))
        if not children:
            return container.with_content("no commit")
        return container.with_messages(*children)

    def _render_fork(self, view: BranchView) -> Message | None:
        """父线在哪儿 —— 一条指针, 不是父线的副本.

        ``at`` 在这套 prompt 里是时间词 (``<command at="D12 18:49:30+8">``); 分叉点是
        父线上的一个**位置**, 所以写 ``coord``. 父线的内容一条坐标就能取到, 不必搬过来.
        """
        parent = view.previous
        if parent is None:
            return None
        attributes = {
            "from": parent.name,
            "index": str(parent.index),
            "total": str(parent.commits_total),
        }
        if view.fork_coord:
            attributes["coord"] = view.fork_coord
        return Message.new(tag="fork", attributes=attributes)

    def _render_days(self, latest: list[CommitView]) -> list[Message]:
        """The detail window, grouped by the day each commit was written on.

        A day is the unit a reader actually folds by ("this is what that day was"), and
        this branch runs 13–25 commits a day, so the date is paid once instead of once
        per line. Commits are chronological inside a branch, so a consecutive group-by
        is enough: if the clock ever went backwards, a second ``<day>`` appears rather
        than a lie.
        """
        days: list[Message] = []
        for day, group in itertools.groupby(latest, key=lambda cv: _local(cv.created).date()):
            entries = list(group)
            days.append(
                Message.new(tag="day", attributes={"date": f"{day.isoformat()}{stamp_zone(_local(entries[0].created))}"})
                .with_messages(*(self._render_commit(cv) for cv in entries))
            )
        return days

    def _render_digest(self, view: BranchView) -> Message:
        """The folded past: how much, which slice, and the titles nearest the light.

        Bounded on purpose — ``digest_titles`` is the whole budget of the far side.
        Anything older than the window is reachable by coordinate (``read`` /
        ``history``), which is why it is safe not to ship it.
        """
        budget = self._config.digest_titles
        recent = view.history[-budget:] if budget > 0 else []
        body = "\n".join(
            f"· {cv.coord}  {escape(cv.title)}" + ("  [node]" if cv.memento is not None else "")
            for cv in recent
        )
        nodes = sum(1 for cv in view.history if cv.memento is not None)
        return Message.new(
            tag="earlier",
            attributes={
                # count 是**原始 commit 数**（未折叠 broken 前）: 它和 first/last 的坐标区间、
                # total 同一套语义; errors 说明为什么标题行可能比 count 少.
                "count": str(view.history_total),
                "errors": str(view.history_errors),
                "first": view.history[0].coord,
                "last": view.history[-1].coord,
                "titles": str(len(recent)),
                "nodes": str(nodes),
                # 折叠段原先完全没有时间 —— "这段记忆有多老"无处可查.
                "since": stamp_day(view.history[0].created),
                "until": stamp_day(view.history[-1].created),
            },
        ).with_content(body)

    def _render_commit(self, cv: CommitView) -> Message:
        attributes = {"seq": cv.coord, "at": stamp_clock(cv.created)}
        if cv.is_broken:
            attributes["error"] = "1"
        if cv.memento is not None:
            # 有没有自己的空间就够了; 路径是可算的, 用它去 ``node`` 取.
            attributes["node"] = "1"
        container = Message.new(tag="commit", attributes=attributes)
        relations = self._render_relations(cv)
        if relations:
            container.with_messages(*relations)
            container.with_content("\n")
        return container.with_content(escape(cv.message))

    def _render_relations(self, cv: CommitView) -> list[Message]:
        """汇入与分出 —— 跨线的写只有这一种形态, 所以它必须在轨迹上看得见.

        只带 title (契约里唯一被复制的内容) 与坐标; 正文留在源线, ``read`` 去取.
        放在 note 正文之前: 它们是"写这个锚点时手上已经有的东西", 不是脚注.
        """
        out: list[Message] = []
        for incoming in cv.confluences:
            attributes = {
                "branch": incoming.branch.branch_name,
                "index": str(incoming.branch.branch_index),
            }
            if incoming.coord:
                attributes["coord"] = incoming.coord
            out.append(Message.new(tag="from", attributes=attributes).with_content(escape(incoming.title)))
        for child in cv.children:
            out.append(Message.new(tag="to", attributes={
                "branch": child.branch.branch_name,
                "index": str(child.branch.branch_index),
                "at": stamp(child.created),
            }))
        return out

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

    def _note_of(self, commit_id: str) -> Note | None:
        """这条 commit 有没有 note (谁写的都算) —— 旁路排不排的唯一判据."""
        branch = self._memento.get_branch(self._config.branch_name)
        if branch is None:
            return None
        return branch.notes().get(commit_id)
