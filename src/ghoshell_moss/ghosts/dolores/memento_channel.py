"""Dolores 的记忆器官 —— memento 反身 channel | 集成 | alpha

把 ``EgoMementoManager`` 的读面反射成 ghost 自己的器官: 看见自己有哪些轨迹线、读某一段的
原文、和某一段的上下文对话. 由 ghost 组装进 ``ghost`` 根 channel, CTML 上是它的子节点.

三层暴露, 边界见 ``channel_builder`` 的 tier:

- cold ``instruction``: 机制介绍 —— memento 是什么 (git-like 轨迹控制器)、坐标拓扑、召回
  路径三段. 不点任何命令名: 控制面由 command 的签名与 docstring 自解释.
- warm ``notice``: branch 一览, 一条线一行. 每 refresh 重算, 但 shell trajectory 会 diff
  增量重供 —— 没变即零成本; commits_total / latest 本来每次落锚就变, 是温数据.
  另有 ``chat`` 有名片段: 当前还在跑的 chat 任务数, 空态显式渲染 ``none`` —— 空槽本身
  是"机制还在"的证据, 不能与"没有这个片段"混为一谈.
- hot: 不占. 刻意自省, 不做每帧轮询.

命令以读为主 (``always_observe=True``): 结果是"信息", 模型要基于内容做下一步推理.
预期失败 (坐标不存在 / 坏 commit / 时间格式错) 直接返回文本提示, 不抛异常 —— 抛错只留给
真正的意外 (兜底).

唯一的写面是 ``commit`` —— 主动落锚的**请求**. 它先登记, 到 turn/end 才落地: 区间 =
``(上个锚点, 最后一个已完成 turn]``, 轮中落不住脚下这一轮. ghost 自己决定要不要当这条
记忆的作者: 写了 note 就自己署名 (旁路让开), 留空则摘要照旧归旁路. 回执只说"已排队",
坐标在落地后的 notice 里. 没有锚点后端 (``commit_anchor``) 时**不挂这条命令** —— 挂出来
却只会答"做不了"的命令, 就是在骗模型.

另有 ``memento_dir`` —— 算出某条锚点自己的目录地址, 不建、不读、不列. 目录内容归写者
(ghost 自带的文件工具); memento 只遵守 ``MEMENTO.md`` 这个索引约定.

面向模型的文本 (instruction / notice / 命令 docstring 与返回值) 一律英文 —— 与 dolores
的提示词体系一致; 中文只留给开发者注释.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, MutableChannel, new_channel
from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.mindflow.mind_delivery_nucleus import new_mind_delivery_signal
from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoManager
from ghoshell_moss.ghosts.dolores.mindset.prompter import (
    DefaultPrompterKeys,
    PromptError,
    PromptManager,
)
from ghoshell_moss.memento.abcd import CommitMessage

__all__ = ["build_memento_channel"]


def _instruction_text() -> str:
    """Mechanism, not a command list: what memento is, how places are addressed, how to recall.

    No command names: the interface reflects them, and a hand-written inventory drifts into a
    lie as the code changes. The notice self-explains the live map; durability under compaction
    is the shell's promise, not this channel's fact to restate.
    """
    return "\n".join([
        "## Memento (control)",
        "",
        "Memento is a git-like trajectory controller. A commit anchors one span of a session: a",
        "note of what happened, and a ref back to the raw conversation. Anchors outlive the",
        "session — readable, chat-able, recallable.",
        "",
        "The trajectory forks into branches, and branches carry confluents — references received",
        "from other lines. Locate a place by its coordinate `{branch}-{seq}`, e.g. `1-27`; an",
        "anchor may own a directory for the mementos tied to that moment.",
        "",
        "Recall by walking the topology: branch → anchor → its content, its conversation, or its",
        "directory.",
    ])


def _branches_text(manager: EgoMementoManager) -> str:
    """notice —— branch 列表; 只有当前 branch 带统计 (commits / latest).

    The list itself is owner data. The current branch is the one exception: the notice has to
    say where the ego is, so it reads that one branch's trajectory — never the others'.
    """
    entries = manager.list_branches()
    if not entries:
        return "[memento] no branch yet."
    current = manager.active_branch
    total, latest = manager.branch_stats(current)
    lines = ["[memento] branches:"]
    for index, ref in entries:
        about = f" — {ref.description}" if ref.description else ""
        if ref.name == current:
            stats = f"  commits={total} latest={latest or '(empty)'}"
            lines.append(f"  {ref.name} #{index}{about}{stats} <- you are here")
        else:
            lines.append(f"  {ref.name} #{index}{about}")
    return "\n".join(lines)


def _zone(tz: str | None) -> tzinfo:
    """``tz`` → tzinfo. Empty = the machine's clock; ``+08:00`` = a fixed offset; else IANA."""
    text = (tz or "").strip()
    if not text:
        return datetime.now().astimezone().tzinfo
    if text[0] in "+-":
        sign = 1 if text[0] == "+" else -1
        hours, _, minutes = text[1:].partition(":")
        return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes or 0)))
    return ZoneInfo(text)


def _parse_date(value: str | None, field: str, tz: str | None = None) -> datetime | None:
    """ISO 时间串 → aware datetime; 空 = None. 解析不了抛 ValueError (命令侧转成文本提示).

    不带偏移的时间按 ``tz`` 解释 (IANA 名或 ``+08:00``); ``tz`` 空 = 本机时钟 —— 与渲染出来的
    ``<day date="2026-11-13+8">`` 同一套口径. 按 UTC 解释会让 19:30 的区间查到凌晨, 而不报错:
    差八小时比报错更难发现.
    """
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(
            f"{field} is not an ISO timestamp: {value!r} "
            "(e.g. 2026-09-01 or 2026-09-01T10:00:00+08:00)"
        ) from None
    if parsed.tzinfo is not None:
        return parsed
    try:
        zone = _zone(tz)
    except Exception:
        raise ValueError(f"tz is not a timezone: {tz!r} (an IANA name or an offset like `+08:00`)") from None
    return parsed.replace(tzinfo=zone)


# The two anchors a checkout writes. Their description lives here once: the command's own doc
# renders from this table, so a wording change cannot rot in one place while the other keeps
# the old story.
_CHECKOUT_OPS: tuple[tuple[str, str], ...] = (
    ("depart", "written on the line you leave; its target is the line you move to"),
    ("arrive", "written on the line you enter; its target is the source line and coordinate"),
)


def _checkout_doc() -> str:
    return (
        "Move this session onto another line — effective at the end of this turn, never inside it.\n\n"
        "The session does not move: the next turn opens with the destination's memory surface\n"
        "delivered, and every later commit belongs to the new line. Two anchors are written,\n"
        "both titled by you:\n"
        + "\n".join(f"- `{name}` — {text}" for name, text in _CHECKOUT_OPS)
        + "\n\n"
        ":param name: the line to move to; it must be active, and not the one you are on.\n"
        ":param title: one line saying why you are leaving; it titles both anchors.\n"
        ":param text__: the note body; both anchors carry it."
    )


def _commit_doc(prompt_manager: PromptManager | None, manager: EgoMementoManager) -> str:
    """The write protocol: when the anchor lands, who authors the note, and the note's guide.

    Rendered from the live ``memento.note`` prompt on every refresh — the guide is not copied
    into the docstring, so editing the prompt edits the surface, and the bypass summarizer reads
    the same text.
    """
    if prompt_manager is not None:
        try:
            guide = prompt_manager.get(DefaultPrompterKeys.MEMENTO_NOTE).content.strip()
        except PromptError as exc:
            guide = f"(commit note rules unavailable: {exc})"
    else:
        guide = ""
    parts = [
        "Queue an anchor — it takes effect at the end of this turn.",
        "A commit is a request, not a write: the span ends at the last completed turn. Write the\n"
        "note yourself and you are its author; leave it empty and the background summarizer writes\n"
        "it. The coordinate arrives in the branch notice after the anchor lands.",
        "`replan` — when the work is done, or you are stuck in a corner, release your attention\n"
        "into one note: what holds, what is open, where you go next. The context folds away and\n"
        "nothing of this span stays in view; a new epoch opens, and you wake into the mind you\n"
        "wrote.",
    ]
    if guide:
        parts.append(guide)
    parts.append(
        f":param text__: the note — a one-line title, then the body "
        f"(≤ {manager.config.note_body_tokens} tokens).\n"
        ":param op: empty for an ordinary anchor; `replan` folds the context behind this note."
    )
    return "\n\n".join(parts)


def build_memento_channel(
    manager: EgoMementoManager,
    *,
    prompt_manager: PromptManager | None = None,
    commit_anchor: Callable[..., str] | None = None,
    checkout_hook: Callable[[str, str, str], str] | None = None,
    name: str = "memento",
    description: str | None = None,
) -> MutableChannel:
    """组装 memento 反身 channel —— ghost 的记忆器官.

    :param manager: ghost 持有的 EgoMementoManager (锚点 / 阈值 / 旁路都归它治理).
    :param prompt_manager: prompt registry; its active ``memento.note`` body becomes the
        ``commit`` command's note rules — the same rules the bypass task uses.
    :param commit_anchor: 落锚后端 (收 note, 返回一条给模型的回执). None =
        不挂 ``commit`` 命令 —— 写面归 ego (turn span 记账在它手里), 这里只透传.
    :param checkout_hook: 换线后端 (收 name / title / note, 返回回执). None = 不挂 ``checkout``;
        换线是 ego 的转场 (turn/end 落两笔锚 + 发 signal), channel 只做校验与排队.
    :param name: CTML 标签名.
    :param description: 覆盖默认描述.
    """
    if description is None:
        description = (
            "Your own trajectory — commits as anchors, lines as directions, and a directory "
            "per anchor."
        )

    chan = new_channel(name=name, description=description)

    # Live chat tasks. The receipt returns at once; the answer arrives as a mind delivery.
    # This set is the only truth of "how many chats are still running" — the named notice
    # reads it cold, and a cancelled or failed task removes its token in finally, so the
    # count cannot drift away from the tasks it counts.
    running_chats: set[object] = set()

    @chan.build.instruction
    async def _instruction() -> str:
        return _instruction_text()

    @chan.build.notice
    async def _branches() -> str:
        return _branches_text(manager)

    @chan.build.named_notices
    async def _chat_tasks() -> dict[str, str | None]:
        # "none" is a business zero, not "unchanged": the fragment is always present, so an
        # idle channel and a channel whose chat mechanism is broken cannot look the same.
        count = len(running_chats)
        return {"chat": "none" if count == 0 else f"{count} running"}

    @chan.build.command(name="view", always_observe=True)
    async def view(
        branch: str | None = None,
        limit: int | None = None,
        coord: str | None = None,
    ) -> str:
        """Look at one line's view — the window at the tip, or one scrolled back by coordinate.

        :param branch: branch name. None = the line you are on.
        :param limit: at most how many commits, counted back from the window's end.
        :param coord: cursor — render the trajectory *before* this commit, e.g. `1-700`.
            The commit itself is excluded: it is the place you already stand, not the view.
            `history` reads a range instead.
        """
        if coord:
            hit = manager.resolve(coord)
            if hit is None:
                return f"[memento] no commit at `{coord}`"
            index = manager.branch_index(branch)
            if index is not None and hit.branch_index != index:
                return (
                    f"[memento] `{coord}` lives on branch #{hit.branch_index}; "
                    "a cursor only scrolls the line it belongs to"
                )
        message = await manager.aview_message(branch, n=limit, until=coord)
        if message is None:
            return f"[memento] no branch `{branch}`"
        return message.to_content_string()

    if checkout_hook is not None:

        @chan.build.command(name="checkout", always_observe=True, doc=_checkout_doc)
        async def checkout(name: str, title: str = "", text__: str = "") -> str:
            # Validation and queueing only: the transition itself runs at turn/end, so a
            # command result can never carry content from the line the ego has not moved to.
            return checkout_hook(name, title, text__)

    @chan.build.command(name="fork")
    async def fork(name: str, description: str = "", coord: str = "") -> str:
        """Open a new line from this one — at the tip, or from a past coordinate on it.

        A line is where a direction of work lives: it keeps the parent as its fork point and
        carries its own ground. Fork from a coordinate when the direction started earlier than
        now.

        :param name: the new line's name — unique, no path separators, not `main`'s.
        :param description: one line on what this line is for; it rides the branch list.
        :param coord: fork from this commit (`{index}-{seq}`) instead of the tip; must be on the current line.
        """
        try:
            branch = manager.fork_branch(name, description, coord)
        except (ValueError, NameError) as error:
            return f"[memento] {error}"
        anchor = coord or "the tip"
        return f"[memento] forked `{branch.ref.name}` #{branch.index} from {anchor}"

    @chan.build.command(name="archive")
    async def archive(name: str) -> str:
        """Close a line and every line forked from it.

        Not deletion: the trajectory stays readable by index, the name is just released.
        `main` cannot be archived, and neither can the line you are living on.

        :param name: the branch to close.
        """
        try:
            closed = manager.archive_branch(name)
        except (ValueError, NameError, FileNotFoundError) as error:
            return f"[memento] {error}"
        if not closed:
            return f"[memento] `{name}` was already closed; nothing to archive"
        names = ", ".join(f"{addr.branch_name} #{addr.branch_index}" for addr in closed)
        return f"[memento] archived {len(closed)} line(s), deepest first: {names}"

    @chan.build.command(name="branches", always_observe=True)
    async def branches(offset: int = 0, limit: int = 20) -> str:
        """List every branch — closed lines included — one page at a time.

        :param offset: start at this 0-based branch position; `total` in the answer says how to page.
        :param limit: rows in this page, 20 by default.
        """
        total, page = manager.branch_catalog(offset, limit)
        if not page:
            return f"[memento] no branches at offset {offset} (total {total})"
        current = manager.active_branch
        lines = [f"[memento] branches {offset + 1}-{offset + len(page)} of {total}:"]
        for index, ref, active in page:
            about = f" — {ref.description}" if ref.description else ""
            status = "" if active else " (archived)"
            here = " <- you are here" if ref.name == current else ""
            lines.append(f"  {ref.name} #{index}{about}{status}{here}")
        return "\n".join(lines)

    @chan.build.command(name="read", always_observe=True)
    async def read(coord: str) -> str:
        """Read a span's raw conversation.

        :param coord: commit coordinate, e.g. `1-27`.
        """
        try:
            text = await manager.read(coord)
        except KeyError:
            return f"[memento] no commit at `{coord}`"
        return text or f"[memento] nothing was said in that span ({coord})"

    @chan.build.command(name="history", always_observe=True)
    async def history(
        branch: str | None = None,
        from_date: str | None = None,
        until_date: str | None = None,
        tz: str = "",
    ) -> str:
        """List one line's commits over a time range — like `git log --oneline`, with detail.

        A timestamp without an offset is read in `tz`; an offset it carries itself wins.

        :param branch: None = the line you are on.
        :param from_date: start of the range, inclusive; ISO.
        :param until_date: end of the range, inclusive; ISO.
        :param tz: an IANA name or an offset like `+08:00`; empty = this machine's clock.
        """
        try:
            start = _parse_date(from_date, "from_date", tz)
            end = _parse_date(until_date, "until_date", tz)
        except ValueError as error:
            return f"[memento] {error}"
        digests = manager.list_commits(branch, from_date=start, until_date=end)
        if not digests:
            return f"[memento] no commits in {branch or manager.active_branch} for that range"
        lines: list[str] = []
        for digest in digests:
            broken = " (broken placeholder, chat unavailable)" if digest.broken else ""
            lines.append(f"{digest.coord} {digest.title}{broken}")
            if digest.body:
                lines.extend(f"    {line}" for line in digest.body.splitlines())
        return "\n".join(lines)

    @chan.build.command(name="chat", always_observe=True)
    async def chat(coord: str, prompt: str) -> str:
        """Talk one round with the context of a span — the answer arrives later.

        The other side sees only what had happened by the time that commit was made. This returns
        a receipt at once, and the answer arrives as a delivery of its own, so nothing holds the
        turn. Slow and token-hungry: read the span first when you only need to recall it.

        :param coord: commit coordinate, e.g. `1-27`.
        :param prompt: what you want to ask.
        """
        view = manager.resolve(coord)
        if view is None:
            return f"[memento] no commit at `{coord}`"
        if view.is_broken:
            return f"[memento] `{coord}` has no context to talk to; read it instead"

        logger = CommandUtil.logger()

        async def _answer() -> Signal:
            # The delivery is the only way back once the receipt is out: a failure must
            # come back as content, not as an exception nobody will ever read. It goes
            # out as a mind delivery (role=mind, mode=next) and is buffered by the
            # MindDeliveryNucleus until the next turn — a lost arbitration never loses it.
            token = object()
            running_chats.add(token)
            started_at = time.monotonic()
            outcome = "aborted"
            try:
                try:
                    text = await manager.chat(coord, prompt)
                except asyncio.CancelledError:
                    # Cancellation is not a failure, but it must not look like success in
                    # the trail either: the token is removed and the reason is recorded.
                    outcome = "cancelled"
                    raise
                except Exception as exc:  # noqa: BLE001
                    outcome = f"failed: {exc}"
                    return new_mind_delivery_signal(
                        f"[memento] chat with {coord} failed: {exc}",
                        kind="memento.chat",
                        description=f"chat with {coord} failed",
                    )
                outcome = "ok"
                return new_mind_delivery_signal(
                    text,
                    kind="memento.chat",
                    description=f"answer from {coord}",
                )
            finally:
                # One line per finished task: which span, how it ended, how long it held,
                # and how many are left. A task cancelled before it delivers still leaves
                # a trace here, so a silent disappearance is always debuggable.
                running_chats.discard(token)
                logger.info(
                    "[memento] chat with %s finished: %s in %.1fs, %d still running",
                    coord, outcome, time.monotonic() - started_at, len(running_chats),
                )

        await CommandUtil.create_signal_task(closure=_answer)
        return f"[memento] asking {coord} — the answer will arrive as a signal"

    if commit_anchor is not None:

        @chan.build.command(
            name="commit",
            always_observe=True,
            doc=lambda: _commit_doc(prompt_manager, manager),
        )
        async def commit(text__: str = "", op: str = "") -> str:
            if not op:
                return commit_anchor(text__)
            note = text__.strip()
            if not note:
                # An op is a claim about the note; letting the summarizer fill it in would forge
                # that claim on a summary. Refuse instead of anchoring something that is not ours.
                return (
                    f"[memento] op `{op}` needs a note — the summarizer cannot write one for you; "
                    f"write the restart point yourself."
                )
            first, _, rest = note.partition("\n")
            try:
                message = CommitMessage(op=op, content=first.strip(), body=rest).to_str()
            except ValueError as exc:
                return f"[memento] refused: {exc}"
            return commit_anchor(message, replan=(op == "replan"))

    @chan.build.command(name="memento_dir", always_observe=True)
    async def memento_dir(coord: str) -> str:
        """The directory this commit owns — where the mementos of that moment live.

        Only the address: the directory is not created for you, and nothing here reads or lists
        it. Write into it with your own file tools. `MEMENTO.md` is its index — its presence is
        what marks this anchor as having a memento space.

        :param coord: commit coordinate, e.g. `1-27`.
        """
        try:
            path = manager.memento_dir(coord)
        except KeyError:
            return f"[memento] no commit at `{coord}`"
        return path

    return chan
