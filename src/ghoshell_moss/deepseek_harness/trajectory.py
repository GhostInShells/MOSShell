"""trajectory — ref 的物化: seed 重建 + transcript 折叠.

两个纯函数, 不碰 transport / client, 输入都是「源 session 的 log 事件序列」:

- ``seed_from_log`` — ref → 冷 seed (verbatim 前缀, tail 截断). 供 plugin 侧
  ``ctx.agents.create({seed})`` 做一次性运行; 不经过官方 ``session.fork`` (后者
  强制 workspace attach).
- ``render_transcript`` — 事件流 → 模型可读纯文本 (``>`` 用户 / ``~`` 模型 /
  ``@`` 工具调用). 供 read / export.

seed 的契约 (dsh 源码锚点, 见 dsh-fusion research):
- ``CreateAgentOptions.seed`` 必须 ``contiguous from seq 0``、lossless-JSON、
  balanced (无 open turn / dangling tool call) — ``core/agent/src/index.ts``.
- 切点必须是 ``turn/end``; 边界后吞 trailing standalone 事件到下一个
  ``turn/start`` (镜像 apiproxy fork 的 cut 规则, ``api-proxy.ts:2303``).
"""

from __future__ import annotations

from ghoshell_moss.deepseek_harness.types.refs import DshSessionRef
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

__all__ = ["SeedUnavailable", "seed_from_log", "render_transcript"]

_TURN_START = "turn/start"
_TURN_END = "turn/end"


class SeedUnavailable(Exception):
    """ref 指向的 end turn 未闭合或边界非法, 无法构 seed."""


def _content_text(content: object) -> str:
    """拼接 content block 里的 text 块 (排除 reasoning/image/tool-*)."""
    blocks = content if isinstance(content, list) else []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)


def _resolve_end_index(events: list[SessionEvent], ref: DshSessionRef) -> int:
    """反查 seed 切点: ref.end_seq 优先, 否则从 events 找 end_turn 的 turn/end."""
    if ref.end_seq is not None:
        if 0 <= ref.end_seq < len(events):
            return ref.end_seq
        raise SeedUnavailable(f"end_seq {ref.end_seq} out of log range")
    for index, event in enumerate(events):
        if event.meta.type == _TURN_END and event.data.get("turn") == ref.end_turn:
            return index
    raise SeedUnavailable(f"no turn/end for turn {ref.end_turn}")


def seed_from_log(events: list[SessionEvent], ref: DshSessionRef) -> list[SessionEvent]:
    """把 ref 物化成冷 seed: 源 log 的 verbatim 前缀, 切在 end_turn 的 turn/end.

    前置条件: ``events`` 是源 session 的完整 log, seq 连续从 0 起 (即
    ``events[i].seq == i``, dsh 的 ``seq = log.length`` 契约). 返回值满足
    ``CreateAgentOptions.seed`` 的全部要求 (连续 / balanced), 可直接喂
    ``ctx.agents.create``.

    切点规则镜像 apiproxy fork: 边界落在 ``end_turn`` 的 ``turn/end`` 上, 再
    向后吞 trailing standalone 事件 (session/title / injection) 到下一个
    ``turn/start`` — 这些事件 standalone, seed 仍 balanced.
    """
    boundary = _resolve_end_index(events, ref)
    if events[boundary].meta.type != _TURN_END:
        raise SeedUnavailable(
            f"boundary seq {boundary} is {events[boundary].meta.type}, not turn/end"
        )
    cut = boundary + 1
    while cut < len(events) and events[cut].meta.type != _TURN_START:
        cut += 1
    return events[:cut]


def _line_for(event: SessionEvent) -> str | None:
    """单事件 → 折叠行; 不参与渲染的事件返回 None."""
    data = event.data or {}
    kind = event.meta.type
    if kind == "user/message":
        source = data.get("source") or {}
        if source.get("kind") != "user":
            return None
        text = _content_text(data.get("content"))
        return f"> {text}" if text else None
    if kind == "assistant/message":
        message = data.get("message") or {}
        text = _content_text(message.get("content"))
        return f"~ {text}" if text else None
    if kind == "tool/call":
        name = data.get("name") or ""
        arguments = data.get("arguments") or ""
        return f"@ {name}({arguments})"
    return None


def render_transcript(
    events: list[SessionEvent],
    *,
    limit_turns: int | None = None,
    indent: str = "  ",
) -> str:
    """事件流 → 模型可读纯文本.

    - ``>`` 用户输入 (``user/message``, 仅 ``source.kind == "user"``, 非 injection).
    - ``~`` 模型输出 (``assistant/message`` 的 text 块; 空 content 的 usage 帧跳过).
    - ``@`` 工具调用 (``tool/call`` 的 name + arguments; ``tool/result`` 不渲染).
    - 其余 (turn/start·end, chunk, request/*, todo, injection) 跳过.
    - ``limit_turns`` 非 None 时只保留最近 N 个 turn (按 ``turn/start`` 分组).
    """
    turns: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            turns.append(current)

    for event in events:
        if event.meta.type == _TURN_START:
            flush()
            current = []
            continue
        line = _line_for(event)
        if line is not None:
            current.append(line)
    flush()

    if limit_turns is not None:
        turns = turns[-limit_turns:] if limit_turns > 0 else []

    return "\n".join(indent + line for group in turns for line in group)
