"""Mindflow reflexive control channel — 反身控制面.

把 mindflow 从 opaque 调度器变成 ghost 可感知、可操纵的透明面.

常驻能力面直接挂在 mindflow 父 channel 上, 不折叠. 全部是命令, 没有 instruction /
notice / context —— 注意力状态本身就在 ghost 的上下文里, 再报一遍是冗余; 需要理解
mindflow 语义时, 走 ``specification`` 拉 blueprint 源码, 而不是常驻展开机制说明.

命令面:
- ``peek`` / ``claim`` / ``nuclei`` — 感知自身的读面 (看 / 取 / 拓扑).
- ``set-priority`` / ``set-signal-bar`` / ``set-impulse-bar`` — 注意力治理 (常驻一等能力).
- ``specification`` — 指向 mindflow 权威契约的模块路径, 平时不用.

gate 只折叠各 nucleus 的子通道 —— 那是"按需展开的细节", 不是 mindflow 自身的控制面.

这是 core 内部面, 经 ``Mindflow.as_channel()`` 挂进 shell, 而非随包分发的
app channel. 反身控制原则: 刻意自省, 不做每帧轮询.
"""

from __future__ import annotations

import time

from ghoshell_moss.core.blueprint.mindflow import Mindflow, Priority
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel, PrimeChannel
from ghoshell_moss.core.concepts.channel import Channel

__all__ = ["build_mindflow_channel"]

_SPECIFICATION_PATH = "ghoshell_moss.core.blueprint.mindflow"

_HEAD_LIMIT = 40


def _parse_priority(value: str) -> Priority:
    """按名称把 CTML 传入的优先级字符串解析为 Priority."""
    try:
        return Priority[value.strip().upper()]
    except KeyError:
        raise ValueError(
            f"invalid priority {value!r}; expected one of {[p.name for p in Priority]}"
        )


def _impulse_head(impulse) -> str:
    """message 载荷的短预览 — 只提示大概内容, 完整载荷经 claim 才出."""
    if not impulse.messages:
        return ""
    text = " ".join(m.to_content_string() for m in impulse.messages)
    text = " ".join(text.split())
    if len(text) > _HEAD_LIMIT:
        return text[:_HEAD_LIMIT] + "…"
    return text


def _impulse_state_line(name: str, impulse) -> str:
    """impulse 的校验状态摘要: source / priority / strength / age / expires.

    age 与 expires 是相对当前的可读时间; message 只给 head 预览, 不展全文.
    """
    age = time.time() - impulse.created_at.timestamp()
    if impulse.stale_timeout > 0:
        remaining = impulse.stale_timeout - age
        expires = "expired" if remaining < 0 else f"in {remaining:.1f}s"
    else:
        expires = "never"
    parts = [
        f"{name} {impulse.priority.name}",
        f"strength={impulse.strength}",
        f"age={age:.1f}s",
        f"expires={expires}",
    ]
    head = _impulse_head(impulse)
    if head:
        parts.append(f'"{head}"')
    return "  " + " ".join(parts)


def build_mindflow_channel(
        mindflow: Mindflow,
        name: str = "mindflow",
        *,
        enable_priority: bool = True,
        enable_bar: bool = True,
        gate: bool = True,
) -> PrimeChannel:
    """构建 mindflow 反身控制 channel.

    常驻能力面全部挂在父 channel 上, 不进 gate —— 这些是 mindflow 自身的控制面,
    折叠后模型不会用到. gate 只折叠各 nucleus 的子通道.

    :param enable_priority: 暴露 set-priority (改当前 attention 优先级), 仅在
        attention 活跃时可见.
    :param enable_bar: 暴露 set-signal-bar / set-impulse-bar (全局水位).
    :param gate: 开启后各 nucleus 的子通道默认关闭, 由 mount_child 披露.
    """
    channel = new_prime_channel(name, description=mindflow.description(), gate=gate)

    # --- 注意力治理 (常驻一等能力) --- #

    @channel.build.command(
        name="set-priority",
        available=lambda: enable_priority and mindflow.attention() is not None,
    )
    async def set_priority(priority: str) -> str:
        """Raise or lower the priority of the current attention (raise to hold, lower to yield).

        Only available while an attention is active.
        """
        value = _parse_priority(priority)
        attn = mindflow.attention()
        if attn is None:
            return "no active attention"
        attn.set_priority(value)
        return f"attention priority set to {value.name}"

    @channel.build.command(name="set-signal-bar", available=lambda: enable_bar)
    async def set_signal_bar(priority: str) -> str:
        """Set the global signal priority floor; signals below it are dropped."""
        value = _parse_priority(priority)
        mindflow.set_signal_priority_bar(value)
        return f"signal bar set to {value.name}"

    @channel.build.command(name="set-impulse-bar", available=lambda: enable_bar)
    async def set_impulse_bar(priority: str) -> str:
        """Set the global impulse priority floor; impulses below it cannot challenge."""
        value = _parse_priority(priority)
        mindflow.set_impulse_priority_bar(value)
        return f"impulse bar set to {value.name}"

    # --- 感知读面 --- #

    @channel.build.command(name="nuclei")
    async def nuclei() -> str:
        """List your sensing units (nuclei): name, running state, description.

        This is the topology of your senses — not what they currently hold.
        """
        lines = ["mindflow nuclei:"]
        for nucleus_name, nucleus in mindflow.nuclei().items():
            state = "running" if nucleus.is_running() else "idle"
            lines.append(f"  {nucleus_name} ({state}): {nucleus.description()}")
        return "\n".join(lines)

    @channel.build.command(name="peek")
    async def peek() -> str:
        """See every sensing unit that holds something, without taking it.

        Lists each held impulse's state — source, priority, strength, age, expiry —
        plus a short message preview. The full content only leaves a unit through
        `claim`.
        """
        lines = ["peek (units holding something):"]
        for nucleus_name, nucleus in mindflow.nuclei().items():
            if not nucleus.is_running():
                continue
            if impulse := nucleus.peek():
                lines.append(_impulse_state_line(nucleus_name, impulse))
        if len(lines) == 1:
            return "peek: nothing held"
        return "\n".join(lines)

    @channel.build.command(name="claim")
    async def claim(nucleus: str) -> str:
        """Take what a sensing unit holds into your next thought.

        If it holds something, the content is queued for your next observation — you
        read it there, not here. A try: never waits for a new impulse. Does not
        reinforce the current attention.
        """
        if mindflow.nuclei().get(nucleus) is None:
            return f"no nucleus {nucleus!r}"
        claimed = mindflow.claim_impulse(nucleus)
        if claimed is None:
            return f"{nucleus}: nothing to claim"
        return f"{nucleus}: claimed — read it in your next thought"

    # --- 权威契约 --- #

    @channel.build.command(name="specification")
    async def specification() -> str:
        """Return the module path of the authoritative mindflow contract.

        Read it (via introspect / get-source) only when you need to steer attention
        deliberately — not for routine use.
        """
        return _SPECIFICATION_PATH

    # --- gate 唯一折叠的对象: 各 running nucleus 的子通道 --- #
    @channel.build.virtual_children
    def virtual_children() -> dict[str, Channel]:
        channels: dict[str, Channel] = {}
        for key, nucleus in mindflow.nuclei().items():
            if nucleus.is_running():
                if chan := nucleus.as_channel():
                    channels[key] = chan
        return channels

    return channel
