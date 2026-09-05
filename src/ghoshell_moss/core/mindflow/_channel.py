"""Mindflow reflexive control channel — 反身控制面.

把 mindflow 从 opaque 调度器变成 ghost 可感知、可操纵的透明面: 自主感知
(self-explanation via instruction/notice), 注意力管理 (set-priority + status),
优先级干预 (set-signal-bar / set-impulse-bar), nucleus pull.

命令与 context 的可用性按"机制"用 build flag 门控: 开启时命令 available()
通过并配套展示其对应状态; 关闭时命令不可见、context 不带该状态.

这是 core 内部面, 经 ``Mindflow.as_channel()`` 挂进 shell, 而非随包分发的
app channel. 反身控制原则: 刻意自省, 不做每帧轮询.
"""

from __future__ import annotations

from ghoshell_moss.core.blueprint.mindflow import Mindflow, Priority
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel, PrimeChannel
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message

__all__ = ["build_mindflow_channel"]


def _parse_priority(value: str) -> Priority:
    """按名称把 CTML 传入的优先级字符串解析为 Priority."""
    try:
        return Priority[value.strip().upper()]
    except KeyError:
        raise ValueError(
            f"invalid priority {value!r}; expected one of {[p.name for p in Priority]}"
        )


def _impulse_text(impulse) -> str:
    """impulse 的短文表示: 优先 messages, 退化到 description."""
    if impulse.messages:
        return " ".join(m.to_content_string() for m in impulse.messages)
    return impulse.description or ""


def _attention_line(attn) -> str:
    imp = attn.draw_from()
    return (
        f"source={imp.source} priority={attn.priority().name} "
        f"protected={attn.is_protected()} abort_reason={attn.abort_reason()!r}"
    )


def build_mindflow_channel(
        mindflow: Mindflow,
        name: str = "mindflow",
        *,
        enable_priority: bool = True,
        enable_bar: bool = True,
        enable_pull: bool = False,
        enable_red_dot: bool = False,
) -> PrimeChannel:
    """构建 mindflow 反身控制 channel.

    :param enable_priority: 暴露 set-priority(改当前 attention 优先级).
    :param enable_bar: 暴露 set-signal-bar / set-impulse-bar(全局水位).
    :param enable_pull: 暴露 pull(从 nucleus 主动拉取 impulse).
    :param enable_red_dot: context 展示各 nucleus 的 status() 红点.
    """
    channel = new_prime_channel(name, description=mindflow.description())

    # --- 静态心智模型 (instruction): 绝不重复罗列命令签名 --- #
    @channel.build.instruction
    def instruction() -> str:
        return (
            "## mindflow channel\n"
            "This is the reflexive surface of your own mind — the control plane over "
            "your parallel sensing and thinking units (nuclei). You are not merely a "
            "passive receiver of impulses: you can see what your sensory units currently "
            "hold and deliberately steer your own attention.\n"
            "Read before you act on your own perception. Actions here are deliberate "
            "self-observation — inspecting the current attention, overriding the global "
            "priority floors, raising or lowering the active attention's priority, or "
            "pulling an impulse out of a nucleus. Do not poll every frame; react to "
            "signals, not to your own relay."
        )

    # --- notice: 动态 nucleus name-description 列表 (拓扑变更自动 diff) --- #
    @channel.build.notice
    def notice() -> str:
        lines = ["mindflow nuclei:"]
        for name, nucleus in mindflow.nuclei().items():
            state = "running" if nucleus.is_running() else "idle"
            lines.append(f"  {name} ({state}): {nucleus.description()}")
        return "\n".join(lines)

    # --- context_messages: 按 flag 展示 Operation 后可变状态 --- #
    @channel.build.context_messages
    def context() -> list[Message]:
        blocks: list[str] = []
        if enable_priority:
            if attn := mindflow.attention():
                blocks.append(f"active attention: {_attention_line(attn)}")
        if enable_bar:
            blocks.append(f"signal bar: {mindflow.signal_priority_bar().name}")
            blocks.append(f"impulse bar: {mindflow.impulse_priority_bar().name}")
        if enable_red_dot:
            for name, nucleus in mindflow.nuclei().items():
                if nucleus.status():
                    blocks.append(f"red dot {name}: {nucleus.status()}")
        if enable_pull:
            for name, nucleus in mindflow.nuclei().items():
                impulse = nucleus.peek()
                if impulse is not None:
                    blocks.append(f"pullable {name}: {_impulse_text(impulse)}")
        if not blocks:
            return []
        return [Message.new().with_content("\n".join(blocks))]

    # --- virtual_children: 自动挂载运行中 nucleus 的子通道(真实功能, 非信息展示) --- #
    @channel.build.virtual_children
    def mindflow_nuclei_children() -> dict[str, Channel]:
        channels: dict[str, Channel] = {}
        for key, nucleus in mindflow.nuclei().items():
            if nucleus.is_running():
                if chan := nucleus.as_channel():
                    channels[key] = chan
        return channels

    # --- 命令面 --- #

    @channel.build.command(name="status", always_observe=True)
    async def status() -> str:
        """Observe your own sensing units and the active attention (self-introspection).

        Reports each nucleus: name, description, whether it is running, and its current
        top impulse (peek). Also reports the active attention if there is one. Use this
        before deciding whether to pull an impulse out of a nucleus.
        """
        lines = ["mindflow status:"]
        if attn := mindflow.attention():
            lines.append(f"  active attention: {_attention_line(attn)}")
        for name, nucleus in mindflow.nuclei().items():
            state = "running" if nucleus.is_running() else "idle"
            line = f"  {name} ({state}): {nucleus.description()}"
            if impulse := nucleus.peek():
                line += f" | peek: {_impulse_text(impulse)}"
            lines.append(line)
        return "\n".join(lines)

    @channel.build.command(name="set-priority", available=lambda: enable_priority)
    async def set_priority(priority: str) -> str:
        """Set the priority of the active attention (raise to survive, lower to forfeit).

        Only meaningful while an attention is active (a nucleus is being attended).
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

    @channel.build.command(name="pull", available=lambda: enable_pull)
    async def pull(nucleus: str) -> str:
        """Best-effort pull of a nucleus' top impulse (a try, never waits for a new one).

        Consumes the current top impulse (marks it attended), reinforces the active
        attention through absorb if one exists, and returns the impulse's messages.
        If the nucleus holds nothing, reports so cleanly — a discardable attempt.
        """
        target = mindflow.nuclei().get(nucleus)
        if target is None:
            return f"no nucleus {nucleus!r}"
        impulse = target.peek()
        if impulse is None:
            return f"{nucleus}: nothing to pull"
        # consume & reinforce current attention (if any).
        target.attended(impulse)
        if attn := mindflow.attention():
            attn.absorb_impulse(impulse)
        content = _impulse_text(impulse)
        return content or f"{nucleus}: pulled (empty)"

    return channel
