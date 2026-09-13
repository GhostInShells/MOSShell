"""Mindflow reflexive control channel — 反身控制面.

把 mindflow 从 opaque 调度器变成 ghost 可感知、可操纵的透明面: 常驻读面
(status 自省 + pull 主动 reach) 留在顶层, 低频的注意力治理 (set-priority /
set-signal-bar / set-impulse-bar) 下移为 gated 虚拟子通道 ``attention``.

虚拟子通道 = ``attention`` 治理 + 各 running nucleus 的 ``as_channel()``.
gate 开启时默认关闭, 由 mount_child 逐一披露; 关闭时直接挂载.

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


def _build_attention_child(
        mindflow: Mindflow,
        enable_priority: bool,
        enable_bar: bool,
) -> Channel | None:
    """构建注意力治理子通道 (gated virtual child).

    只承载「调方向盘」类命令 (改当前 attention 优先级 + 全局水位), 不含常驻
    读面 (status) 与主动 reach (pull). enable_priority / enable_bar 决定哪些
    机制注册进来; 两者都关则不注册该子通道.
    """
    if not enable_priority and not enable_bar:
        return None

    child = new_prime_channel(
        "attention",
        description="govern your own attention: override priority and global floors",
    )

    @child.build.instruction
    def instruction() -> str:
        return (
            "## attention governance\n"
            "Deliberate steering of your own attention arbitration. You are overriding "
            "runtime state, not the declared priorities of nuclei. Raise the active "
            "attention's priority to protect it, lower it to yield; raise a global floor "
            "to drop weaker signals/impulses. React to what `status` shows — do not tune "
            "blindly."
        )

    if enable_priority:
        @child.build.command(name="set-priority")
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

    if enable_bar:
        @child.build.command(name="set-signal-bar")
        async def set_signal_bar(priority: str) -> str:
            """Set the global signal priority floor; signals below it are dropped."""
            value = _parse_priority(priority)
            mindflow.set_signal_priority_bar(value)
            return f"signal bar set to {value.name}"

        @child.build.command(name="set-impulse-bar")
        async def set_impulse_bar(priority: str) -> str:
            """Set the global impulse priority floor; impulses below it cannot challenge."""
            value = _parse_priority(priority)
            mindflow.set_impulse_priority_bar(value)
            return f"impulse bar set to {value.name}"

    return child


def build_mindflow_channel(
        mindflow: Mindflow,
        name: str = "mindflow",
        *,
        enable_priority: bool = True,
        enable_bar: bool = True,
        enable_pull: bool = False,
        gate: bool = False,
) -> PrimeChannel:
    """构建 mindflow 反身控制 channel.

    :param enable_priority: 在 attention 子通道暴露 set-priority(改当前 attention 优先级).
    :param enable_bar: 在 attention 子通道暴露 set-signal-bar / set-impulse-bar(全局水位).
    :param enable_pull: 顶层暴露 pull(从 nucleus 主动拉取 impulse).
    :param gate: 开启后声明的虚拟子通道(attention + nucleus channel)默认关闭, 由 mount_child 披露.
    """
    channel = new_prime_channel(name, description=mindflow.description(), gate=gate)
    attention_child = _build_attention_child(mindflow, enable_priority, enable_bar)

    # --- 静态心智模型 (instruction): 绝不重复罗列命令签名 --- #
    @channel.build.instruction
    def instruction() -> str:
        return (
            "## mindflow channel\n"
            "This is the reflexive surface of your own mind — the control plane over "
            "your parallel sensing and thinking units (nuclei). You are not merely a "
            "passive receiver of impulses: you can see what your sensory units currently "
            "hold and deliberately steer your own attention.\n"
            "Read before you act on your own perception. `status` inspects the current "
            "attention and what each nucleus holds; `pull` reaches into a nucleus for its "
            "top impulse. Deliberate attention governance lives in the `attention` child. "
            "Do not poll every frame; react to signals, not to your own relay."
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
        if enable_pull:
            for name, nucleus in mindflow.nuclei().items():
                impulse = nucleus.peek()
                if impulse is not None:
                    blocks.append(f"pullable {name}: {_impulse_text(impulse)}")
        if not blocks:
            return []
        return [Message.new().with_content("\n".join(blocks))]

    # --- virtual_children: attention 治理 + 运行中 nucleus 的子通道 --- #
    @channel.build.virtual_children
    def virtual_children() -> dict[str, Channel]:
        channels: dict[str, Channel] = {}
        if attention_child is not None:
            channels[attention_child.name()] = attention_child
        for key, nucleus in mindflow.nuclei().items():
            if nucleus.is_running():
                if chan := nucleus.as_channel():
                    channels[key] = chan
        return channels

    # --- 命令面 (常驻读面) --- #

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
