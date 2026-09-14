"""Mindflow reflexive control channel — 反身控制面.

把 mindflow 从 opaque 调度器变成 ghost 可感知、可操纵的透明面.

常驻能力面直接挂在 mindflow 父 channel 上, 不折叠: 自解释 + 注意力治理 + 自省.
注意力治理是这里的一等能力 —— 运行时提升/降低当前 attention 的优先级、抬高全局水位.
它必须常驻可见: 一旦折叠进 gate, 模型不会主动 mount, 能力等于不存在.

gate 只折叠各 nucleus 的子通道 —— 那是"按需展开的细节", 不是 mindflow 自身的控制面.

治理状态 (当前 attention / 水位) 是状态级变更 (温数据), 走 notice, 不占每帧的
context_messages (热面). nucleus 讯息走 ``nuclei`` 方法, 不散在 notice/status 里重复罗列.

这是 core 内部面, 经 ``Mindflow.as_channel()`` 挂进 shell, 而非随包分发的
app channel. 反身控制原则: 刻意自省, 不做每帧轮询.
"""

from __future__ import annotations

from ghoshell_moss.core.blueprint.mindflow import Mindflow, Priority
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel, PrimeChannel
from ghoshell_moss.core.concepts.channel import Channel

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
        gate: bool = True,
) -> PrimeChannel:
    """构建 mindflow 反身控制 channel.

    常驻能力面 (注意力治理 + 自省) 全部挂在父 channel 上, 不进 gate ——
    这些是 mindflow 自身的控制面, 折叠后模型不会用到. gate 只折叠各 nucleus 的子通道.

    :param enable_priority: 暴露 set-priority (改当前 attention 优先级).
    :param enable_bar: 暴露 set-signal-bar / set-impulse-bar (全局水位).
    :param gate: 开启后各 nucleus 的子通道默认关闭, 由 mount_child 披露.
    """
    channel = new_prime_channel(name, description=mindflow.description(), gate=gate)

    # --- 静态心智模型 (instruction): 绝不重复罗列命令签名 --- #
    @channel.build.instruction
    def instruction() -> str:
        return (
            "## mindflow channel\n"
            "This is the reflexive surface of your own mind — the control plane over "
            "your parallel sensing and thinking units (nuclei). You are not merely a "
            "passive receiver of impulses: you can see what your sensory units currently "
            "hold and deliberately steer your own attention.\n"
            "Raise the current attention's priority to hold your focus; lower it to yield. "
            "`nuclei` shows what each unit holds, `status` inspects the current attention. "
            "Do not poll every frame; react to signals, not to your own relay."
        )

    # --- notice: 温数据 (状态级变更), 不占每帧 context --- #
    @channel.build.notice
    def notice() -> str:
        lines: list[str] = []
        if enable_priority:
            if attn := mindflow.attention():
                lines.append(f"active attention: {_attention_line(attn)}")
        if enable_bar:
            lines.append(f"signal bar: {mindflow.signal_priority_bar().name}")
            lines.append(f"impulse bar: {mindflow.impulse_priority_bar().name}")
        return "\n".join(lines)

    # --- 注意力治理 (常驻一等能力) --- #

    @channel.build.command(name="set-priority", available=lambda: enable_priority)
    async def set_priority(priority: str) -> str:
        """Raise or lower the priority of the current attention (raise to hold, lower to yield).

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

    # --- 自省面 --- #

    @channel.build.command(name="status", always_observe=True)
    async def status() -> str:
        """Inspect the current attention (deliberate self-introspection)."""
        if attn := mindflow.attention():
            return f"active attention: {_attention_line(attn)}"
        return "no active attention"

    @channel.build.command(name="nuclei")
    async def nuclei() -> str:
        """List your sensing units (nuclei): name, running state, description, top impulse.

        Use this to see what your sensory units currently hold before tuning the floors.
        """
        lines = ["mindflow nuclei:"]
        for nucleus_name, nucleus in mindflow.nuclei().items():
            state = "running" if nucleus.is_running() else "idle"
            line = f"  {nucleus_name} ({state}): {nucleus.description()}"
            if impulse := nucleus.peek():
                line += f" | peek: {_impulse_text(impulse)}"
            lines.append(line)
        return "\n".join(lines)

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
