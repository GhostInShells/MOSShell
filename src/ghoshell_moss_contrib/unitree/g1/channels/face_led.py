"""
Face LED — G1 面部 LED 灯条 channel (L4).

物理位置: G1 面部中线横向灯条 (眼睛区域). 全 G1 唯一一组 RGB LED, 是这个身体
唯一的视觉表达通道.

命令分三层:
- **idle 修改** (solid / breath / set_idle): 改"默认灯效", 无命令时 LED 显示的样子.
  立刻生效, 也是后续 idle 时的回落状态.
- **表现动画** (fade / blink / pulse / rainbow / police): animation 轨表演, blocking
  到动画播完 → 保 CTML 时序. 播完自动让位 idle 底色.
- **清理** (clear / off): clear 停当前表现回 idle; off 连 idle 一起灭.

context_messages 每次 refresh 上报当前 idle 元信息, 让 LLM 感知自己的"表情底色".

event 轨预留系统用 (未来 mindflow signal / 异常报警等), 不暴露给 LLM.

runtime 依赖: `ghoshell_moss_contrib.unitree.g1.runtime.led`. led runtime 由本
channel 的 startup 自动启动 (face_led 自包含). 上游需保证 sdk 已 bootstrap
(mode 加载路径 / 集成层承担 — sdk 是整机级依赖, 不属单个 channel).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from ghoshell_moss.core.blueprint.channel_builder import (
    new_channel, CommandUtil, Observe,
)
from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1.runtime import led
from ghoshell_moss_contrib.unitree.g1.channels._utils import (
    check_g1_available, check_channel_information,
)

logger = logging.getLogger("moss.g1.channels.face_led")

__all__ = ["face_led"]


# ═══════════════════════════════════════════════════════════════════════════════
# idle 状态 — 模块级单一真相, idle 生命周期与 context_messages 共享
# ═══════════════════════════════════════════════════════════════════════════════

# 存"人可读的元信息"而不是 Animation 对象: 反推 style/color/period 麻烦, 且
# context_messages 要序列化给 LLM 时元信息更直观. Animation 由 to_animation() 派生.
@dataclass
class _IdleState:
    style: Literal["breath", "solid", "off"] = "breath"
    color: str = "dim"          # 微暗中性色, 不吸引注意
    period_ms: int = 3000       # breath 时用, 略慢让呼吸不焦虑

    def to_animation(self) -> led.Animation | None:
        if self.style == "off":
            return None
        if self.style == "solid":
            return led.solid(self.color)
        return led.breath(self.color, period_ms=self.period_ms)


_idle_state = _IdleState()


def _apply_idle_state() -> None:
    """把 _idle_state 落到 background 轨. off 时 clear_background."""
    anim = _idle_state.to_animation()
    if anim is None:
        led.clear_background()
    else:
        led.set_background(anim)


# ═══════════════════════════════════════════════════════════════════════════════
# instruction — 一次生成的静态 prompt
# ═══════════════════════════════════════════════════════════════════════════════

# 命名色按色系分组呈现, 比字母序更利于 LLM 挑色. runtime 里 `black` 与 `off`
# 是同色别名 (Color(0,0,0)), 呈现层只留 `off` (跟命令 off() 同名, 心智模型统一).
# 断言防未来 runtime 加新色时呈现漏同步.
_PRESENTED_COLOR_GROUPS: list[list[str]] = [
    ["red", "green", "blue", "yellow", "cyan", "magenta"],  # 主色
    ["orange", "purple", "pink"],                            # 混色
    ["white", "warm", "dim"],                                # 亮度
    ["off"],                                                 # 灭
]
_RUNTIME_ALIASED_COLORS = {"black"}  # runtime 保留但呈现层不吐给 LLM 的别名
assert (
    {c for g in _PRESENTED_COLOR_GROUPS for c in g} | _RUNTIME_ALIASED_COLORS
    == set(led.named_colors())
), (
    f"face_led 命名色呈现漏同步 runtime: "
    f"presented+aliased={ {c for g in _PRESENTED_COLOR_GROUPS for c in g} | _RUNTIME_ALIASED_COLORS }, "
    f"runtime={set(led.named_colors())}"
)
_NAMED_COLORS_STR = "  |  ".join(" / ".join(g) for g in _PRESENTED_COLOR_GROUPS)

_INSTRUCTION = f"""\
你的面部有一条横向 LED 灯条, 位于眼睛区域. 这是你身体唯一的视觉表达通道.

它有两层表达:

- 默认灯效 (idle): 你无命令时 LED 显示的样子, 是你的"表情底色".
  通过 solid / breath / set_idle 修改. 修改后立刻生效, 且长期生效.
- 表现动画: 有始有终的一段表演. fade / blink / pulse / rainbow / police.
  播放期间盖住 idle, 播完自动回到 idle.

如果你想"停下表演回默认", 用 clear.
如果你想"完全灭灯不显任何东西", 用 off (会把 idle 也改成灭灯, 恢复需再调 idle 类命令).

颜色接受两种格式:
- hex: #RRGGBB (如 #ff0000 表示纯红)
- 命名色: {_NAMED_COLORS_STR}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# channel 组装
# ═══════════════════════════════════════════════════════════════════════════════

face_led = new_channel(
    name="face_led",
    description="G1 面部 LED 灯条 — 唯一视觉表达通道. 支持表现动画与默认底色两层.",
)
face_led.build.instruction(_INSTRUCTION)


# ─── context_messages: 每次 refresh 上报当前 idle 元信息 ────────────────────

# 用 attributes 序列化元信息, 让 LLM 一眼看清结构; content 附一句人话补充语义.
# tag 命名 face_led_idle 而不是 face_led — 避免跟 channel 自身 static messages 冲突,
# 也让 LLM 立即分辨这是 "idle 状态感知" 消息.
@face_led.build.context_messages
async def _idle_context() -> list[Message]:
    style = _idle_state.style
    attrs: dict = {"style": style}
    if style != "off":
        attrs["color"] = _idle_state.color
    if style == "breath":
        attrs["period_ms"] = _idle_state.period_ms

    if style == "off":
        note = "面部 LED 当前无 idle 底色 (灭灯). 想恢复表达需重设 idle."
    elif style == "solid":
        note = f"面部 LED 默认: {_idle_state.color} 常亮."
    else:
        note = f"面部 LED 默认: {_idle_state.color} 呼吸, 周期 {_idle_state.period_ms}ms."

    return [Message.new(tag="face_led_idle", attributes=attrs).with_content(note)]


# ─── startup: 启动 led runtime driver 线程. 让 face_led 自包含 ─────────────
# channel 直接依赖的 runtime 由 channel 自己启动. 幂等: led.start() 内部自检
# 重入直接 return. sdk 是整机级依赖, 假设加载路径 (mode / 集成层) 已 bootstrap;
# 未 bootstrap 时 led.start() 内的 get_audio_client() 会立刻 raise 露出清晰错误.
@face_led.build.startup
async def _on_startup() -> None:
    try:
        if not led.is_running():
            led.start()
            logger.info("face_led startup: led runtime started.")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("face_led startup 异常 — led runtime 未起, 后续命令会挂.")
        raise


# ─── idle: 落 background 轨. 承担 channel 启动 + off/clear 后的全部回填 ───────
# 触发时机: channel 启动瞬间就是"无命令输入", 立刻走 idle; LLM 用 off/clear
# 后没继续下命令, 再次走 idle. 幂等: set_background 反复调无副作用.
@face_led.build.idle
async def _on_idle() -> None:
    try:
        _apply_idle_state()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("face_led on_idle 异常 (保持活).")


# ═══════════════════════════════════════════════════════════════════════════════
# idle 类命令 — 改默认底色, 立刻生效
# blocking=False: idle 命令是"设置", 不占 channel 时序. LLM 可以边说话边改.
# ═══════════════════════════════════════════════════════════════════════════════

@face_led.build.command(blocking=False)
async def solid(color: str) -> None:
    """把默认灯效改成 <color> 常亮.

    立刻生效, 且成为你无命令时的表情底色. 想恢复呼吸用 breath, 想灭用 off.

    color: hex '#RRGGBB' 或命名色 (见 instruction).
    """
    check_g1_available()
    try:
        # 先试解析, 失败快速 raise, 避免落库到 _idle_state 后再挂
        led.parse_color(color)
    except ValueError as e:
        CommandUtil.raise_observe(f"color 参数无效: {e}")
    _idle_state.style = "solid"
    _idle_state.color = color
    _apply_idle_state()


@face_led.build.command(blocking=False)
async def breath(color: str, period_ms: int = 2000) -> None:
    """把默认灯效改成 <color> 呼吸.

    立刻生效, 且成为你无命令时的表情底色.

    color: hex '#RRGGBB' 或命名色.
    period_ms: 一次呼吸周期 (ms), 建议 1500-4000.
    """
    check_g1_available()
    if period_ms <= 0:
        CommandUtil.raise_observe(f"period_ms 必须 > 0, 给的是 {period_ms}")
    try:
        led.parse_color(color)
    except ValueError as e:
        CommandUtil.raise_observe(f"color 参数无效: {e}")
    _idle_state.style = "breath"
    _idle_state.color = color
    _idle_state.period_ms = period_ms
    _apply_idle_state()


@face_led.build.command(blocking=False)
async def set_idle(
    color: str = "dim",
    style: Literal["breath", "solid", "off"] = "breath",
    period_ms: int = 3000,
) -> None:
    """通用 idle 设置. 一般用 solid / breath 就够, 只有需要 off (灭 idle) 才用它.

    style: breath (呼吸) / solid (常亮) / off (灭灯, 忽略 color).
    color: hex 或命名色, style=off 时忽略.
    period_ms: breath 时的呼吸周期.
    """
    check_g1_available()
    if style not in ("breath", "solid", "off"):
        CommandUtil.raise_observe(f"style 必须是 breath/solid/off, 给的是 {style}")
    if style == "breath" and period_ms <= 0:
        CommandUtil.raise_observe(f"period_ms 必须 > 0, 给的是 {period_ms}")
    if style != "off":
        try:
            led.parse_color(color)
        except ValueError as e:
            CommandUtil.raise_observe(f"color 参数无效: {e}")
    _idle_state.style = style
    _idle_state.color = color
    _idle_state.period_ms = period_ms
    _apply_idle_state()


# ═══════════════════════════════════════════════════════════════════════════════
# 表现动画 — animation 轨, blocking + await sleep 到播完保 CTML 时序
# 播完自动清 animation 轨 → background (idle) 显形恢复.
# cancel 时立刻 clear_animation, 避免动画残留.
# ═══════════════════════════════════════════════════════════════════════════════

# 表现类命令限制 count >= 1 (不允许 -1 永久), 避免 LLM 用表现命令表达"持续状态".
# 持续状态是 idle 类命令的语义, 命令面强制这条纪律.
_COUNT_MIN = 1
_COUNT_MAX = 20


def _check_count(count: int) -> None:
    if not (_COUNT_MIN <= count <= _COUNT_MAX):
        CommandUtil.raise_observe(
            f"count 必须在 [{_COUNT_MIN}, {_COUNT_MAX}], 给的是 {count}. "
            f"想要持续状态请改 idle (solid/breath), 表现动画应有始有终."
        )


@face_led.build.command()
async def fade(
    from_color: str,
    to_color: str,
    duration_ms: int = 1000,
    easing: Literal["linear", "smooth", "step", "hsv"] = "linear",
) -> None:
    """从 <from_color> 平滑渐变到 <to_color>, 播一次, 阻塞到播完.

    from_color / to_color: hex 或命名色.
    duration_ms: 渐变总时长 (ms).
    easing: linear (匀速) / smooth (慢起慢收) / step (阶跃) / hsv (色环最短弧, 适合彩色过渡).
    """
    check_g1_available()
    if duration_ms <= 0:
        CommandUtil.raise_observe(f"duration_ms 必须 > 0, 给的是 {duration_ms}")
    try:
        anim = led.fade(from_color, to_color, duration_ms=duration_ms, easing=led.Easing(easing))
    except (ValueError, KeyError) as e:
        CommandUtil.raise_observe(f"参数无效: {e}")
        return  # unreachable, satisfy type checker

    led.play_animation(anim)
    try:
        await asyncio.sleep(duration_ms / 1000)
    except asyncio.CancelledError:
        led.clear_animation()
        raise


@face_led.build.command()
async def blink(color: str, count: int = 3, period_ms: int = 500) -> None:
    """闪烁 <color> 灯 <count> 次, 阻塞到播完.

    用于反馈"收到 / 完成 / 出错"这类瞬间信号.

    color: hex 或命名色.
    count: 闪烁次数 (1-20).
    period_ms: 一次亮灭的总时长 (ms).
    """
    check_g1_available()
    _check_count(count)
    if period_ms <= 0:
        CommandUtil.raise_observe(f"period_ms 必须 > 0, 给的是 {period_ms}")
    try:
        anim = led.blink(color, count=count, period_ms=period_ms)
    except ValueError as e:
        CommandUtil.raise_observe(f"参数无效: {e}")
        return

    led.play_animation(anim)
    try:
        await asyncio.sleep(count * period_ms / 1000)
    except asyncio.CancelledError:
        led.clear_animation()
        raise


@face_led.build.command()
async def pulse(color: str, count: int = 1, period_ms: int = 800) -> None:
    """脉冲: 灭→亮→灭 平滑一次, 播 <count> 遍, 阻塞到播完.

    比 blink 更柔和, 用于"轻收到 / 温和确认".

    color: hex 或命名色.
    count: 脉冲次数 (1-20).
    period_ms: 一次脉冲的时长 (ms).
    """
    check_g1_available()
    _check_count(count)
    if period_ms <= 0:
        CommandUtil.raise_observe(f"period_ms 必须 > 0, 给的是 {period_ms}")
    try:
        anim = led.pulse(color, count=count, period_ms=period_ms)
    except ValueError as e:
        CommandUtil.raise_observe(f"参数无效: {e}")
        return

    led.play_animation(anim)
    try:
        await asyncio.sleep(count * period_ms / 1000)
    except asyncio.CancelledError:
        led.clear_animation()
        raise


@face_led.build.command()
async def rainbow(period_ms: int = 4000, count: int = 1) -> None:
    """彩虹环 (HSV 全色环) 循环 <count> 次, 阻塞到播完.

    用于表达强烈情绪 / 秀一下. 想做持续状态请改 idle.

    period_ms: 一次色环周期 (ms).
    count: 循环次数 (1-20).
    """
    check_g1_available()
    _check_count(count)
    if period_ms <= 0:
        CommandUtil.raise_observe(f"period_ms 必须 > 0, 给的是 {period_ms}")

    led.play_animation(led.rainbow(period_ms=period_ms, count=count))
    try:
        await asyncio.sleep(count * period_ms / 1000)
    except asyncio.CancelledError:
        led.clear_animation()
        raise


@face_led.build.command()
async def police(period_ms: int = 600, count: int = 5) -> None:
    """红蓝交替警示 <count> 次, 阻塞到播完.

    用于警戒 / 紧急场景.

    period_ms: 一次红蓝切换的总时长 (ms).
    count: 交替次数 (1-20).
    """
    check_g1_available()
    _check_count(count)
    if period_ms <= 0:
        CommandUtil.raise_observe(f"period_ms 必须 > 0, 给的是 {period_ms}")

    led.play_animation(led.police(period_ms=period_ms, count=count))
    try:
        await asyncio.sleep(count * period_ms / 1000)
    except asyncio.CancelledError:
        led.clear_animation()
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# 清理命令
# ═══════════════════════════════════════════════════════════════════════════════

@face_led.build.command(blocking=False)
async def clear() -> None:
    """立刻停止当前表现动画, 回到默认灯效 (idle).

    不修改 idle 设置本身, 只是停当前表演.
    """
    check_g1_available()
    led.clear_animation()


@face_led.build.command(blocking=False)
async def off() -> None:
    """完全灭灯 — 停止表现 + idle 改为灭灯. 恢复表达需再调 idle 类命令.

    等价于: clear() + set_idle(style='off').
    """
    check_g1_available()
    _idle_state.style = "off"
    led.clear_all()


if __name__ == "__main__":
    check_channel_information(face_led)
