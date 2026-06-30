"""
LED runtime — G1 眼条 LED 的关键帧动画 + 优先级轨道.

物理事实 (来自 scripts/sdk/08_audio_led.py 实测):
  - G1 仅一组 RGB LED, 物理位置在面部"眼睛"处的横向灯条.
  - 接口: AudioClient.LedControl(r, g, b), r/g/b ∈ [0, 255].
  - 单次调用 = 单帧颜色. 无原生插值, 无渐变, 无亮度独立通道.
  - 所有"渐变/呼吸"由本模块在驱动线程内按 fps 实时插值后逐帧下发.

设计三段 (见 runtime/README.md 通用纪律):
  - 数据模型 — Color / Easing / Keyframe / Clip / Animation, pydantic, Field
    description 直接进 LLM prompt (channel 拿 model_json_schema 拼).
  - 工厂函数 — solid/blink/breath/fade/pulse/police/rainbow/sequence, 都返回 Animation.
  - 三轨道驱动 — event(高) / animation(中) / background(低), daemon 线程每 tick 取
    最高优先级且未播完的轨道, 渲染当前色, 与上次下发不同才调 sdk.

轨道语义:
  event       播完即弃, 抢占其他轨 — 用于告警/确认反馈
  animation   播完即弃, 模型主导 — 用于表达性动画
  background  常驻循环, 被前两轨临时压制 — 用于状态色/呼吸/情绪

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import led

    bootstrap(nic)
    led.start()                                              # 启动驱动线程
    led.set_background(led.breath("#0066ff", period_ms=3000)) # 底色: 蓝色呼吸
    led.play_event(led.blink("#ff0000", count=3, period_ms=200))  # 抢占: 红闪三次
    # ... event 播完, 自动让位 background, 看到蓝色呼吸恢复 ...
    led.stop()                                               # 关灯 + join 线程
"""
from __future__ import annotations

import colorsys
import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field

from ghoshell_moss_contrib.unitree.g1.sdk import get_audio_client

logger = logging.getLogger("moss.g1.runtime.led")

__all__ = [
    # 数据模型
    "Color", "Easing", "Keyframe", "Clip", "Animation",
    # 工厂
    "solid", "off", "blink", "breath", "fade", "pulse", "police", "rainbow", "sequence",
    "parse_color",
    # 生命周期
    "start", "stop", "is_running",
    # 轨道
    "set_background", "play_animation", "play_event",
    "clear_event", "clear_animation", "clear_background", "clear_all",
    # 基线 + 调试
    "set_led", "health", "animation_schema",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 数据契约 — 字段 description 直接进 LLM prompt (channel 通过 model_json_schema 喂)
# ═══════════════════════════════════════════════════════════════════════════════

class Color(BaseModel):
    """RGB 颜色. G1 仅一组 LED, 整条灯条共用一个 (r, g, b)."""
    r: int = Field(ge=0, le=255, description="红色分量 0-255.")
    g: int = Field(ge=0, le=255, description="绿色分量 0-255.")
    b: int = Field(ge=0, le=255, description="蓝色分量 0-255.")

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class Easing(str, Enum):
    """关键帧之间的插值模式."""
    STEP = "step"        # 阶跃: 不插值, 保持当前关键帧颜色直到下一关键帧时刻才切换.
    LINEAR = "linear"    # 线性: RGB 三通道各自独立 lerp. 默认.
    SMOOTH = "smooth"    # 慢起慢收 (ease-in-out, sine 曲线). 用于呼吸/脉冲.
    HSV = "hsv"          # HSV 色环最短路径插值. 用于彩虹过渡, RGB lerp 会发灰.


class Keyframe(BaseModel):
    """单个关键帧 — 在 clip 内的归一化时刻 (t) 锁定一个颜色."""
    t: float = Field(
        ge=0.0, le=1.0,
        description="此关键帧在 clip 内的归一化时刻, 0=clip 开始, 1=clip 结束. 同 clip 内 t 递增.",
    )
    color: Color = Field(description="此时刻的目标颜色.")


class Clip(BaseModel):
    """动画片段 — 共享一种插值模式的连续关键帧序列, 有自己的时长."""
    keyframes: list[Keyframe] = Field(
        min_length=1,
        description="至少 1 个关键帧 (单关键帧 = 整段保持该色); 多关键帧按 t 递增.",
    )
    duration_ms: int = Field(
        gt=0,
        description="本 clip 总播放时长 (毫秒). t∈[0,1] 映射到 [0, duration_ms].",
    )
    easing: Easing = Field(
        default=Easing.LINEAR,
        description="本 clip 内关键帧之间的插值模式.",
    )


class Animation(BaseModel):
    """完整动画 — 多个 clip 顺序拼接, 整体可循环."""
    clips: list[Clip] = Field(
        min_length=1,
        description="顺序播放的 clip 列表. 一个循环周期 = 所有 clip 时长之和.",
    )
    loop: int = Field(
        default=1,
        description="循环次数. 1=播一次后停; N=播 N 次; -1=永久循环 (背景动画必用 -1).",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 颜色解析 — hex / 命名色 / dict → Color
# ═══════════════════════════════════════════════════════════════════════════════

_NAMED_COLORS: dict[str, Color] = {
    "off":      Color(r=0, g=0, b=0),
    "black":    Color(r=0, g=0, b=0),
    "white":    Color(r=255, g=255, b=255),
    "red":      Color(r=255, g=0, b=0),
    "green":    Color(r=0, g=255, b=0),
    "blue":     Color(r=0, g=0, b=255),
    "yellow":   Color(r=255, g=255, b=0),
    "cyan":     Color(r=0, g=255, b=255),
    "magenta":  Color(r=255, g=0, b=255),
    "orange":   Color(r=255, g=128, b=0),
    "purple":   Color(r=128, g=0, b=255),
    "pink":     Color(r=255, g=128, b=192),
    "warm":     Color(r=255, g=180, b=120),  # 暖白
    "dim":      Color(r=32, g=32, b=32),     # 微亮中性色
}

ColorLike = Union[Color, str, dict]


def parse_color(value: ColorLike) -> Color:
    """接受 Color / hex str ('#ff0000' / '#f00') / 命名色 ('red' / 'off' / ...) / dict.

    Raises ValueError 输入无法解析.
    """
    if isinstance(value, Color):
        return value
    if isinstance(value, dict):
        return Color.model_validate(value)
    if not isinstance(value, str):
        raise ValueError(f"无法解析颜色: {value!r} (期望 Color / hex str / 命名色 / dict)")
    s = value.strip().lower()
    if s in _NAMED_COLORS:
        return _NAMED_COLORS[s]
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6 and all(c in "0123456789abcdef" for c in s):
        return Color(r=int(s[0:2], 16), g=int(s[2:4], 16), b=int(s[4:6], 16))
    if len(s) == 3 and all(c in "0123456789abcdef" for c in s):
        return Color(r=int(s[0]*2, 16), g=int(s[1]*2, 16), b=int(s[2]*2, 16))
    raise ValueError(
        f"无法解析颜色: {value!r}. 接受 #RRGGBB / #RGB / 命名色: "
        f"{sorted(_NAMED_COLORS.keys())}"
    )


def named_colors() -> list[str]:
    """已注册命名色清单. channel 层可拼进 docstring 引导模型."""
    return sorted(_NAMED_COLORS.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# 插值与渲染
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp_u8(x: float) -> int:
    return max(0, min(255, int(round(x))))


def _apply_easing(easing: Easing, t: float) -> float:
    """把 raw t∈[0,1] 映射到 eased t'∈[0,1]. STEP 总返回 0 (保持 kf0 直到段末)."""
    if easing == Easing.STEP:
        return 0.0
    if easing == Easing.SMOOTH:
        # ease-in-out, sine 曲线
        return 0.5 - 0.5 * math.cos(math.pi * t)
    # LINEAR / HSV — 后者的"环空间"切换在 _interp_color, 不在 easing
    return t


def _lerp_rgb(c0: Color, c1: Color, t: float) -> Color:
    return Color(
        r=_clamp_u8(c0.r + (c1.r - c0.r) * t),
        g=_clamp_u8(c0.g + (c1.g - c0.g) * t),
        b=_clamp_u8(c0.b + (c1.b - c0.b) * t),
    )


def _lerp_hsv(c0: Color, c1: Color, t: float) -> Color:
    """HSV 空间 lerp, hue 走色环最短路径."""
    h0, s0, v0 = colorsys.rgb_to_hsv(c0.r/255.0, c0.g/255.0, c0.b/255.0)
    h1, s1, v1 = colorsys.rgb_to_hsv(c1.r/255.0, c1.g/255.0, c1.b/255.0)
    # 色环最短弧
    dh = h1 - h0
    if dh > 0.5:
        dh -= 1.0
    elif dh < -0.5:
        dh += 1.0
    h = (h0 + dh * t) % 1.0
    s = s0 + (s1 - s0) * t
    v = v0 + (v1 - v0) * t
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return Color(r=_clamp_u8(r*255), g=_clamp_u8(g*255), b=_clamp_u8(b*255))


def _interp_color(easing: Easing, c0: Color, c1: Color, t: float) -> Color:
    if easing == Easing.HSV:
        return _lerp_hsv(c0, c1, t)
    return _lerp_rgb(c0, c1, t)


def _render_clip(clip: Clip, t: float) -> Color:
    """t ∈ [0, 1] within the clip. 找到 t 所在的关键帧段, 应用插值."""
    kfs = clip.keyframes
    if len(kfs) == 1 or t <= kfs[0].t:
        return kfs[0].color
    if t >= kfs[-1].t:
        return kfs[-1].color
    for i in range(len(kfs) - 1):
        kf0, kf1 = kfs[i], kfs[i + 1]
        if kf0.t <= t <= kf1.t:
            span = kf1.t - kf0.t
            if span <= 0:
                return kf1.color
            local = (t - kf0.t) / span
            eased = _apply_easing(clip.easing, local)
            return _interp_color(clip.easing, kf0.color, kf1.color, eased)
    return kfs[-1].color  # 不应到达, 保险


def _render_animation(anim: Animation, elapsed_ms: int) -> Optional[Color]:
    """渲染 anim 在 elapsed_ms 时刻的颜色. 返回 None 表示已播完 (有限 loop 已耗尽)."""
    cycle_ms = sum(c.duration_ms for c in anim.clips)
    if cycle_ms <= 0:
        return anim.clips[0].keyframes[0].color

    if anim.loop > 0:
        total = cycle_ms * anim.loop
        if elapsed_ms >= total:
            return None

    t = elapsed_ms % cycle_ms
    for clip in anim.clips:
        if t < clip.duration_ms:
            local = t / clip.duration_ms
            return _render_clip(clip, local)
        t -= clip.duration_ms
    # 边界 (浮点取末尾色)
    return anim.clips[-1].keyframes[-1].color


# ═══════════════════════════════════════════════════════════════════════════════
# 工厂函数 — 构建常用 Animation
# ═══════════════════════════════════════════════════════════════════════════════

_BLACK = Color(r=0, g=0, b=0)


def solid(color: ColorLike) -> Animation:
    """单色常亮 (loop=-1, 永久)."""
    c = parse_color(color)
    return Animation(
        clips=[Clip(keyframes=[Keyframe(t=0.0, color=c)], duration_ms=1000, easing=Easing.STEP)],
        loop=-1,
    )


def off() -> Animation:
    """灭灯."""
    return solid("off")


def blink(color: ColorLike, count: int = 3, period_ms: int = 500) -> Animation:
    """闪烁: 亮-灭交替 count 次, 每次周期 period_ms (一半亮 + 一半灭)."""
    c = parse_color(color)
    half = max(period_ms // 2, 20)
    return Animation(
        clips=[
            Clip(keyframes=[Keyframe(t=0.0, color=c)],      duration_ms=half, easing=Easing.STEP),
            Clip(keyframes=[Keyframe(t=0.0, color=_BLACK)], duration_ms=half, easing=Easing.STEP),
        ],
        loop=count,
    )


def breath(color: ColorLike, period_ms: int = 2000, count: int = -1) -> Animation:
    """呼吸: 暗→亮→暗 平滑循环, 默认永久. 最暗保留 1/16 余晖让节奏感更自然."""
    c = parse_color(color)
    dim = Color(r=c.r // 16, g=c.g // 16, b=c.b // 16)
    return Animation(
        clips=[Clip(
            keyframes=[
                Keyframe(t=0.0, color=dim),
                Keyframe(t=0.5, color=c),
                Keyframe(t=1.0, color=dim),
            ],
            duration_ms=period_ms,
            easing=Easing.SMOOTH,
        )],
        loop=count,
    )


def fade(
    from_color: ColorLike,
    to_color: ColorLike,
    duration_ms: int = 1000,
    easing: Easing = Easing.LINEAR,
) -> Animation:
    """从 A 渐变到 B, 播一次. easing 可选 linear/smooth/hsv."""
    c0 = parse_color(from_color)
    c1 = parse_color(to_color)
    return Animation(
        clips=[Clip(
            keyframes=[Keyframe(t=0.0, color=c0), Keyframe(t=1.0, color=c1)],
            duration_ms=duration_ms,
            easing=easing,
        )],
        loop=1,
    )


def pulse(color: ColorLike, count: int = 1, period_ms: int = 800) -> Animation:
    """脉冲: 灭→亮→灭 smooth, 用于'收到/确认'反馈. 默认播一次."""
    c = parse_color(color)
    return Animation(
        clips=[Clip(
            keyframes=[
                Keyframe(t=0.0, color=_BLACK),
                Keyframe(t=0.35, color=c),
                Keyframe(t=1.0, color=_BLACK),
            ],
            duration_ms=period_ms,
            easing=Easing.SMOOTH,
        )],
        loop=count,
    )


def police(period_ms: int = 600, count: int = -1) -> Animation:
    """红蓝交替警示, 默认永久 (用 event 轨配合, count 有限场景)."""
    half = max(period_ms // 2, 20)
    return Animation(
        clips=[
            Clip(keyframes=[Keyframe(t=0.0, color=Color(r=255, g=0, b=0))],
                 duration_ms=half, easing=Easing.STEP),
            Clip(keyframes=[Keyframe(t=0.0, color=Color(r=0, g=0, b=255))],
                 duration_ms=half, easing=Easing.STEP),
        ],
        loop=count,
    )


def rainbow(period_ms: int = 4000, count: int = -1) -> Animation:
    """HSV 全色环循环 — 关键帧已经是色环六分点, 用 LINEAR easing 即可."""
    return Animation(
        clips=[Clip(
            keyframes=[
                Keyframe(t=0.0 / 6, color=Color(r=255, g=0, b=0)),
                Keyframe(t=1.0 / 6, color=Color(r=255, g=255, b=0)),
                Keyframe(t=2.0 / 6, color=Color(r=0, g=255, b=0)),
                Keyframe(t=3.0 / 6, color=Color(r=0, g=255, b=255)),
                Keyframe(t=4.0 / 6, color=Color(r=0, g=0, b=255)),
                Keyframe(t=5.0 / 6, color=Color(r=255, g=0, b=255)),
                Keyframe(t=1.0,     color=Color(r=255, g=0, b=0)),
            ],
            duration_ms=period_ms,
            easing=Easing.LINEAR,
        )],
        loop=count,
    )


def sequence(*anims: Animation) -> Animation:
    """把多个 anim 拼接顺序播一遍.

    注意: 子 anim 自带的 loop 信息被吞 — 每个子 anim 只播一次. 想保留循环效果,
    手动复制 clips 或用更复杂的关键帧组合.
    """
    if not anims:
        raise ValueError("sequence 至少需要 1 个 anim")
    clips: list[Clip] = []
    for a in anims:
        clips.extend(a.clips)
    return Animation(clips=clips, loop=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级状态
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Track:
    animation: Optional[Animation] = None
    started_at: float = 0.0  # time.monotonic() 时间戳, animation 装载时记一次


_TRACK_NAMES = ("event", "animation", "background")  # 高 → 低 优先级

_state_lock = threading.Lock()
_tracks: dict[str, _Track] = {name: _Track() for name in _TRACK_NAMES}

_running: bool = False
_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_drive_fps: int = 20

_last_sent_color: Optional[Color] = None
_sdk_call_count: int = 0
_error_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 公开接口 — 生命周期
# ═══════════════════════════════════════════════════════════════════════════════

def start(*, drive_fps: int = 20) -> None:
    """启动 LED 驱动线程. 幂等.

    前置: sdk.bootstrap() 已完成 (否则 get_audio_client 立刻 raise).

    drive_fps 决定动画视觉平滑度. 20Hz (默认) 对眼条 LED fade/breath 足够;
    实测 LedControl 调用率上限后可上调到 30-50, 太高可能触发 RPC timeout.

    :param drive_fps: 驱动线程 tick 频率, 范围 [1, 100].
    """
    global _running, _thread, _stop_event, _drive_fps, _last_sent_color
    global _sdk_call_count, _error_count

    if not 1 <= drive_fps <= 100:
        raise ValueError(f"drive_fps 应在 [1, 100], 给的是 {drive_fps}")

    with _state_lock:
        if _running:
            logger.debug("start() 重入 — 已运行, 跳过.")
            return
        # 触发一次 client 检查 (未 bootstrap 立刻 raise, 不要让驱动线程才发现)
        get_audio_client()
        _drive_fps = drive_fps
        _last_sent_color = None
        _sdk_call_count = 0
        _error_count = 0
        _stop_event = threading.Event()
        _running = True
        _thread = threading.Thread(target=_drive_loop, name="g1-led-drive", daemon=True)
        _thread.start()
        logger.info("LED runtime started (fps=%d).", drive_fps)


def stop(timeout: float = 2.0) -> None:
    """停止驱动线程, 灭灯, 清空所有轨道. 幂等.

    daemon 线程随进程退出而死, 这里 join 是干净退出兜底.
    """
    global _running, _thread, _stop_event

    with _state_lock:
        if not _running:
            logger.debug("stop() 重入 — 未在运行, 跳过.")
            return
        _running = False
        if _stop_event is not None:
            _stop_event.set()
        thread = _thread
        for t in _tracks.values():
            t.animation = None
            t.started_at = 0.0

    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "LED drive thread 未在 %.1fs 内 join 完成 (daemon, 进程退出兜底).",
                timeout,
            )

    # 关灯 (lock 外, 容忍失败 — 进程退出会重置)
    try:
        get_audio_client().LedControl(0, 0, 0)
    except Exception:
        logger.exception("LedControl(0,0,0) 收尾关灯失败 (忽略).")

    with _state_lock:
        _thread = None
        _stop_event = None
    logger.info("LED runtime stopped.")


def is_running() -> bool:
    with _state_lock:
        return _running


# ═══════════════════════════════════════════════════════════════════════════════
# 公开接口 — 轨道操作 + 基线
# ═══════════════════════════════════════════════════════════════════════════════

def set_background(anim: Animation) -> None:
    """安置 background 轨 — 常驻底色, 被 event / animation 临时压制后自动恢复显形."""
    _put_track("background", anim)


def play_animation(anim: Animation) -> None:
    """放上 animation 轨 — 播完自动让位 background. 被 event 压制."""
    _put_track("animation", anim)


def play_event(anim: Animation) -> None:
    """放上 event 轨 — 最高优先级, 抢占 animation/background, 播完自动让位."""
    _put_track("event", anim)


def clear_event() -> None:
    _clear_track("event")


def clear_animation() -> None:
    _clear_track("animation")


def clear_background() -> None:
    _clear_track("background")


def clear_all() -> None:
    """清掉所有轨, 驱动线程下一 tick 输出灭灯."""
    with _state_lock:
        for t in _tracks.values():
            t.animation = None
            t.started_at = 0.0


def set_led(color: ColorLike) -> None:
    """直接下发单帧颜色 — 不进 track, 绕过驱动线程. 仅供策略层 / debug.

    注意: 驱动线程下一 tick 会用 track 的当前色覆盖. 如果驱动正在运行且任何 track
    非空, 这个调用的视觉效果几乎瞬间消失. 适合驱动未启动或所有 track 空时单测用.
    """
    c = parse_color(color)
    get_audio_client().LedControl(c.r, c.g, c.b)


def _put_track(name: str, anim: Animation) -> None:
    with _state_lock:
        track = _tracks[name]
        track.animation = anim
        track.started_at = time.monotonic()


def _clear_track(name: str) -> None:
    with _state_lock:
        t = _tracks[name]
        t.animation = None
        t.started_at = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 公开接口 — 调试
# ═══════════════════════════════════════════════════════════════════════════════

def health() -> dict:
    """暴露 runtime 内部状态. 供 monitor / channel debug / sen 脚本看进度."""
    now = time.monotonic()
    with _state_lock:
        tracks_state = {}
        for name, t in _tracks.items():
            if t.animation is None:
                tracks_state[name] = {"active": False}
            else:
                tracks_state[name] = {
                    "active": True,
                    "elapsed_ms": int((now - t.started_at) * 1000),
                    "loop": t.animation.loop,
                    "clips_count": len(t.animation.clips),
                }
        return {
            "running": _running,
            "drive_fps": _drive_fps,
            "sdk_call_count": _sdk_call_count,
            "error_count": _error_count,
            "last_sent_color": _last_sent_color.to_hex() if _last_sent_color else None,
            "tracks": tracks_state,
        }


def animation_schema() -> dict:
    """返回 Animation 的 JSON schema.

    channel 层把它拼进 led_play(text__) 的 docstring, 喂给 LLM 当 prompt:

        async def led_play(text__: str) -> None:
            \"\"\"播放自定义关键帧动画. text__ = JSON, 格式如下:
            \"\"\" + json.dumps(led.animation_schema(), indent=2, ensure_ascii=False)
    """
    return Animation.model_json_schema()


# ═══════════════════════════════════════════════════════════════════════════════
# 内部: 驱动线程
# ═══════════════════════════════════════════════════════════════════════════════

def _drive_loop() -> None:
    """驱动线程主循环.

    异常处理纪律 (Runtime README §4): 任何异常 log.exception + 短 sleep + 继续,
    绝不 raise 出循环. LED 抖动 (一拍空帧) 比线程死掉好得多.
    """
    global _last_sent_color, _sdk_call_count, _error_count

    # 缓存 client (bootstrap 单例保证); fps 也缓存为局部, 避免每 tick 全局读
    client = get_audio_client()
    tick_period = 1.0 / _drive_fps
    stop_event = _stop_event
    logger.info("LED drive loop entered (fps=%d, tick=%dms).",
                _drive_fps, int(tick_period * 1000))

    while not stop_event.is_set():
        loop_start = time.monotonic()
        try:
            color = _compute_current_color()
            if color != _last_sent_color:
                # TODO(sdk-separation): LED 暂走 AudioClient.LedControl, 这是 SDK 抽象漏洞 —
                # LED 不属于 audio. 另一个实例正在抽离, 完成后改为:
                #     from ghoshell_moss_contrib.unitree.g1.sdk import led_control
                #     led_control(color.r, color.g, color.b)
                client.LedControl(color.r, color.g, color.b)
                _sdk_call_count += 1
                _last_sent_color = color
        except Exception:
            _error_count += 1
            logger.exception("LED drive loop 异常 (累计 %d, 保持线程活).", _error_count)
            time.sleep(0.1)

        elapsed = time.monotonic() - loop_start
        remaining = tick_period - elapsed
        if remaining > 0:
            stop_event.wait(timeout=remaining)

    logger.info("LED drive loop exited.")


def _compute_current_color() -> Color:
    """按优先级找当前应该输出的颜色. 顺手清掉播完的轨道."""
    now = time.monotonic()
    finished: list[str] = []
    chosen: Color = _BLACK  # 所有轨道空 → 灭灯
    with _state_lock:
        for name in _TRACK_NAMES:
            track = _tracks[name]
            if track.animation is None:
                continue
            elapsed_ms = int((now - track.started_at) * 1000)
            c = _render_animation(track.animation, elapsed_ms)
            if c is None:
                finished.append(name)
                continue
            chosen = c
            break  # 拿到最高优先级的轨道色, 不再看更低的
        for name in finished:
            t = _tracks[name]
            t.animation = None
            t.started_at = 0.0
    return chosen
