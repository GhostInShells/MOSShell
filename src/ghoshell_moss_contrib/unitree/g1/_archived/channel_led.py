"""
G1 LED 控制 channel — 帧+duration+loop 动画 + idle 复位 + pub 子线程.

架构:
  - 模块级 _target_rgb: 当前应播的颜色 (threading.Lock 保护)
  - 子线程 _pub_loop: 20Hz 读 _target_rgb, 跟上一帧不同才发 LedControl (去抖)
  - running hook: 启动子线程, 等 runtime.wait_closed, 关闭时停子线程
  - idle hook: 空闲时把 target 设回 idle_color (idle hook 是一次性, 不阻塞)
  - command: 修改 target_rgb + asyncio.sleep 走帧序列

LedControl 是阻塞 RPC. 跑在子线程避免阻塞 event loop.

不做渐变 — 帧是离散的 (r,g,b,dur).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.core.concepts.channel import ChannelCtx

from ._bootstrap import get_audio_client

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级状态 — 一个进程一个 led channel, 共享
# ═══════════════════════════════════════════════════════════════════════════════

_target_rgb: tuple[int, int, int] = (0, 0, 0)
_target_lock = threading.Lock()

_idle_color: tuple[int, int, int] = (0, 0, 0)
"""命令空闲 + idle hook 触发时, 灯回到这个色. 默认熄灭."""

_pub_hz: float = 20.0
"""子线程刷新频率. 20Hz = 50ms 间隔. 这个值是保守起点, 实测过载再降."""

_stop_event = threading.Event()
_pub_thread: threading.Thread | None = None


def _set_target(rgb: tuple[int, int, int]) -> None:
    global _target_rgb
    with _target_lock:
        _target_rgb = rgb


def _get_target() -> tuple[int, int, int]:
    with _target_lock:
        return _target_rgb


def _pub_loop() -> None:
    """子线程: 周期读 target_rgb, 跟上一帧不同则发 LedControl.

    去抖: 同上一帧不发. LedControl 失败 log + 继续.
    退出时熄灭 (0,0,0).
    """
    last_sent: tuple[int, int, int] | None = None
    interval = 1.0 / _pub_hz

    logger.info("led pub loop: started @%.1fHz", _pub_hz)
    while not _stop_event.is_set():
        try:
            target = _get_target()
            if target != last_sent:
                client = get_audio_client()
                code = client.LedControl(*target)
                if code != 0:
                    logger.warning("LedControl(%s) failed code=%d", target, code)
                last_sent = target
        except Exception:
            logger.exception("led pub loop iteration failed")

        # interruptible sleep
        if _stop_event.wait(timeout=interval):
            break

    # 退出时熄灭
    try:
        client = get_audio_client()
        client.LedControl(0, 0, 0)
        logger.info("led pub loop: stopped (LED turned off)")
    except Exception:
        logger.exception("led pub loop: failed to turn off LED at exit")


# ═══════════════════════════════════════════════════════════════════════════════
# 帧解析 — 接受 tuple/list/dict 三种形态
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_frame(f: Any) -> tuple[int, int, int, float]:
    """把单个 frame 标准化为 (r, g, b, dur).

    接受:
      dict {"r":, "g":, "b":, "dur":}
      list/tuple [r, g, b, dur]
    """
    if isinstance(f, dict):
        return (int(f['r']), int(f['g']), int(f['b']), float(f['dur']))
    if isinstance(f, (list, tuple)) and len(f) == 4:
        return (int(f[0]), int(f[1]), int(f[2]), float(f[3]))
    raise ValueError(f"invalid frame: {f!r}")


def _validate_rgb(r: int, g: int, b: int) -> None:
    for name, v in (('r', r), ('g', g), ('b', b)):
        if not 0 <= int(v) <= 255:
            raise ValueError(f"{name} must be in [0, 255], got {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# Channel 构建
# ═══════════════════════════════════════════════════════════════════════════════


def build_led_channel() -> MutableChannel:
    """构建 LED channel.

    暴露命令:
      set_color(r, g, b)              -> 立刻改色 (单帧)
      play(frames, loop=1)            -> 播一段动画
      set_idle(r, g, b)               -> 改 idle color (命令结束 + idle 时回到)
      stop()                          -> 立刻停, target 设到 idle_color

    生命周期:
      running -> pub_loop 子线程在跑
      idle    -> target 设回 idle_color (一次性)
      close   -> 子线程退出 + 灯熄灭

    无门控. 无 warrant. 任何 G1 模式可用.
    """
    chan = new_channel(
        name="led",
        description="G1 机身 RGB LED 控制. 支持帧动画 + idle 复位.",
    )

    # -- running: pub loop 子线程 -------------------------------------------------

    @chan.build.running
    async def _running_pub_loop() -> None:
        """启动子线程, 等 runtime 关闭, 关闭时停子线程."""
        global _pub_thread
        runtime = ChannelCtx.runtime()
        if runtime is None:
            raise RuntimeError("led channel running hook: no runtime")

        _stop_event.clear()
        _pub_thread = threading.Thread(target=_pub_loop, daemon=True, name="g1-led-pub")
        _pub_thread.start()
        try:
            await runtime.wait_closed()
        finally:
            _stop_event.set()
            if _pub_thread is not None:
                _pub_thread.join(timeout=2.0)
                if _pub_thread.is_alive():
                    logger.warning("led pub thread did not stop within 2s")
            _pub_thread = None

    # -- idle: target 回到 idle_color ---------------------------------------------

    @chan.build.idle
    async def _on_idle() -> None:
        """命令空闲时回到 idle color. 一次性 hook."""
        _set_target(_idle_color)

    # -- 命令 ----------------------------------------------------------------------

    @chan.build.command()
    async def set_color(r: int, g: int, b: int) -> str:
        """立刻设置 LED 颜色.

        Args:
            r, g, b: 0-255

        子线程下一周期(<50ms)发 LedControl. 命令立刻返回.
        """
        _validate_rgb(r, g, b)
        _set_target((int(r), int(g), int(b)))
        return "ok"

    @chan.build.command()
    async def play(frames: list, loop: int = 1) -> str:
        """播放一段帧动画.

        Args:
            frames: 帧列表. 每帧 (r,g,b,dur) tuple, [r,g,b,dur] list, 或
                    {"r":, "g":, "b":, "dur":} dict.
                    dur 单位是秒.
            loop: 循环次数. 1=播一遍, 0=无限循环直到 cancel.

        命令时长 = sum(dur for f in frames) * loop. 阻塞 channel 内后续命令.
        命令结束/cancel 时 target 回到 idle_color, idle hook 之后接管.

        Raises:
            ValueError: frames 格式错或 rgb/dur 越界
            asyncio.CancelledError: 被取消
        """
        normalized = [_normalize_frame(f) for f in frames]
        for r, g, b, dur in normalized:
            _validate_rgb(r, g, b)
            if dur < 0:
                raise ValueError(f"frame dur must be >= 0, got {dur}")
        if loop < 0:
            raise ValueError(f"loop must be >= 0 (0 = infinite), got {loop}")
        if not normalized:
            return "ok"  # 空 frames, 无操作

        try:
            count = 0
            while loop == 0 or count < loop:
                for r, g, b, dur in normalized:
                    _set_target((r, g, b))
                    if dur > 0:
                        await asyncio.sleep(dur)
                count += 1
            return "ok"
        finally:
            # 不论怎么退出(正常完成 / cancel / 异常), 都复位
            _set_target(_idle_color)

    @chan.build.command()
    async def set_idle(r: int, g: int, b: int) -> str:
        """设置 idle color. 命令结束 + idle hook 触发时灯回到这个色.

        默认 (0,0,0) 熄灭.
        """
        global _idle_color
        _validate_rgb(r, g, b)
        _idle_color = (int(r), int(g), int(b))
        return "ok"

    @chan.build.command()
    async def stop() -> str:
        """立刻停, target 设到 idle_color. 不取消正在播的 play (用 channel 级 cancel)."""
        _set_target(_idle_color)
        return "ok"

    chan.build.instruction(
        # ⚠️ 2026-06-29 校正: 这里把命令面写进 instruction 了, 是错的范式.
        # instruction 应该只描述 channel 本身的存在意义, 命令用法应该在每个 command 的
        # docstring 里 (因为 docstring 跟函数同生共死, 状态机改可见性时自然跟随).
        # 这里只是写代码时偏了, 等下次回炉时统一改, 改成类似 "G1 机身 LED" 这样的极简描述.
        "G1 机身 LED. set_color 单帧, play 多帧动画(frames=[(r,g,b,dur),...] + loop), "
        "set_idle 改空闲色. 命令结束自动复位 idle_color."
    )

    return chan
