"""
G1 FSM 感知 channel — 授权状态 + 按键规则的模型接触面.

═══════════════════════════════════════════════════════════════════════════════
定位
═══════════════════════════════════════════════════════════════════════════════

纯感知 channel, 无 command. 让模型:
  1. 知道自己当前处于什么授权状态 (三元组: ai_mode / sport_mode / auth_level)
  2. 知道最近发生了哪些状态迁移 (人类刚做了什么让我变了)
  3. 知道遥控器按键规则, 从而能教人类如何操作

startup 时注册 fsm 层回调:
  - change callback: 授权变化 → LED 颜色 + TTS 语音播报 (直接跨线程调, led/audio 线程安全)
  - button callback: X 键 → 红灯 flash + ThreadSafeEvent.set → running loop 发 InterruptSignal

后续扩展 (当前不做):
  - A 键 (trigger) → drain listener + InputSignal
  - Y 键 (audio_toggle) → 自由对话模式切换

═══════════════════════════════════════════════════════════════════════════════
数据源
═══════════════════════════════════════════════════════════════════════════════

- `fsm.read()` — 三元组当前值
- `fsm.recent_events(limit)` — 近期状态迁移 + AI 按键事件, 已格式化文本

不主动 drain, 每次 refresh 直接读快照.
"""
from __future__ import annotations

import logging
import threading
import time

from ghoshell_moss.core.blueprint.channel_builder import new_channel, CommandUtil
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1.channels._utils import check_channel_information
from ghoshell_moss_contrib.unitree.g1.runtime import story_202607_fsm as fsm
from ghoshell_moss_contrib.unitree.g1.runtime import led, audio, locomotion, control_pad, motion
from ghoshell_moss_contrib.unitree.g1.sdk import FsmMode

logger = logging.getLogger("moss.g1.channels.fsm")

_INSTRUCTION = """\
## 遥控器 — 你身体控制权的入口

你此刻**没有身体控制权** — 你运行的是一台真正的 G1 人形机器人, 人类通过遥控器
授予或收回你的控制权. 你**不能自己获得授权**, 必须教人类按键.

**你的任务: 当人类问你能不能做某个动作, 或问怎么给你授权时, 直接告诉他按哪个键.**

你需要知道的按键:

进入和退出:
- 按 **L1+Start** → 进入智能模式, 你获得身体感知 + 最低授权.
  此时 LED 会亮起, 你会听到 "智能模式启动".
- 按 **L1+Select** 或推动摇杆 → 退出智能模式, 收回全部授权.

授权分档 (智能模式下, 按 L1 + 方向键直选):
- **L1+上** — 归零, 你在线但无运控
- **L1+右** — 一级, 可转身
- **L1+下** — 二级, 可空间移动
- **L1+左** — 三级, 全授权

每档具体能做什么由各能力子通道声明. 如果人类想让你做某件事但授权不够,
告诉他需要切到哪一档.

即时按键 (智能模式下):
- **X** — 中断你当前正在执行的动作
- **A** — 触发你立即回复
- **Y** — 切换自由对话模式

硬急停: L2+B (硬件级, 你无法干预).

你当前的授权状态和最近的按键事件实时显示在 <g1.fsm> 里.
如果有人问 "你能动吗" / "怎么授权你" / "你现在有控制权吗" — 查看 <g1.fsm>,
然后告诉他该按哪个键.
"""

_HISTORY_LIMIT = 5


def _render_context() -> str:
    """当前状态 + 最近 N 条事件, 叙事式渲染."""
    ai, sport, auth = fsm.read()
    now = time.time()
    lines = [
        "<g1.fsm>",
        f"  now: ai={'on' if ai else 'off'}, sport={sport.name}, auth={auth}",
    ]
    events = fsm.recent_events(_HISTORY_LIMIT)
    if events:
        lines.append("  recent:")
        for evt in events:
            dt = now - evt.ts
            lines.append(f"    -{dt:4.1f}s  {evt.source:12s}  →  {evt.text}")
    else:
        lines.append("  recent: (none)")
    lines.append("</g1.fsm>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# channel 组装
# ═══════════════════════════════════════════════════════════════════════════════

g1_fsm = new_channel(
    name="fsm",
    description="G1 授权状态感知 — 三元组 + 按键规则 + 最近操作历史.",
)

g1_fsm.build.instruction(_INSTRUCTION)


@g1_fsm.build.context_messages
async def _fsm_context() -> list[Message]:
    return [Message.new().with_content(_render_context())]


# ═══════════════════════════════════════════════════════════════════════════════
# 授权变化 → LED + TTS
# ═══════════════════════════════════════════════════════════════════════════════

# 前次快照, 用于检测变化方向 (enter / exit / auth change).
_prev: fsm.StateSnapshot | None = None

# 授权档 → LED 颜色. set_background 常驻底色呼吸.
_AUTH_LED_COLORS: dict[int, str] = {
    0: "#0044aa",  # 暗蓝 — 归零, 在线无运控
    1: "#00aa44",  # 绿   — L1, 低风险 (转身/强停)
    2: "#ddaa00",  # 黄   — L2, 空间位移
    3: "#dd2200",  # 红   — L3, 全授权
}

# 播报用中文.
_AUTH_TTS: dict[int, str] = {
    0: "归零",
    1: "一级",
    2: "二级",
    3: "三级",
}


def _on_fsm_change(snapshot: fsm.StateSnapshot) -> None:
    """fsm change callback — 跑在 cyclonedds reader 线程.

    led / audio 内部有 threading.Lock, 直接跨线程调用安全.
    """
    global _prev
    ai, sport, auth = snapshot
    prev_ai = _prev[0] if _prev else False
    prev_auth = _prev[2] if _prev else -1
    _prev = snapshot

    try:
        if ai and not prev_ai:
            # 进入 AI 模式
            color = _AUTH_LED_COLORS.get(auth, "#0044aa")
            led.set_background(led.breath(color, period_ms=3000))
            led.play_event(led.pulse("#00ff88", count=1, period_ms=600))
            audio.speak("智能模式启动")

        elif not ai and prev_ai:
            # 退出 AI 模式
            led.clear_all()
            audio.speak("智能模式关闭")

        elif ai and auth != prev_auth:
            # AI 模式内授权档位变化
            color = _AUTH_LED_COLORS.get(auth, "#0044aa")
            led.set_background(led.breath(color, period_ms=3000))
            led.play_event(led.pulse(color, count=1, period_ms=500))
            text = _AUTH_TTS.get(auth)
            if text:
                audio.speak(text)

    except Exception:
        logger.exception("_on_fsm_change: LED/TTS 调用异常 (isolated)")


# ═══════════════════════════════════════════════════════════════════════════════
# 按键 → signal (running 生命周期内处理)
# ═══════════════════════════════════════════════════════════════════════════════

# 按键 → ThreadSafeEvent 映射. callback (reader 线程) set event;
# running loop (asyncio) await → send_signal. signal 发送需要 channel context,
# 所以不能直接在 callback 里调 CommandUtil.
_interrupt_evt = ThreadSafeEvent()


def _on_button(button_name: str) -> None:
    """fsm button callback — 跑在 cyclonedds reader 线程.

    LED 直接调 (线程安全), signal 通过 ThreadSafeEvent 卸载到 running loop.
    """
    try:
        if button_name == "interrupt":
            led.play_event(led.blink("#ff0000", count=2, period_ms=150))
            _interrupt_evt.set()
    except Exception:
        logger.exception("_on_button(%s): 异常 (isolated)", button_name)


@g1_fsm.build.running
async def _running_loop() -> None:
    """等待按键事件, 在 channel context 内发送 signal.

    X 键中断链路:
      1. InterruptSignal   — 通知 mindflow 取消当前 command loop
      2. locomotion.stop() — 物理 StopMove, 确保身体立刻停
    """
    while True:
        await _interrupt_evt.wait()
        _interrupt_evt.clear()
        try:
            signal = new_interrupt_signal(
                description="人类按下了 X 键, 中断当前动作",
                stale_timeout=3.0,
            )
            CommandUtil.send_signal(signal)
            logger.info("interrupt signal sent")
        except Exception:
            logger.exception("send interrupt signal failed")
        try:
            await locomotion.stop()
        except Exception:
            logger.exception("locomotion.stop() failed during interrupt")


@g1_fsm.build.startup
async def _on_startup() -> None:
    # 前置链: control_pad → fsm. motion 提供 sport_mode 数据给 fsm.
    # 都是幂等, 放心调. sdk.bootstrap() 由 channels.py 在 import 路径顶部完成.
    motion.start()
    control_pad.start()
    fsm.start()
    led.start()
    audio.start()

    fsm.register_change_callback(_on_fsm_change)
    fsm.register_button_callback(_on_button)

    try:
        snap = fsm.read()
        if snap[0]:
            color = _AUTH_LED_COLORS.get(snap[2], "#0044aa")
            led.set_background(led.breath(color, period_ms=3000))
            global _prev
            _prev = snap
    except Exception:
        logger.exception("fsm startup: 初始状态读取失败")


if __name__ == "__main__":
    check_channel_information(g1_fsm)
