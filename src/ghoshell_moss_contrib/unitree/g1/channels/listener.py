"""
Listener — G1 蓝牙耳机近场流式 ASR channel (L4).

**近场听觉通道** (与 G1 内置远场 asr channel 对称). 佩戴者对着蓝牙耳机说话时,
ghost 通过本 channel 感知到.

授权模型:
  - 默认: 耳机聆听关闭. 佩戴者按耳机按键 (KEY_PLAYCD 单击) 切换开关.
  - Y 键: 切换自由对话模式. 开启后, VAD 每判停一句就自动 drain + send NotifySignal.
  - A 键: 立即 drain 当前累积内容, 非空则 send NotifySignal (FATAL, 模型立即响应).

  感知模型: context_messages 始终 tail-N 只读, 永不 drain.

信号发送走 janus.Queue: 多个 callback 源 (VAD / A 键) 都可以从各自线程
sync_q.put(), running loop 单点 async_q.get() → CommandUtil.send_signal().

runtime 依赖:
  - ghoshell_moss_contrib.unitree.g1.runtime.listener
  - ghoshell_moss_contrib.unitree.g1.runtime.headphone_buttons
  - ghoshell_moss_contrib.unitree.g1.runtime.story_202607_fsm (button callbacks)
  - ghoshell_moss_contrib.unitree.g1.runtime.led (Y 键自由对话 LED 反馈)
"""
from __future__ import annotations

import asyncio
import logging
import time

import janus

from ghoshell_moss.core.blueprint.channel_builder import new_channel, CommandUtil
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1.runtime import listener
from ghoshell_moss_contrib.unitree.g1.runtime import headphone_buttons
from ghoshell_moss_contrib.unitree.g1.runtime import story_202607_fsm as fsm
from ghoshell_moss_contrib.unitree.g1.runtime import led, audio
from ghoshell_moss_contrib.unitree.g1.channels._utils import (
    check_g1_available, check_channel_information,
)

logger = logging.getLogger("moss.g1.channels.listener")

__all__ = ["listener_channel"]

# 每回合装配 context_messages 时, 从 finalized buffer 尾部读多少条历史.
_RECENT_N = 12

# janus queue 上限. 人的短句不会堆积上百条, 512 只是防止内存爆炸.
_Q_MAXSIZE = 512

# ═══════════════════════════════════════════════════════════════════════════════
# 自由对话模式 — Y 键翻转, LED pulse 确认, context 告知模型
# ═══════════════════════════════════════════════════════════════════════════════

_free_dialog: bool = False


def _toggle_free_dialog() -> None:
    """Y 键 callback — 跑在 reader 线程. LED 直接调, 线程安全."""
    global _free_dialog
    _free_dialog = not _free_dialog
    try:
        if _free_dialog:
            led.play_event(led.pulse("#aa44ff", count=1, period_ms=600))
        else:
            led.play_event(led.pulse("#444444", count=1, period_ms=400))
    except Exception:
        logger.exception("_toggle_free_dialog: LED 异常 (isolated)")


# ═══════════════════════════════════════════════════════════════════════════════
# janus Queue — callback 线程 → asyncio running loop
# ═══════════════════════════════════════════════════════════════════════════════

# 在 startup 中创建, running 中消费. 模块级变量供 callback 访问.
_event_q: janus.Queue | None = None


def _try_drain_and_enqueue(*, force_finalize_partial: bool = False) -> None:
    """drain listener, 非空则入队. 可被多个 callback 源安全调用.

    force_finalize_partial=True: 把当前正在说的 partial 也强制当 final 拿走,
    并 abort 当前 recognize session — 服务端后续对这句的 final 更新被丢弃.
    A 键 (立即打断 + 交付) 用这个, 自由对话 VAD 判停 (整句已完成) 走默认 False.
    """
    if _event_q is None:
        return
    batch = listener.drain(force_finalize_partial=force_finalize_partial)
    if batch.items:
        try:
            _event_q.sync_q.put_nowait(batch)
        except janus.SyncQueueFull:
            logger.warning("listener channel: event queue full, dropping drain batch")


# ═══════════════════════════════════════════════════════════════════════════════
# callbacks — 跑在各自源线程, 不能阻塞
# ═══════════════════════════════════════════════════════════════════════════════

def _on_headphone_btn() -> None:
    """耳机按键 → 按 listener status 分状态反馈.

    status != "ok" 时不做假 resume — LED/TTS 反馈实际原因, 避免"绿闪 +
    '聆听开启'"骗人 (真实 ASR 链路仍断).
    """
    logger.info("headphone button pressed, checking listener status...")
    try:
        h = listener.health()
        logger.info("listener health: status=%s paused=%s", h.status, h.paused)

        if h.status == "no_config":
            logger.warning(
                "headphone toggle blocked: no_config. run "
                "`python -m ghoshell_moss_contrib.unitree.g1.runtime._listener_sen_setup` "
                "on PC2 to generate ~/.moss_g1_listener.json."
            )
            led.play_event(led.blink("#ff8800", count=3, period_ms=150))
            audio.speak("耳机未配置")
            return
        if h.status == "no_device":
            logger.warning("headphone toggle blocked: no_device (bluetooth headphone not connected)")
            led.play_event(led.blink("#ff8800", count=3, period_ms=150))
            audio.speak("蓝牙耳机未连接")
            return
        if h.status in ("device_down", "ws_error"):
            logger.warning("headphone toggle blocked: status=%s (backend recovering)", h.status)
            led.play_event(led.blink("#ff8800", count=3, period_ms=150))
            audio.speak("耳机后台重连中")
            return
        if h.status == "stopped":
            logger.warning("headphone toggle blocked: listener not started")
            led.play_event(led.blink("#ff8800", count=3, period_ms=150))
            audio.speak("耳机未启动")
            return

        # h.status == "ok" — 真正可 toggle
        if h.paused:
            logger.info("currently paused → resuming")
            listener.resume()
            led.play_event(led.blink("#00ff44", count=2, period_ms=150))
            audio.speak("聆听开启")
            logger.info("listener resumed + LED green + TTS")
        else:
            logger.info("currently listening → pausing")
            listener.pause()
            led.play_event(led.blink("#ff2200", count=2, period_ms=150))
            audio.speak("聆听关闭")
            logger.info("listener paused + LED red + TTS")
    except Exception:
        logger.exception("_on_headphone_btn: 异常 (isolated)")


def _on_fsm_button(button_name: str) -> None:
    """fsm 按键回调 — 跑在 cyclonedds reader 线程.

    A 键 (trigger): 无差别 drain → 非空入队 + 白闪确认.
    Y 键 (audio_toggle): 翻转自由对话模式 + LED 确认.
    F1 键 (listener_toggle): 翻转 listener ASR 硬开关, 等价耳机中键.
    X 键 (interrupt): 不管, 由 fsm channel 处理.
    """
    logger.info("fsm button: %s", button_name)
    try:
        if button_name == "trigger":
            # A 键 = 立即打断 + 交付. 强制拿 partial + abort session, 丢弃后续 final.
            _try_drain_and_enqueue(force_finalize_partial=True)
            led.play_event(led.blink("#ffffff", count=2, period_ms=200))
        elif button_name == "audio_toggle":
            _toggle_free_dialog()
        elif button_name == "listener_toggle":
            _on_headphone_btn()
    except Exception:
        logger.exception("_on_fsm_button(%s): 异常 (isolated)", button_name)


def _on_sentence(_utterance) -> None:
    """listener sentence callback — 跑在 backend asyncio 线程.

    仅在自由对话模式下 drain + 入队.
    """
    if _free_dialog:
        _try_drain_and_enqueue()


# ═══════════════════════════════════════════════════════════════════════════════
# instruction — 近场听觉 + 交互模型 + 状态解读
# ═══════════════════════════════════════════════════════════════════════════════

_INSTRUCTION = """\
你有近场听觉 — 通过蓝牙耳机接收佩戴者对你说话的声音.

**当前状态: 默认关闭.** 你**现在听不到**耳机的声音. 佩戴者有两个入口开启聆听:

  - **按耳机上的中键** (最方便, 无授权前置)
  - **按遥控器 F1 键** (需先按 L1+Start 进 AI 模式)

两个入口等价, 开启后你会听到 "聆听开启" 的语音确认, 此后佩戴者的每一句话
你都能在 context 里看到.

**你不能自己开启聆听** — 交互权在佩戴者手里. 如果有人问你怎么通过耳机和你说话,
或问你能不能听到 — 直接告诉他: "按一下耳机上的中键就可以, 或者按遥控器 F1
(需要先按 L1+Start 进 AI 模式). 我听到后会给你确认."

开启后:
- 佩戴者说话 → 实时出现在你的 context (<g1.listener_utterance>)
- 佩戴者可以长篇说话不会被打断. VAD 自动切句.
- 再按耳机中键 / F1 键 → 关闭聆听 ("聆听关闭" 语音确认)

你的近场听觉状态随时在 <g1.listener_status> 里. paused=true 就是关着的.

自由对话模式 (遥控器 Y 键, 需 AI 模式):
- 佩戴者按 Y 键切换. 开启后, 每句 VAD 判停时自动通知你, 你会被唤醒.
- 前置: 聆听必须已开 (F1 或耳机中键). 聆听关时自由对话开关无效.
- <g1.listener_free_dialog> 反映当前是否开启.

A 键 (需 AI 模式):
- 无论自由对话是否开启, 按 A 键立即把当前累积内容通知你.

**Y/A/F1 键的前置**: 需要先按遥控器 L1+Start 进 AI 模式, 才生效. 未进 AI 模式时
按 Y/A/F1 只在 fsm history 留一条 "人按了但没生效" 事件 (你能通过 g1_fsm channel
的 <g1.fsm_events> 看到), 不触发本 channel 效果. 如果佩戴者反映 "按了 Y/A/F1
没反应" — 直接告诉他: "按一下 L1+Start 进 AI 模式先, 然后再按". 耳机中键不受
此前置约束.

远场听觉:
- G1 机身自带麦克风阵列 (g1.asr channel). 走到 G1 身边说话会被自动识别进 context.
  不需要按键, 但距离有限 (< 3m), 没有声源方位.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# channel 组装
# ═══════════════════════════════════════════════════════════════════════════════

listener_channel = new_channel(
    name="listener",
    description="G1 蓝牙耳机近场流式 ASR — 佩戴者近场语音输入, 由佩戴者按键控制开关与提交.",
)
listener_channel.build.instruction(_INSTRUCTION)


# ─── startup: 启 runtime + 注册 callbacks ──────────────────────────────────

@listener_channel.build.startup
async def _on_startup() -> None:
    global _event_q
    _event_q = janus.Queue(maxsize=_Q_MAXSIZE)

    # runtime 启动链 (全部幂等).
    # sdk.bootstrap() 由 channels.py 在 import 路径顶部完成.
    # control_pad / fsm 由 fsm channel 的 startup 负责, 这里不重复调.
    # 但 fsm channel 的 startup 可能未完成 — fsm.start() 幂等, 放心调.
    fsm.start()
    headphone_buttons.start()
    listener.start()
    audio.start()

    # 显式 report listener runtime status — 5 个非终态 (stopped / no_config /
    # no_device / device_down / ok) 里只有 ok 才能真实听. 假设都是 ok 会给
    # 模型/人类假反馈 (LED 绿闪 + '聆听开启' TTS 但真实链路断).
    h = listener.health()
    if h.status == "no_config":
        logger.warning(
            "listener started in status=no_config: config file not found. "
            "run `python -m ghoshell_moss_contrib.unitree.g1.runtime._listener_sen_setup` "
            "on PC2 to generate ~/.moss_g1_listener.json before headphone toggle will work."
        )
    elif h.status not in ("ok", "stopped"):
        logger.warning("listener started in status=%s, backend may not be ready", h.status)
    else:
        logger.info("listener started: status=%s", h.status)

    # 默认关闭聆听. start() 后立刻 pause — pause 只设 _paused flag,
    # backend 线程检查 flag 后不会开 ASR session.
    listener.pause()

    # 注册耳机按键回调 → 翻转聆听开关
    headphone_buttons.register_callback(_on_headphone_btn)

    # 注册 fsm 按键回调 → A 键 drain, Y 键自由对话
    fsm.register_button_callback(_on_fsm_button)

    # 注册 listener sentence callback → 自由对话模式下 drain
    listener.register_sentence_listener(_on_sentence)

    logger.info("listener channel startup: runtimes started, callbacks registered, "
                "listening paused by default.")


# ─── running: janus queue → NotifySignal ────────────────────────────────────

@listener_channel.build.running
async def _running_loop() -> None:
    """消费 janus queue, 在 channel context 内发送 NotifySignal.

    payload 格式: 每条一行 "[HH:MM:SS 类型] 文本". 类型: FINAL (VAD 判停) /
    FORCED (A 键强制打断, partial 当 final). items 按到达时间升序 — 最早说的
    在最上, 最新的在最下, 让模型顺序读.
    """
    while True:
        batch = await _event_q.async_q.get()
        try:
            lines: list[str] = ["=== 佩戴者近场语音 (蓝牙耳机 ASR) ==="]
            for u in batch.items:
                ts = time.strftime("%H:%M:%S", time.localtime(u.received_at))
                tag = "FORCED" if u.forced else "FINAL"
                lines.append(f"[{ts} {tag}] {u.text}")
            if batch.forgotten:
                lines.append(f"(此前有 {batch.forgotten} 句被 buffer 挤掉未收到)")
            content = "\n".join(lines)
            description = f"近场语音 ({len(batch.items)}句): {batch.items[-1].text if batch.items else '(空)'}"
            signal = new_notify_signal(
                content,
                priority=Priority.FATAL,
                description=description,
                stale_timeout=30.0,
            )
            CommandUtil.send_signal(signal)
            logger.info("listener: NotifySignal sent, items=%d", len(batch.items))
        except Exception as e:
            logger.exception("listener running: send signal failed: %s", e)


# ─── context_messages: tail-N 只读 + 状态告知 ──────────────────────────────

@listener_channel.build.context_messages
async def _listener_context() -> list[Message]:
    check_g1_available()

    h = listener.health()
    messages: list[Message] = [_status_message(h), _free_dialog_message()]

    if h.status != "ok" or h.paused:
        return messages

    recent = listener.peek_recent_finalized(_RECENT_N)
    for u in recent:
        ts = time.strftime("%H:%M:%S", time.localtime(u.received_at))
        messages.append(Message.new(
            tag="g1.listener_utterance",
            attributes={"ts": ts},
        ).with_content(u.text))

    if h.forgotten_since_last_drain:
        messages.append(Message.new(
            tag="g1.listener",
            attributes={"forgotten": h.forgotten_since_last_drain},
        ).with_content(f"有 {h.forgotten_since_last_drain} 句话被遗忘了."))

    return messages


def _status_message(h) -> Message:
    if h.status == "stopped":
        note = "listener 未启动."
    elif h.status == "no_config":
        note = "耳机未配置, 需先跑 _listener_sen_setup."
    elif h.status == "no_device":
        note = "蓝牙耳机未连接."
    elif h.status == "device_down":
        note = "蓝牙耳机断流, 后台重试中."
    elif h.status == "ws_error":
        note = "ASR 服务端异常, 后台重连中."
    elif h.paused:
        note = "聆听已关闭. 按耳机按键开启."
    elif h.status == "ok":
        note = "聆听已开启, 佩戴者说话你能听到."
    else:
        note = f"耳机状态: {h.status}."

    return Message.new(tag="g1.listener_status").with_content(note)


def _free_dialog_message() -> Message:
    return Message.new(tag="g1.listener_free_dialog").with_content(
        f"自由对话: {'开启' if _free_dialog else '关闭'} (Y 键切换)."
    )


if __name__ == "__main__":
    check_channel_information(listener_channel)
