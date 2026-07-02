"""
_headphone_sen_toggle — 耳机按键 → 聆听开关 + LED + TTS 等价验证.

场景:
  跑起来后, 按耳机按键:
    - status=ok + paused → resumed, 绿闪两次, TTS "聆听开启"
    - status=ok + resumed → paused, 红闪两次, TTS "聆听关闭"
    - status != ok → 按 status 分路反馈 (耳机未配置 / 未连接 / 后台重连中等)

  这个脚本独立于 channel 层, 直接调 runtime 模块, 验证 callback 链是否完整.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._headphone_sen_toggle

前置:
  - sdk.bootstrap() 已完成 (DDS + monitor). 脚本默认先调.
  - 蓝牙耳机已连且 listener 配置已生成 (~/.moss_g1_listener.json)
  - 火山引擎 ASR 环境变量已设 (.moss_ws/.env)
  - 用户在 input 组 (能读 /dev/input/event*)
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    print("[bootstrap] starting sdk...")
    from ghoshell_moss_contrib.unitree.g1 import sdk

    sdk.bootstrap()

    print("[bootstrap] starting runtimes...")
    from ghoshell_moss_contrib.unitree.g1.runtime import (
        headphone_buttons,
        listener,
        led,
        audio,
    )

    led.start()
    audio.start()
    listener.start()
    listener.pause()  # 默认关闭聆听, 跟 channel 一致
    headphone_buttons.start()

    # headphone_buttons + listener 都是 daemon 线程, start() 只创建线程,
    # 不等待设备发现完成. evdev 扫描 ~0.3s, listener backend (asyncio loop
    # + miniaudio capture) ~1-2s. 给 2s 上限, 提前就绪则提前跳出.
    _deadline = time.time() + 2.0
    while time.time() < _deadline:
        hb = headphone_buttons.health()
        hl = listener.health()
        if hb["device"] is not None and hl.status == "ok":
            break
        time.sleep(0.1)

    hb = headphone_buttons.health()
    hl = listener.health()

    # 就绪检查: device 仍为 None = evdev 设备发现失败; status 非 ok 做分路报告.
    if hb["device"] is None:
        print("[WARN] headphone_buttons 未能找到蓝牙 evdev 设备.")
        print("       按键事件将无法被监听 — 按键不会触发任何效果.")
        print("       检查: 蓝牙耳机是否已连接? moss 用户在 input 组?")
        print()

    if hl.status == "no_config":
        print("[WARN] listener status=no_config — ~/.moss_g1_listener.json 不存在.")
        print("       跑 _listener_sen_setup 生成后重试.")
        print()
    elif hl.status == "no_device":
        print("[WARN] listener status=no_device — 蓝牙耳机未连接或不在 HFP profile.")
        print("       检查 bluetoothctl + pactl set-card-profile headset_head_unit.")
        print()
    elif hl.status not in ("ok", "stopped"):
        print(f"[WARN] listener status={hl.status} — backend 尚未就绪, 按键可能不生效.")
        print()

    def on_btn() -> None:
        """耳机按键 callback — 跟 channels/listener.py:_on_headphone_btn 同逻辑.
        status != ok 时做分路反馈, 不做假 resume.
        """
        print()
        h = listener.health()
        print(f"[TOGGLE] status={h.status}, paused={h.paused}")

        try:
            if h.status == "no_config":
                led.play_event(led.blink("#ff8800", count=3, period_ms=150))
                audio.speak("耳机未配置")
                print("[TOGGLE] → no_config, blocked")
                return
            if h.status == "no_device":
                led.play_event(led.blink("#ff8800", count=3, period_ms=150))
                audio.speak("蓝牙耳机未连接")
                print("[TOGGLE] → no_device, blocked")
                return
            if h.status in ("device_down", "ws_error"):
                led.play_event(led.blink("#ff8800", count=3, period_ms=150))
                audio.speak("耳机后台重连中")
                print(f"[TOGGLE] → {h.status}, blocked")
                return
            if h.status == "stopped":
                led.play_event(led.blink("#ff8800", count=3, period_ms=150))
                audio.speak("耳机未启动")
                print("[TOGGLE] → stopped, blocked")
                return

            # h.status == "ok" — 真正可 toggle
            if h.paused:
                listener.resume()
                led.play_event(led.blink("#00ff44", count=2, period_ms=150))
                audio.speak("聆听开启")
                print("[TOGGLE] resumed, led green, TTS '聆听开启'")
            else:
                listener.pause()
                led.play_event(led.blink("#ff2200", count=2, period_ms=150))
                audio.speak("聆听关闭")
                print("[TOGGLE] paused, led red, TTS '聆听关闭'")
        except Exception as e:
            print(f"[TOGGLE] ERROR: {e}")
            import traceback
            traceback.print_exc()

    headphone_buttons.register_callback(on_btn)

    hl = listener.health()
    hb = headphone_buttons.health()
    print(f"[state] paused={hl.paused}, status={hl.status}, device={hl.device_name}")
    print(f"[state] headphone_buttons: {hb}")
    print()
    print("等待按键... Ctrl+C 退出")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()

    print("[cleanup] stopping...")
    headphone_buttons.stop()
    listener.stop()
    print("[cleanup] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
