"""
_headphone_sen_toggle — 耳机按键 → 聆听开关 + LED + TTS 等价验证.

场景:
  跑起来后, 按耳机按键:
    - 第一次: listener 从 paused → resumed, 绿闪两次, TTS "聆听开启"
    - 第二次: listener 从 resumed → paused, 红闪两次, TTS "聆听关闭"
    - 以此类推.

  这个脚本独立于 channel 层, 直接调 runtime 模块, 验证 callback 链是否完整.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._headphone_sen_toggle

前置:
  - sdk.bootstrap() 已完成 (DDS + monitor). 脚本默认先调.
  - 蓝牙耳机已连且 listener 配置已生成 (~/.moss_g1_listener.json)
  - 火山引擎 ASR 环境变量已设 (.moss_ws/.env)
  - 用户在 input 组 (能读 /dev/input/event*)

预期:
  [bootstrap] sdk ok, listener started, headphone_buttons started.
  [state] paused=True
  等待按键...

  按耳机按键 →
  [TOGGLE] paused=True → resuming
  [TOGGLE] listener resumed, led green blink, TTS "聆听开启"
  [state] paused=False

  再按 →
  [TOGGLE] paused=False → pausing
  [TOGGLE] listener paused, led red blink, TTS "聆听关闭"
  [state] paused=True

  Ctrl+C → stop all + exit.
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

    def on_btn() -> None:
        """耳机按键 callback — 完全复制 channel 里的 _on_headphone_btn 逻辑."""
        paused = listener.health().paused
        print(f"\n[TOGGLE] paused={paused} → {'resuming' if paused else 'pausing'}")
        try:
            if paused:
                listener.resume()
                led.play_event(led.blink("#00ff44", count=2, period_ms=150))
                audio.speak("聆听开启")
                print("[TOGGLE] listener resumed, led green blink, TTS '聆听开启'")
            else:
                listener.pause()
                led.play_event(led.blink("#ff2200", count=2, period_ms=150))
                audio.speak("聆听关闭")
                print("[TOGGLE] listener paused, led red blink, TTS '聆听关闭'")
        except Exception as e:
            print(f"[TOGGLE] ERROR: {e}")
            import traceback
            traceback.print_exc()

    headphone_buttons.register_callback(on_btn)

    h = listener.health()
    print(f"[state] paused={h.paused}, status={h.status}, device={h.device_name}")
    print(f"[state] headphone_buttons: {headphone_buttons.health()}")
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
