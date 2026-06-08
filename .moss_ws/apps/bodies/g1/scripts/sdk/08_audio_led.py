#!/usr/bin/env python3
"""
音频和灯光交互验证。非运动，安全。
验证: SetVolume, LedControl(RGB), TtsMaker(短文本), TTS 中断探路

SDK 参考:
  example/g1/audio/g1_audio_client_example.py  — AudioClient 基础用法
  example/g1/audio/g1_audio_client_play_wav.py  — PlayStream + PlayStop 用法
  unitree_sdk2py/g1/audio/g1_audio_client.py    — AudioClient 实现
  src/unitree_sdk2_python/

安全: 此脚本不涉及运动控制。仅音频+LED。

前置:
  G1 开机 + RPC 服务运行 + 人类在场确认音频输出
  source .venv/bin/activate
  python 00_import_verify.py

用法: python 08_audio_led.py <networkInterface>
"""
import sys
import time

def main():
    if len(sys.argv) < 2:
        print("用法: python 08_audio_led.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()
    print("AudioClient 初始化完成\n")

    # ── 1. GetVolume + SetVolume ──
    print("=" * 40)
    print("1. GetVolume")
    code, vol = audio.GetVolume()
    if code == 0:
        print(f"OK: 当前音量 = {vol}")
    else:
        print(f"FAIL: code={code}")
        return

    print("\n2. SetVolume(30)")
    audio.SetVolume(30)
    time.sleep(3)
    code, vol = audio.GetVolume()
    print(f"确认音量 = {vol}")

    time.sleep(3)

    # ── 3. LedControl RGB ──
    print("\n3. LedControl — RGB 循环")
    colors = [("红", 255, 0, 0), ("绿", 0, 255, 0), ("蓝", 0, 0, 255), ("白", 255, 255, 255)]
    for name, r, g, b in colors:
        print(f"  LED {name}...")
        audio.LedControl(r, g, b)
        time.sleep(3)

    # ── 4. TtsMaker 短文本 ──
    print("\n4. TtsMaker — 短文本测试")
    print("  文本: 'MOSS音频测试。'")
    code = audio.TtsMaker("MOSS音频测试。", 0)
    print(f"  TtsMaker code={code}")
    print("  请人类计时: 从发送到开始播放的延迟？")

    time.sleep(3)

    # ── 5. TTS 中断探路 ──
    print("\n5. TTS 中断探路 — 长文本 + PlayStop")
    print("  发送长文本 (预计播放 >5s)...")
    long_text = "这是一条较长的测试文本。用于验证G1的语音播放是否可以被中断。请注意听语音是否在播放中被终止。"
    code = audio.TtsMaker(long_text, 0)
    print(f"  TtsMaker code={code}")
    print("  等待 2 秒后发送 PlayStop...")
    time.sleep(2)
    code = audio.PlayStop("voice")
    print(f"  PlayStop('voice') code={code}")
    print("  请人类判断: 语音是否被中断？")

    time.sleep(3)

    # ── 6. PlayStream 最小推送 ──
    print("\n6. PlayStream — 最小 PCM 推送 (160 字节，约 5ms 静音)")
    silent_pcm = bytes([0] * 160)
    stream_id = str(int(time.time() * 1000))
    code, data = audio.PlayStream("moss_test", stream_id, silent_pcm)
    print(f"  PlayStream code={code}")
    code = audio.PlayStop("moss_test")
    print(f"  PlayStop code={code}")

    time.sleep(3)

    # ── 恢复音量 ──
    print(f"\n恢复音量到 {vol}...")
    audio.SetVolume(vol)

    print("\n验证结论:")
    print("  [ ] GetVolume/SetVolume 是否正常？")
    print("  [ ] LED RGB 颜色是否正确？")
    print("  [ ] TTS 语音清晰可闻？延迟约多少秒？")
    print("  [ ] TTS 长文本能否被 PlayStop 中断？")
    print("  [ ] PlayStream 是否接受最小 PCM？")
    print("\n二阶实验 (后续):")
    print("  - TTS 播放完成的可靠回调 — play_state DDS 通知是否触发？")
    print("  - PlayStream 流式续播 — 同 stream_id 多次推送行为？")
    print("  - TTS 取消后能否立即播下一条？有无冷却时间？")

if __name__ == "__main__":
    main()
