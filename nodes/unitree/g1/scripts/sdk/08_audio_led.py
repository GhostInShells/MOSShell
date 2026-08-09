#!/usr/bin/env python3
"""
音频和灯光交互验证。非运动，安全。
验证: SetVolume, LedControl(RGB), TtsMaker(短文本), TTS 中断探路, PlayStream 最小推送

修正记录 (2026-06-15):
  - GetVolume 返回 (code, dict)，dict 形如 {"volume": N}。前任版本直接把 dict 当 int
    传回 SetVolume(vol) — 会写入异常音量。本版用 _vol_value() 提取。
  - 收尾把 LED 复位到 (0, 0, 0)，避免脚本结束后停在白色。

SDK 参考:
  example/g1/audio/g1_audio_client_example.py — AudioClient 基础用法
  unitree_sdk2py/g1/audio/g1_audio_client.py   — AudioClient 实现:
    GetVolume() → (code, json.loads(data))     # data 是 {"volume": N}
    PlayStream(app, sid, pcm_bytes) → (code, data)

安全: 仅音频+LED。不涉及运动控制。

用法: python 08_audio_led.py <networkInterface>
"""
import sys
import time


def _vol_value(vol, default=100):
    """GetVolume 返回 {"volume": N}；兼容直接返回 int 的 SDK 变体。"""
    if isinstance(vol, dict):
        return int(vol.get("volume", default))
    if isinstance(vol, (int, float)):
        return int(vol)
    return default


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
    code, vol_raw = audio.GetVolume()
    if code != 0:
        print(f"FAIL: code={code}")
        return
    vol_orig = _vol_value(vol_raw, 100)
    print(f"OK: 原始音量 raw={vol_raw} → 解析 = {vol_orig}")

    print("\n2. SetVolume(30)")
    audio.SetVolume(30)
    time.sleep(2)
    code, vol_check = audio.GetVolume()
    print(f"确认音量 raw={vol_check} → 解析 = {_vol_value(vol_check)}")

    time.sleep(2)

    # ── 3. LedControl RGB ──
    print("\n3. LedControl — RGB 循环")
    colors = [("红", 255, 0, 0), ("绿", 0, 255, 0), ("蓝", 0, 0, 255), ("白", 255, 255, 255)]
    for name, r, g, b in colors:
        print(f"  LED {name}...")
        audio.LedControl(r, g, b)
        time.sleep(2)

    # ── 4. TtsMaker 短文本 ──
    print("\n4. TtsMaker — 短文本测试")
    print("  文本: 'MOSS音频测试。'")
    code = audio.TtsMaker("MOSS音频测试。", 0)
    print(f"  TtsMaker code={code}")
    print("  请人类计时: 从发送到开始播放的延迟？")

    time.sleep(4)

    # ── 5. TTS 中断探路 ──
    print("\n5. TTS 中断探路 — 长文本 + PlayStop")
    long_text = "这是一条较长的测试文本。用于验证G1的语音播放是否可以被中断。请注意听语音是否在播放中被终止。"
    code = audio.TtsMaker(long_text, 0)
    print(f"  TtsMaker code={code}")
    print("  等待 2 秒后发送 PlayStop('voice')...")
    time.sleep(2)
    code = audio.PlayStop("voice")
    print(f"  PlayStop code={code}")
    print("  请人类判断: 语音是否被中断？")

    time.sleep(3)

    # ── 6. PlayStream 最小推送 ──
    # 16kHz mono s16le: 16000 * 0.005 * 2 = 160 bytes ≈ 5ms 静音
    print("\n6. PlayStream — 最小 PCM 推送 (160 字节 ≈ 5ms 静音 @ 16kHz mono s16le)")
    silent_pcm = bytes([0] * 160)
    stream_id = f"moss_probe_{int(time.time()*1000)}"
    code, data = audio.PlayStream("moss_test", stream_id, silent_pcm)
    print(f"  PlayStream code={code}")
    code = audio.PlayStop("moss_test")
    print(f"  PlayStop code={code}")

    time.sleep(1)

    # ── 收尾: 恢复音量 + LED 关闭 ──
    print(f"\n恢复音量到 {vol_orig}...")
    audio.SetVolume(vol_orig)
    print("LED 关闭 (0, 0, 0)...")
    audio.LedControl(0, 0, 0)

    print("\n验证结论:")
    print("  [ ] GetVolume/SetVolume 是否正常？")
    print("  [ ] LED RGB 颜色是否正确？")
    print("  [ ] TTS 语音延迟约多少秒？(配合前任 6/15 结论: TTS 质量低不可用)")
    print("  [ ] TTS 长文本能否被 PlayStop 中断？")
    print("  [ ] PlayStream 是否接受最小 PCM？code=0 即接受")


if __name__ == "__main__":
    main()