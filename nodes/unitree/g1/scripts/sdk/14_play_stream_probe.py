#!/usr/bin/env python3
"""
PlayStream 流式行为探路 — 回答 6 个流式 TTS 关键命题。

为什么需要这个：
  G1 内置 TTS 质量太低不可用，但 G1 内置音频通路有 ASR 一体化（推测含回声消除、声源定位）
  价值不可替代。如果 PlayStream 支持流式推送 + 即时打断，MOSS 可以自己合成高质量 TTS，
  通过 PlayStream 推送给 G1 喇叭，同时保留 ASR 能力。

6 个命题（按递进顺序，每步交互式等待人类反馈）：
  Q1. 单次推送 1 秒 sine 是否完整播放？
  Q2. 同 stream_id 多次推送是否拼接续播？
  Q3. 流式中 PlayStop 是否即时打断？
  Q4. 新 stream_id 是否抢占旧 stream_id？
  Q5. TtsMaker 播放中推 PlayStream 行为？
  Q6. 采样率猜测验证（16kHz mono / 48kHz stereo）

约定：
  app_name = "moss_probe"（不和 G1 内置 "voice" 冲突）
  sine 频率 440Hz（A4，清晰可辨）
  振幅 50% 满量程，避免削顶

用法: python 14_play_stream_probe.py <networkInterface>
"""
import sys
import time
import math
import struct


def gen_sine(samplerate: int, channels: int, duration_s: float, freq: float = 440.0, amp: float = 0.5) -> bytes:
    """生成 sine 波 PCM (s16le)"""
    n_samples = int(samplerate * duration_s)
    amp_int = int(amp * 32767)
    out = bytearray()
    for i in range(n_samples):
        v = int(amp_int * math.sin(2 * math.pi * freq * i / samplerate))
        sample = struct.pack("<h", v)
        for _ in range(channels):
            out.extend(sample)
    return bytes(out)


def wait_human(prompt: str):
    print(f"\n→ {prompt}")
    input("  (听完后按回车继续，Ctrl+C 退出)\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python 14_play_stream_probe.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)

    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()

    code, vol_orig = audio.GetVolume()
    print(f"原始音量: {vol_orig}")
    audio.SetVolume(100)

    APP = "moss_probe"

    # ====== Q1 单次推送完整性 ======
    print("\n" + "=" * 50)
    print("Q1. 单次推送 1 秒 sine — 验证基本播放")
    print("=" * 50)

    # 假设 16kHz mono（语音常见配置）— Q6 再验证
    SR = 16000
    CH = 1
    pcm = gen_sine(SR, CH, duration_s=1.0)
    print(f"  PCM 参数: {SR}Hz {CH}ch {len(pcm)} bytes (~1s)")
    sid = f"q1_{int(time.time()*1000)}"
    code, data = audio.PlayStream(APP, sid, pcm)
    print(f"  PlayStream code={code}")
    wait_human("Q1 听到 1 秒 440Hz 蜂鸣？时长是否准确？")
    audio.PlayStop(APP)

    # ====== Q2 同 stream_id 续播 ======
    print("\n" + "=" * 50)
    print("Q2. 同 stream_id 续播 — 5 块 × 200ms")
    print("=" * 50)
    sid = f"q2_{int(time.time()*1000)}"
    chunk_dur = 0.2
    n_chunks = 5
    chunk_pcm = gen_sine(SR, CH, chunk_dur)
    print(f"  分 {n_chunks} 块推送，每块 {chunk_dur}s 同 stream_id={sid}")
    for i in range(n_chunks):
        code, data = audio.PlayStream(APP, sid, chunk_pcm)
        print(f"  chunk {i+1}/{n_chunks} code={code}")
        # 不 sleep，连续推送
    wait_human("Q2 听到连续 1 秒 sine 还是 5 段断裂？断裂的话能否听出间隔？")
    audio.PlayStop(APP)
    time.sleep(1)

    # ====== Q3 流式中打断 ======
    print("\n" + "=" * 50)
    print("Q3. 流式中 PlayStop — 即时打断验证")
    print("=" * 50)
    sid = f"q3_{int(time.time()*1000)}"
    pcm_3s = gen_sine(SR, CH, 3.0)
    print(f"  推送 3 秒长音 stream_id={sid}")
    code, data = audio.PlayStream(APP, sid, pcm_3s)
    print(f"  PlayStream code={code}")
    print(f"  等待 1 秒后 PlayStop...")
    time.sleep(1)
    audio.PlayStop(APP)
    print(f"  PlayStop 发出（人类应在 ~1s 听到声音停止）")
    wait_human("Q3 是否在 ~1 秒处即时停止？还是继续播完？停止有无尾音？")

    # ====== Q4 新 stream_id 抢占 ======
    print("\n" + "=" * 50)
    print("Q4. 新 stream_id 抢占 — 多流冲突行为")
    print("=" * 50)
    sid_old = f"q4_old_{int(time.time()*1000)}"
    sid_new = f"q4_new_{int(time.time()*1000)}"
    pcm_2s_low = gen_sine(SR, CH, 2.0, freq=440.0)
    pcm_2s_high = gen_sine(SR, CH, 2.0, freq=880.0)
    print(f"  推送 2 秒 440Hz (低音) stream_id={sid_old}")
    audio.PlayStream(APP, sid_old, pcm_2s_low)
    print(f"  立即推送 2 秒 880Hz (高音) stream_id={sid_new}")
    code, data = audio.PlayStream(APP, sid_new, pcm_2s_high)
    print(f"  PlayStream code={code}")
    wait_human("Q4 听到的是: (a)只有低音 (b)只有高音 (c)叠加 (d)前低后高拼接")
    audio.PlayStop(APP)
    time.sleep(1)

    # ====== Q5 TTS + PlayStream 互动 ======
    print("\n" + "=" * 50)
    print("Q5. TtsMaker + PlayStream — 不同 app_name 冲突")
    print("=" * 50)
    print(f"  发起长 TTS：'让我念一段比较长的文本来观察打断行为'")
    audio.TtsMaker("让我念一段比较长的文本来观察打断行为。", 0)
    print(f"  1 秒后推 PlayStream (app_name={APP})...")
    time.sleep(1)
    sid = f"q5_{int(time.time()*1000)}"
    pcm_1s = gen_sine(SR, CH, 1.0)
    audio.PlayStream(APP, sid, pcm_1s)
    wait_human("Q5 听到的是: (a)TTS+sine 叠加 (b)只 TTS (c)只 sine (d)TTS 被打断换 sine")
    audio.PlayStop(APP)
    audio.PlayStop("voice")
    time.sleep(1)

    # ====== Q6 采样率验证 ======
    print("\n" + "=" * 50)
    print("Q6. 采样率验证 — 16kHz/48kHz 对比")
    print("=" * 50)
    print("  之前所有实验假设 16kHz mono。如果听起来正常说明猜对了。")
    print("  现在用 48kHz stereo 播相同 1 秒 440Hz：")
    sid = f"q6_{int(time.time()*1000)}"
    pcm_48k_stereo = gen_sine(48000, 2, 1.0)
    code, data = audio.PlayStream(APP, sid, pcm_48k_stereo)
    print(f"  PlayStream(48000Hz 2ch) code={code}")
    wait_human("Q6 48k stereo 听起来: (a)同样 1 秒 440Hz (b)变速变调 (c)噪音 (d)无声")
    audio.PlayStop(APP)

    # ====== 收尾 ======
    if isinstance(vol_orig, dict):
        v = vol_orig.get("volume", 100)
    else:
        v = vol_orig
    print(f"\n恢复音量到 {v}")
    audio.SetVolume(v)

    print("\n" + "=" * 50)
    print("人类反馈汇总")
    print("=" * 50)
    print("  Q1 单次推送完整性: ____")
    print("  Q2 同 stream_id 续播: ____")
    print("  Q3 即时打断: ____")
    print("  Q4 新 stream_id 抢占行为: ____")
    print("  Q5 TTS + PlayStream 互动: ____")
    print("  Q6 采样率: ____")
    print("\n推断:")
    print("  - 流式 TTS 是否可行：Q1 & Q2 决定")
    print("  - 即时打断是否可行：Q3 决定")
    print("  - 多流管理策略：Q4 & Q5 决定")
    print("  - PCM 格式契约：Q6 决定")


if __name__ == "__main__":
    main()
