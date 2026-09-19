#!/usr/bin/env python3
"""AEC 回声消除与 far/near 对齐实测 —— interleaved-voice 留档脚本.

本脚本是 interleaved-voice 工作项「前置修复 3: AEC 屏蔽细节」的验证载体, 也是
「回声消除路线盘点」的调研留档. 一个文件装三样东西: 调研结论、可复现的实测、跑完
填回的数字. 跑一次就在, 不用再靠某次 node 运行的终端输出.

## 为什么需要 AEC

交错语音对话 (interleaved voice) 的核心是「边说边听」: ghost 说话时人还能插话
(barge-in). 单进程里既开麦又出声, 麦克风会听见扬声器 —— 这就是回声. 回声不处理,
ASR 会把 ghost 自己的话当成用户输入, 对话自激.

AEC 的定位是**门控「拦回声」那一格的实现**: 门控活在音频层、ASR 之前, 拦静音 +
拦回声. AEC 只负责后一格.

## 路线盘点 (2026-09-19 调研, 结论)

miniaudio 不提供 AEC —— capture 与 player 都是裸 PCM. 三条路线:

| 路线 | 机制 | 代价 | 结论 |
|------|------|------|------|
| OS 级 | macOS VoiceProcessingIO AudioUnit (pyobjc / 原生辅助) | 平台锁定, 破坏 miniaudio 零系统依赖的现状 | 否 |
| pip 级 | WebRTC AEC3 (`pywebrtc-audio`) | 需对齐 near/far + 收敛时间 | **选它** |
| 门控级 | 纯半双工: 说时闸麦 | 与 barge-in 冲突 (ghost 说完人才能打断) | 否 |

选 `pywebrtc-audio`: pip 装, arm64 + py3.12 wheel 命中, 无系统依赖. AEC3 自带
delay estimator —— **对齐是插件自带的机制 (``stream_delay_ms`` 提示 + 内部自适应
估计), 不是事后互相关/能量起点的 hack 对齐**. 这正是「对齐延迟要在抽象上有机制」
这条要求的落点.

已知踩坑 (2026-09-19): miniaudio ``DuplexStream`` 播侧在本机静默失效 (无双工设备),
所以单进程听说必须用**独立的** capture + player, 不能用 DuplexStream.

## pywebrtc-audio 接口 (本脚本依赖的)

    EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)
    .process(near: np.ndarray, far: np.ndarray) -> np.ndarray   # 逐 10ms 帧
    .stream_delay_ms: int    # 读写; AEC 的延迟提示
    .reset()                 # 保留配置, 清 AEC 滤波器状态

- ``near`` = 麦克风采集 (含回声), ``far`` = 参考信号 (写入设备的播放帧).
- 两者是**同一时间轴**的帧: near[t] 里含的是 far[t-D] 的回声, AEC 内部据
  ``stream_delay_ms`` (预期延迟) + delay estimator 找回归路径.
- 非线程安全: 一个实例一个线程.

参考信号在本进程内直接可得: ``StreamAudioPlayer.on_play`` (真实写入设备的帧) /
``observe(PlaybackSample)``. 数据不缺, 缺的是对齐与收敛.

## 用法

离线合成验证 (无设备, 可复现, 默认):

    .venv/bin/python aec_alignment_probe.py offline

真实设备单进程听说实测, speak 一句后看 ASR 是否听见自己 (需外放; 戴耳机测到的是
「没回声」):

    .venv/bin/python aec_alignment_probe.py live [device_pattern] [sentence]
    .venv/bin/python aec_alignment_probe.py live --no-aec   # AEC off 地面真值

live 走的是「链路自搭」形态: ``capture → [AEC] → resample → ASR``, 不改核心 ——
它就是 ``HostListenerState._audio_gen`` 那条边界 (音频层、ASR 之前) 的原型, 也是
门控「拦回声」那一格该插的位置.

## 实测结论

### offline —— 合成验证 (2026-09-19)

    .venv/bin/python .../aec_alignment_probe.py offline

| 实验 | 稳态 ERLE | 早期 ERLE (0.05~0.6s) | 近端偏差 |
|------|-----------|------------------------|----------|
| 单讲 / hint=0 (未提示) | 21.1 dB | 37.4 dB | - |
| 单讲 / hint=60ms (=真延迟) | 21.1 dB | 35.9 dB | - |
| 单讲 / hint=500ms (严重失配) | 21.2 dB | 21.3 dB | - |
| 双讲 / 近端人声保留 | 23.3 dB | 35.9 dB | -4.1 dB |

读出的结论:

1. **AEC3 可用**: 对约 60ms 的声学延迟 + 三点早期反射, 稳态抑制 ~21dB.
2. **``stream_delay_ms`` 确实生效, 且 estimator 会兜底**: hint 严重失配 (500ms) 时
   早期抑制只有 ~21dB (收敛慢), 贴近真延迟时 ~36dB; 但最终**稳态都收敛到 ~21dB** ——
   错的 hint 只拖慢前 1 秒, 不改变极限. 所以: hint 是「加速收敛的提示 (buffer/delay
   mechanism)」, 不是「必须精确的对齐参数」.
3. **因此不需要、也不应该在外层写互相关/能量起点的「事后 hack 对齐」** —— AEC3 的
   delay estimator 就是那个机制 (前置修复 3 要求的正面证据).
4. **双讲下回声仍被压 ~23dB, 但近端人声被削了 ~4dB** (近端偏差 -4.1dB) —— AEC3
   的双讲行为对近端有可观测损伤. 真实对话里是否影响 barge-in, 待 live 与真机听感判断.
5. 合成信号是类语音 (带限噪声 × 音节包络) + 三点早期反射的简化房间; 数字用于验证
   管线与量级, **不等于真机声学**.

### live —— 真机 (待跑)

判据: AEC=off 时 ASR 应认出 ghost 自己说的话 (回声 = 地面真值); on 时应听不到.
两者都要外放跑, 且环境安静. 结果跑完回填此处.

真机对齐的已知风险 (对应 hint 的有效性): far 来自 ``player.on_play`` (即时), near 来自
capture 并经 transport 到达 (**有传输延迟**), 两条流的时间基准不同. 脚本用 ``far_ring``
缓冲取「最近帧」—— 若传输延迟超过 AEC3 estimator 的搜索范围, 对齐会失败, 届时需要一个
显式的 delay/buffer 对齐 (测出 near 链路固定延迟, 从 far_ring 取对应时刻的帧).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

#: AEC3 的帧长是 10ms.
FRAME_MS = 10

#: AEC 处理采样率 — pywebrtc-audio 的 EchoCanceller 只支持 16k/32k/48k, 设备 44.1k 需重采样.
_AEC_RATE = 16000

#: live 模式说完话后的观测窗口 (覆盖尾包/回声尾巴).
LIVE_OBSERVE_S = 6.0

#: live 双讲轮开始前的提示留白 (让人类准备开口).
LIVE_DOUBLETALK_WARN_S = 4.0

#: live 默认测试句.
_DEFAULT_SENTENCE = "你好，这是一句测试。听一听，耳朵能不能听见我自己说话。"


# ──────────────────────────────────────────────────────────────────────────
# 信号合成 (offline 模式) —— 造出 far / 回声路径 / near, 喂 AEC 验证
# ──────────────────────────────────────────────────────────────────────────


def _synth_voice(n: int, sr: int, rng: np.random.Generator, *,
                 active: Optional[slice] = None) -> np.ndarray:
    """类语音信号: 带限噪声 (语音频谱倾斜) × 音节 on/off 包络.

    不追求像真人语音, 只求频谱足够丰富 (激发自适应滤波器) 且有说话/停顿结构.
    ``active`` 限定活跃区 —— 双讲实验里让近端人声只在某段时间出现.
    """
    noise = rng.standard_normal(n).astype(np.float32)
    # 一阶低通 (卷积一个衰减核) 近似语音的频谱倾斜.
    kernel = 0.72 ** np.arange(64)
    voice = np.convolve(noise, kernel, mode="same").astype(np.float32)

    env = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        on = int(rng.uniform(0.18, 0.5) * sr)
        off = int(rng.uniform(0.06, 0.28) * sr)
        end = min(pos + on, n)
        env[pos:end] = 1.0
        pos = end + off
    # 矩形包络卷积短窗, 平滑边沿 (避免方波高频泄漏).
    win = np.ones(max(1, int(0.01 * sr)), dtype=np.float32)
    win /= win.sum()
    env = np.convolve(env, win, mode="same").astype(np.float32)
    if active is not None:
        gate = np.zeros(n, dtype=np.float32)
        gate[active] = 1.0
        env *= gate

    voice *= env
    peak = float(np.max(np.abs(voice))) or 1.0
    return (voice / peak).astype(np.float32)


def _room_path(
        far: np.ndarray,
        sr: int,
        *,
        delay_ms: float,
        gain_db: float,
) -> np.ndarray:
    """模拟扬声器→空气→麦克风的声学路径: 纯延迟 + 增益 + 两个早期反射."""
    d = int(delay_ms * sr / 1000)
    g = 10.0 ** (gain_db / 20.0)
    echo = np.zeros_like(far)
    if d < len(far):
        echo[d:] += far[: len(far) - d] * g
    # 早期反射 (模拟房间).
    for r_ms, r_gain in ((12.0, 0.5), (23.0, 0.32), (37.0, 0.18)):
        rd = int(r_ms * sr / 1000)
        if rd < len(far):
            echo[rd:] += far[: len(far) - rd] * g * r_gain
    return echo


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * np.log10(max(rms, 1e-12))


# ──────────────────────────────────────────────────────────────────────────
# offline —— 合成验证: AEC 是否真的消掉回声, stream_delay_ms 对齐是否有用
# ──────────────────────────────────────────────────────────────────────────


def _run_aec(near: np.ndarray, far: np.ndarray, sr: int, stream_delay_ms: int) -> np.ndarray:
    """逐 10ms 帧跑 AEC, 返回处理后的 near (残余)."""
    from pywebrtc_audio import EchoCanceller

    frame = int(sr * FRAME_MS / 1000)
    aec = EchoCanceller(sample_rate=sr, num_channels=1, stream_delay_ms=stream_delay_ms)
    out = np.zeros_like(near)
    for i in range(0, len(near) - frame + 1, frame):
        blk = aec.process(near[i: i + frame], far[i: i + frame])
        out[i: i + frame] = np.asarray(blk, dtype=np.float32)
    return out


def _window_erle(residual: np.ndarray, echo: np.ndarray, sr: int,
                 start_s: float, end_s: float) -> float:
    """一段上的回声抑制量: 输入回声电平 - AEC 输出电平 (dB).

    静音窗口 (远低于全段回声电平) 返回 nan —— 那里没有回声可测, 硬算只会得噪声.
    """
    a, b = int(start_s * sr), int(end_s * sr)
    ew, rw = echo[a:b], residual[a:b]
    if ew.size == 0:
        return float("nan")
    if _rms_db(ew) < _rms_db(echo) - 15.0:
        return float("nan")
    mask = np.abs(ew) > float(np.max(np.abs(ew))) * 0.05
    if not np.any(mask):
        return float("nan")
    return _rms_db(ew[mask]) - _rms_db(rw[mask])


def _mean_erle(residual: np.ndarray, echo: np.ndarray, sr: int,
               start_s: float, end_s: float, win_s: float = 0.05) -> float:
    """一段区间上按窗平均的 ERLE (跳过静音窗)."""
    vals: list[float] = []
    t = start_s
    while t < end_s - win_s:
        v = _window_erle(residual, echo, sr, t, t + win_s)
        if not np.isnan(v):
            vals.append(v)
        t += win_s
    return float(np.mean(vals)) if vals else float("nan")


def _erle_steady(residual: np.ndarray, echo: np.ndarray, sr: int) -> float:
    """稳态 ERLE: 后半段 (避开收敛期) 的平均抑制."""
    half = len(echo) // 2
    return _mean_erle(residual, echo, sr, half / sr, len(echo) / sr)


def _erle_early(residual: np.ndarray, echo: np.ndarray, sr: int) -> float:
    """早期 ERLE: 收敛前段 (0.05s ~ 0.6s) 的平均抑制 —— 反映 hint 质量 / 收敛快慢."""
    return _mean_erle(residual, echo, sr, 0.05, 0.6)


@dataclass
class _OfflineResult:
    label: str
    erle_steady_db: float           # 稳态回声抑制 (dB)
    erle_early_db: float            # 收敛期 (0.2~1.0s) 平均抑制 (dB)
    near_out_db: float = float("nan")   # 双讲: AEC 输出的电平
    near_ref_db: float = float("nan")   # 双讲: 纯近端人声的电平

    @property
    def near_delta_db(self) -> float:
        """输出相对纯人声的偏差. ≈0 = 人声基本无损 (残余回声可忽略)."""
        if np.isnan(self.near_ref_db):
            return float("nan")
        return self.near_out_db - self.near_ref_db


def run_offline(sr: int = 16000, seconds: float = 6.0, delay_ms: float = 60.0,
                gain_db: float = -12.0, seed: int = 7) -> list[_OfflineResult]:
    n = int(sr * seconds)
    rng = np.random.default_rng(seed)

    far = _synth_voice(n, sr, rng)
    echo = _room_path(far, sr, delay_ms=delay_ms, gain_db=gain_db)

    results: list[_OfflineResult] = []

    # 实验 1: 单讲 (near = 纯回声). 对齐质量看 stream_delay_ms 提示对/错/缺三类 ——
    # 早期 ERLE 反映收敛快慢 (hint 质量), 稳态 ERLE 反映 estimator 兜底后的极限.
    for label, hint in (("单讲 / hint=0 (未提示)", 0),
                        (f"单讲 / hint={int(delay_ms)}ms (=真延迟)", int(delay_ms)),
                        ("单讲 / hint=500ms (严重失配)", 500)):
        residual = _run_aec(echo.copy(), far, sr, hint)
        results.append(_OfflineResult(
            label=label,
            erle_steady_db=_erle_steady(residual, echo, sr),
            erle_early_db=_erle_early(residual, echo, sr),
        ))

    # 实验 2: 双讲 (近端人声只在后半段出现). 前半段是纯回声窗口;
    # 后半段额外对比 AEC 输出与纯人声的电平, 看近端是否被误伤.
    half = n // 2
    speech = _synth_voice(n, sr, np.random.default_rng(seed + 1), active=slice(half, n)) * 0.5
    residual_dt = _run_aec(echo + speech, far, sr, int(delay_ms))
    # 回声抑制取前半段 (纯回声) —— 后半段有近端人声, 残余里人声与回声无法分离.
    results.append(_OfflineResult(
        label="双讲 / 近端人声保留",
        erle_steady_db=_mean_erle(residual_dt, echo, sr, 1.5, half / sr),
        erle_early_db=_erle_early(residual_dt, echo, sr),
        near_out_db=_rms_db(residual_dt[half:]),
        near_ref_db=_rms_db(speech[half:]),
    ))

    return results


def _print_offline(results: list[_OfflineResult]) -> None:
    print(f"{'实验':<30} {'稳态ERLE':>9} {'早期ERLE':>9} {'近端偏差':>9}")
    print("-" * 62)
    for r in results:
        delta = "  -  " if np.isnan(r.near_delta_db) else f"{r.near_delta_db:+.1f}dB"
        print(f"{r.label:<30} {r.erle_steady_db:>8.1f}dB {r.erle_early_db:>8.1f}dB {delta:>9}")
    print()
    print("解读:")
    print("  稳态ERLE  = 后半段 (收敛后) 回声被压掉多少 dB, 越大越好.")
    print("  早期ERLE  = 收敛期 (0.2~1.0s) 平均抑制; hint 贴近真延迟时更高 = 收敛更快.")
    print("  近端偏差  = 双讲时 AEC 输出相对纯人声的电平差; ≈0 = 人声保留、残余回声可忽略.")


# ──────────────────────────────────────────────────────────────────────────
# live —— 真实设备: 单进程 capture + player, AEC 处理 near, 看 ASR 是否听到自己
# ──────────────────────────────────────────────────────────────────────────


def run_live(device: Optional[str], sentence: str, *, aec_on: bool = True) -> None:
    """真实设备单进程听说实测 —— 两轮: 单讲 (看回声) + 双讲 (看 ASR 合成).

    链路 (脚本自搭, 不改核心): ``capture → [AEC] → resample → ASR``.
    这正是 ``HostListenerState._audio_gen`` 那条「音频层、ASR 之前」的边界 ——
    AEC 作为门控「拦回声」那一格, 就该插在这个位置.

    AEC 的 far 参考来自 ``player.on_play`` (真实写入设备的帧), near 来自 capture.
    两轮判据:

    - **单讲**: AEC=off 时 ASR 应认出 ghost 自己的话 (回声 = 地面真值);
      on 时应听不到.
    - **双讲**: ghost 说话时人也开口, 看 ASR 合成结果 —— 人的话要被识别,
      ghost 的话应被 AEC 压住 (off 时两者混在一起).

    需要外放, 且人要在设备旁 (双讲轮要配合开口). 戴耳机测到的是「没回声」, 不是
    脚本的问题.
    """
    import asyncio

    from ghoshell_moss.core.blueprint.matrix import Matrix

    matrix = Matrix.new("aec_alignment_probe", category="cli")
    matrix.run(lambda m: _live_main(m, device=device, sentence=sentence, aec_on=aec_on))


async def _live_main(matrix, *, device: Optional[str], sentence: str, aec_on: bool) -> None:
    import asyncio
    import contextlib as _ctx

    from ghoshell_moss.contracts.asr import ASR
    from ghoshell_moss.contracts.audio import (
        AudioCaptureConfig,
        AudioCaptureSource,
        resample,
    )
    from ghoshell_moss.contracts.configs import get_or_create_conf
    from ghoshell_moss.contracts.listener import ASRListener
    from ghoshell_moss.contracts.speech import Speech, TTSSpeech

    con = matrix.container
    if device:
        get_or_create_conf(con, AudioCaptureConfig()).device_pattern = device

    listener = con.get(ASRListener)
    asr = listener.asr() if listener is not None else con.get(ASR)
    if asr is None:
        print("[live] FATAL: no ASR (ASRListener / ASR not provided by IoC)")
        return
    capture = con.get(AudioCaptureSource)
    if capture is None:
        print("[live] FATAL: no AudioCaptureSource")
        return
    speech = con.get(Speech)
    if not isinstance(speech, TTSSpeech):
        print(f"[live] FATAL: Speech is {type(speech).__name__}, not TTSSpeech")
        return

    await speech.start()
    player = speech.player()
    play_rate, cap_rate = player.sample_rate, capture.sample_rate
    asr_rate = asr.get_info().sample_rate
    print(f"[live] player={play_rate}Hz capture={cap_rate}Hz asr={asr_rate}Hz")

    # far 参考: player.on_play 的真实写入帧, 统一到 _AEC_RATE 存放 (AEC3 只支持 16k/32k/48k).
    far_ring: list[np.ndarray] = []

    def _on_play(frame: np.ndarray) -> None:
        arr = np.asarray(frame).reshape(-1)
        if play_rate != _AEC_RATE:
            arr = resample(arr.astype(np.int16), origin_rate=play_rate, target_rate=_AEC_RATE)
        far_ring.append(arr.astype(np.float32))
        total = sum(a.size for a in far_ring)
        while far_ring and total > _AEC_RATE * 2:
            total -= far_ring[0].size
            far_ring.pop(0)

    player.on_play(_on_play)

    aec = None
    if aec_on:
        from pywebrtc_audio import EchoCanceller
        aec = EchoCanceller(sample_rate=_AEC_RATE, num_channels=1, stream_delay_ms=0)
    print(f"[live] AEC={'on' if aec else 'off'}  (far reference = player.on_play @{_AEC_RATE}Hz)")

    await capture.start()
    consumer = capture.new_sequential_consumer()
    await consumer.__aenter__()

    frame = int(_AEC_RATE * FRAME_MS / 1000)
    pending = np.zeros(0, dtype=np.float32)
    say_at = [0.0]

    async def audio_gen():
        nonlocal pending
        async for chunk in consumer:
            samples = np.asarray(chunk.samples).ravel().astype(np.float32)
            if samples.size == 0:
                continue
            # 统一到工作采样率: AEC 用 16k (AEC3 限制), 无 AEC 时直接到 asr 率.
            work_rate = _AEC_RATE if aec is not None else asr_rate
            if cap_rate != work_rate:
                samples = resample(samples.astype(np.int16), origin_rate=cap_rate,
                                   target_rate=work_rate).astype(np.float32)
            if aec is None:
                out = samples
            else:
                pending = np.concatenate([pending, samples])
                blocks: list[np.ndarray] = []
                while pending.size >= frame:
                    blk = pending[:frame]
                    pending = pending[frame:]
                    far = _far_window(far_ring, frame)
                    blocks.append(np.asarray(aec.process(blk, far), dtype=np.float32))
                out = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
            if out.size == 0:
                continue
            if aec is not None and _AEC_RATE != asr_rate:
                out = resample(out.astype(np.int16), origin_rate=_AEC_RATE,
                               target_rate=asr_rate).astype(np.float32)
            yield out.astype(np.int16)

    recognition = asr.recognize(audio_gen())

    async def pump():
        async for ev in recognition:
            dt = (time.monotonic() - say_at[0]) if say_at[0] else 0.0
            sid = ev.segment_id[-4:] if ev.segment_id else "????"
            print(f"  [{ev.phase.value:<7} +{dt:5.2f}s seg={sid}] {ev.text}", flush=True)

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(1.0)  # 让 capture/ASR 链路稳定

    async def _say_round(label: str, hint: str = "") -> None:
        print(f"\n[round:{label}] {hint}", flush=True)
        say_at[0] = time.monotonic()
        stream = speech.new_segment()
        stream.feed(sentence, complete=True)
        print(f'  [ghost say] "{sentence}"', flush=True)
        with _ctx.suppress(Exception):
            await stream.play([])
        await asyncio.sleep(LIVE_OBSERVE_S)

    await _say_round("单讲", "只有 ghost 说话 —— 观察 ASR 有没有听见自己.")
    print(f"\n[double-talk] {LIVE_DOUBLETALK_WARN_S:.0f}s 后 ghost 再说一句 —— "
          f"请你同时开口说任意一句话.", flush=True)
    await asyncio.sleep(LIVE_DOUBLETALK_WARN_S)
    await _say_round("双讲", "ghost 与人同时说 —— 看 ASR 合成结果.")

    consumer.shutdown()
    await recognition.close()  # 关 ASR session task (WS), 否则 event loop 有 pending task 不退
    pump_task.cancel()
    with _ctx.suppress(Exception):
        await pump_task
    await consumer.__aexit__(None, None, None)
    await capture.close()
    with _ctx.suppress(Exception):
        await speech.close()


def _far_window(far_ring: list[np.ndarray], frame: int) -> np.ndarray:
    """取最近写入设备的 far 样本 (单帧长), 不足补零.

    实时流同节拍时, far 的「最近」就是 near 块对应的当前播放参考; 残余延迟由 AEC3
    自带的 delay estimator 吸收 —— 不用外部互相关去 hack 对齐.
    """
    if not far_ring:
        return np.zeros(frame, dtype=np.float32)
    acc = np.concatenate(far_ring)
    if acc.size >= frame:
        return acc[-frame:]
    return np.concatenate([np.zeros(frame - acc.size, dtype=np.float32), acc])


# ──────────────────────────────────────────────────────────────────────────


def _main() -> None:
    argv = sys.argv[1:]
    mode = argv[0] if argv else "offline"
    if mode == "offline":
        results = run_offline()
        _print_offline(results)
    elif mode == "live":
        aec_on = "--no-aec" not in argv
        rest = [a for a in argv[1:] if a != "--no-aec"]
        device = rest[0] if len(rest) > 0 and rest[0] else None
        sentence = rest[1] if len(rest) > 1 else _DEFAULT_SENTENCE
        run_live(device, sentence, aec_on=aec_on)  # run_live 内部 matrix.run 同步阻塞
    else:
        print(__doc__)
        print(f"unknown mode: {mode!r} (use 'offline' or 'live')")
        sys.exit(2)


if __name__ == "__main__":
    _main()
