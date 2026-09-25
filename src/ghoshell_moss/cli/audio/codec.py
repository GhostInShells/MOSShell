"""Audio source material — tune synthesis, WAV codec, fragment slicing.

codec 是"源材料"层——CLI 作为来源按片段喂入, 或从文件读入.
与播放/捕获实现解耦: 只用标准库 + numpy, 不依赖具体播放器.
"""

from __future__ import annotations

import wave as _wave
from pathlib import Path

import numpy as np


def _fragments(pcm: np.ndarray, rate: int, frag_ms: int = 100) -> list[np.ndarray]:
    """来源粒度切片 — CLI 作为来源按片段喂入, 每个片段一个 PlaybackSample.

    player 自己负责重采样与底层帧切分; 这里的切片只为波形观测,
    语义与 TTS 逐片段产出一致 (fragment_id 递增).
    """
    step = max(int(rate * frag_ms / 1000), 1)
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


def _synthesize_tune(seconds: float, rate: int) -> np.ndarray:
    """温和的 C 大调和弦 pad — 基频 + 柔和泛音, 缓慢颤音/振幅起伏, 淡入淡出.

    目标是"不太难听"的体感测试音, 不是旋律. 峰值控制在 0.5, 听感不刺耳.
    """
    notes = [(261.63, 0.30), (329.63, 0.24), (392.00, 0.24), (523.25, 0.12)]
    n = int(seconds * rate)
    t = np.arange(n) / rate
    wave = np.zeros(n)
    for f, a in notes:
        vib = 0.6 * np.sin(2 * np.pi * 0.35 * t)  # 缓慢颤音, 声音"活"一点
        phase = 2 * np.pi * (f + vib) * t
        wave += a * np.sin(phase) + a * 0.25 * np.sin(2 * phase)
    wave *= 0.9 + 0.1 * np.sin(2 * np.pi * 1.1 * t)  # 慢振幅起伏
    attack = max(int(0.02 * rate), 1)
    release = max(int(0.12 * rate), 1)
    if n > attack + release:
        env = np.ones(n)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        env[-release:] *= np.linspace(1.0, 0.0, release)
        wave *= env
    peak = float(np.max(np.abs(wave))) or 1.0
    wave = wave / peak * 0.5
    return (wave * 32767).astype(np.int16)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """读 WAV PCM — 用标准库, 不引入具体播放实现. 多声道下混到单声道."""
    with _wave.open(str(path), "rb") as w:
        params = w.getparams()
        nch, sampwidth, rate, nframes = params[:4]
        raw = w.readframes(nframes)
    if sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) * 256
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        data = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return data, rate


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    """Write int16 PCM data as a WAV file."""
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm.astype(np.int16).tobytes())
