"""Audio capability probing CLI — organized by protocol probe layer.

每个子模块探测音频协议的一层:
  - contracts.py — IoC 注册面 (哪个 provider 支撑每个核心音频抽象)
  - devices.py   — 设备面 (输入/输出端点枚举)
  - play.py      — 纯输出体感, 无协议 (tune/wav 播放)
  - speak.py     — speak 协议: TTS 片段 (text+audio, 以 stream_id 为维度)
                    + 播放完成样本 (fragment_id 对齐)
  - capture.py   — capture 协议: 音频聆听片段
  - echo.py      — capture(1) 音频片段 + playback(3) 播放完成样本的组合
  - asr.py       — capture 协议: 音频片段 + ASR 结果

共享基建:
  - render.py    — PlaybackSample 观测显示 (频谱/波形/实时帧渲染)
  - codec.py     — 音频源材料 (tune 合成, WAV codec, 片段切片)

协议数据载体在 ghoshell_moss.contracts (PlaybackSample, ASRResult) 与
ghoshell_moss.topics (AudioPlaybackTopic, SpeechTopic, AudioRuntimeTopic) —
未来 matrix 级别广播承载的正是同一批结构. CLI 是协议最早的那个探测面.
"""

from __future__ import annotations

import typer

audio_app = typer.Typer(
    help="Audio capability probing — capture, playback, TTS, ASR.",
    no_args_is_help=True,
)

from . import asr, capture, contracts, devices, echo, play, speak  # noqa: E402
