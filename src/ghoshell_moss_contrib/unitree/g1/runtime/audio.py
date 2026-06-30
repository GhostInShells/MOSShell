"""
Audio runtime — G1 内置 TTS + PCM 流播放 + 音量的运行时模块.

跟 asr.py 范式对称: asr 是 "上行感知" (G1 → MOSS), audio 是 "下行表达" (MOSS → G1).
没有 reader 线程, 没有 ring buffer — 所有调用都是同步 RPC 转发, runtime 维护少量
状态 (running flag / 播放结束估算时间 / 错误累计) 供 health() / is_playing() 用.

跟 audio_player.py 的分工:
  - audio_player.py = G1StreamPlayer (StreamAudioPlayer ABC 实现).
    服务 ghoshell_moss.contracts.speech 整套流水线 (SpeechStream + TTS engine + IoC).
    MOSS 端合成 PCM 后, 通过它推给 G1 喇叭.
  - audio.py (本文件) = 模块级单例 + 函数式 API.
    给 channel:audio 命令直接用 — 调 G1 自带 TTS / 放预录音频 / 调音量.
    跳过 MOSS TTS 流水线, 走 G1 内置能力.
  - 两者共享底层 sdk.get_audio_client(), 但互不 import.

物理事实 (来自 SDK 源码 .../g1/audio/g1_audio_client.py + 6-16 实机):
  - TtsMaker(text, speaker_id) — G1 自带 TTS RPC. 立即返回 code, 实际语音异步播放.
    speaker_id 含义待实测 (0 通常是默认音色).
  - PlayStream(app_name, stream_id, pcm) — 推 16kHz mono s16le PCM 给 G1 喇叭.
    同 stream_id 分块拼接, 新 stream_id 抢占旧流 (6-16 验证).
  - PlayStop(app_name) — 中断当前播放.
  - GetVolume() / SetVolume(volume) — 音量范围待实测 (官方文档说 0-9?, _archived 用过 0-100,
    需实机校准, 见 TODO).

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import audio

    bootstrap(nic="eth0")
    audio.start()
    audio.speak("你好, 我是 G1")           # G1 自带 TTS
    audio.play_wav_file("/path/to/x.wav")  # 自己解码 → PlayStream
    audio.cancel()                         # PlayStop
    audio.set_volume(80)
    audio.stop()
"""
from __future__ import annotations

import json
import logging
import threading
import time
import wave
from typing import Optional

from ghoshell_moss_contrib.unitree.g1.sdk import get_audio_client

logger = logging.getLogger("moss.g1.runtime.audio")


# ── 常量 ────────────────────────────────────────────────────────────────
# G1 PlayStream 契约: 16kHz mono s16le. 推其他格式 G1 不会拒, 但出来是噪声.
# 来源: docs/index.md + 6-16 实机验证.

_SAMPLE_RATE = 16000
_CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2  # s16le
_BYTES_PER_SECOND = _SAMPLE_RATE * _CHANNELS * _SAMPLE_WIDTH_BYTES  # 32000


# ── 模块级私有状态 ────────────────────────────────────────────────────────
# runtime/README.md §1: 模块即单例. 所有可变状态在模块级, _state_lock 保护.

_state_lock = threading.Lock()

_running: bool = False
_app_name: str = "moss_audio"  # PlayStream 流空间 — 同 app_name 内 stream_id 抢占
_stream_counter: int = 0
_last_stream_id: str = ""
_last_play_at: float = 0.0
_estimated_end_time: float = 0.0  # last_play_at + PCM 时长. is_playing 据此估算.
_tts_call_count: int = 0
_play_stream_call_count: int = 0
_play_stop_call_count: int = 0
_error_count: int = 0


# ── 公开接口: 生命周期 ───────────────────────────────────────────────────

def start(*, app_name: str = "moss_audio") -> None:
    """进入可用状态. 幂等 — 已运行则直接 return.

    前置: sdk.bootstrap() 已完成 (否则后续 get_audio_client() raise).

    跟 asr.start() 的差异: audio 没有后台 reader 线程, 也无 DDS subscriber 需启动.
    start() 仅 reset 状态计数 + 设 running flag. 保留接口名是为跟 runtime/README.md §3
    标准接口形态对齐, 不为接口本身没事干.

    :param app_name: PlayStream 的 app_name. 同 app_name 内的 stream_id 互相抢占,
      跨 app_name 互不影响. 默认 "moss_audio" — 跟 audio_player.py 的 "moss_tts"
      错开, 避免 MOSS TTS 流水线和本模块的 play_pcm 互踩.
    """
    global _running, _app_name, _stream_counter, _last_stream_id
    global _last_play_at, _estimated_end_time
    global _tts_call_count, _play_stream_call_count, _play_stop_call_count, _error_count

    with _state_lock:
        if _running:
            logger.debug("start() 重入 — 已在运行, 跳过.")
            return
        # 不在锁内调 get_audio_client — RPC 客户端获取应在 bootstrap 之后是常态,
        # 失败时 raise 直接出. 不吞.
        _ = get_audio_client()

        _running = True
        _app_name = app_name
        _stream_counter = 0
        _last_stream_id = ""
        _last_play_at = 0.0
        _estimated_end_time = 0.0
        _tts_call_count = 0
        _play_stream_call_count = 0
        _play_stop_call_count = 0
        _error_count = 0

    logger.info("audio runtime started (app_name=%s).", app_name)


def stop() -> None:
    """退出可用状态. 幂等. 调 PlayStop 中断在播流, 然后清状态.

    不关 AudioClient — sdk 层管 client 生命周期, audio runtime 是消费者.
    """
    global _running

    with _state_lock:
        if not _running:
            logger.debug("stop() 重入 — 未在运行, 跳过.")
            return
        app = _app_name
        _running = False

    # 出锁后做 RPC. PlayStop 失败仅 warn — stop 是清理路径, 不让 G1 残留播放比一切重要.
    try:
        client = get_audio_client()
        client.PlayStop(app)
    except Exception as e:
        logger.warning("audio stop() PlayStop 抛异常 %s (忽略).", e)

    logger.info("audio runtime stopped.")


def is_running() -> bool:
    """当前是否处于可用状态."""
    with _state_lock:
        return _running


# ── 公开接口: 输出能力 ───────────────────────────────────────────────────

def speak(text: str, *, speaker_id: int = 0) -> int:
    """让 G1 用自带 TTS 说一句话. 立即返回 code, 实际语音由 G1 异步播放.

    TtsMaker 是单次 RPC — 不流式, 不分块, 一次性把整段文本扔过去. G1 内部 TTS
    引擎合成后通过自带喇叭播放. MOSS 端不掌握合成时长, 只能粗估 (本实现按
    字符数 × 0.15s 估算, 仅供 is_playing 判断, 不是物理精确值).

    跟 audio_player.py 的差别: audio_player 是 "MOSS 自己合成 PCM 再推过去", 用于
    SpeechStream / 高级 TTS 引擎. 本函数是 "让 G1 用它自己的 TTS", 简单、
    无依赖 MOSS TTS 体系, 但音色/语速不可控.

    :param text: 要说的文本. 空字符串直接返回 0 不调 RPC.
    :param speaker_id: G1 TTS 音色编号. 含义实测中, 默认 0 — TODO 实机校准
      0/1/2 是否对应不同音色.
    :return: TtsMaker 返回的 code. 0 = 成功, 非 0 = 失败 (常见 3104 = 服务暂时不可用,
      7401 = 互斥锁占用).
    """
    global _tts_call_count, _last_play_at, _estimated_end_time, _error_count

    if not text:
        return 0

    with _state_lock:
        if not _running:
            logger.warning("speak() 在 stop 状态调用, 拒绝.")
            return -1
        _tts_call_count += 1

    client = get_audio_client()
    try:
        code = client.TtsMaker(text, speaker_id)
    except Exception:
        with _state_lock:
            _error_count += 1
        logger.exception("audio.speak() TtsMaker 抛异常.")
        return -2

    if code != 0:
        with _state_lock:
            _error_count += 1
        logger.warning("audio.speak() TtsMaker code=%s text=%r", code, text[:40])
        return code

    # 粗估 TTS 时长: 中文 ~5 字/秒, 英文 ~3 词/秒. 一刀切按 0.15s/char.
    # 实测精度不重要 — is_playing 只是给上层 "大概还在说" 的提示, 不是物理同步信号.
    # TODO 实机标定中英文真实播放速率, 也许差距大到要分语言估算.
    now = time.time()
    duration = max(len(text) * 0.15, 0.5)
    with _state_lock:
        _last_play_at = now
        _estimated_end_time = max(_estimated_end_time, now + duration)
    return 0


def play_pcm(pcm: bytes, *, stream_id: Optional[str] = None) -> str:
    """把一段 16kHz mono s16le PCM 推给 G1 喇叭. 返回本次使用的 stream_id.

    流式语义 (6-16 实机验证):
      - 同 stream_id 多次调用 → G1 端无缝拼接.
      - 新 stream_id → 抢占当前流, 老流被丢弃.
    本函数默认每次自动 new 一个 stream_id (抢占旧的). 想 "续播" 同段流则显式传入.

    :param pcm: 原始 PCM 字节. 格式必须是 16kHz mono s16le, 否则出来是噪声.
      偶数长度 (s16le 每样本 2 字节) — 奇数长度会被裁掉最后一字节.
    :param stream_id: 显式指定 stream_id 以续播. None 则 new 一个.
    :return: 本次实际使用的 stream_id (用于后续续播).
    """
    global _stream_counter, _last_stream_id, _last_play_at, _estimated_end_time
    global _play_stream_call_count, _error_count

    if not pcm:
        with _state_lock:
            return _last_stream_id

    # 截尾对齐: s16le 每样本 2 字节, 奇数长度的尾字节会被 G1 解码成噪声.
    if len(pcm) % 2 == 1:
        pcm = pcm[:-1]

    with _state_lock:
        if not _running:
            logger.warning("play_pcm() 在 stop 状态调用, 拒绝.")
            return ""
        if stream_id is None:
            _stream_counter += 1
            stream_id = f"moss_{int(time.time() * 1000)}_{_stream_counter}"
        _last_stream_id = stream_id
        _play_stream_call_count += 1
        app = _app_name

    client = get_audio_client()
    try:
        code, _ = client.PlayStream(app, stream_id, pcm)
    except Exception:
        with _state_lock:
            _error_count += 1
        logger.exception("audio.play_pcm() PlayStream 抛异常.")
        return stream_id

    if code != 0:
        with _state_lock:
            _error_count += 1
        logger.warning(
            "audio.play_pcm() PlayStream code=%s stream_id=%s len=%d",
            code, stream_id, len(pcm),
        )

    # 估算播放结束时间. 若是续播同 stream_id, 拼接到现有 estimated_end_time 之后.
    duration = len(pcm) / _BYTES_PER_SECOND
    now = time.time()
    with _state_lock:
        if stream_id == _last_stream_id and _estimated_end_time > now:
            _estimated_end_time += duration
        else:
            _estimated_end_time = now + duration
        _last_play_at = now

    return stream_id


def play_wav_file(path: str) -> str:
    """读取 wav 文件, 校验格式后推给 G1 喇叭. 返回 stream_id.

    便利包装 — 校验失败 raise ValueError, 让调用方知道是文件格式不对, 不是 RPC 失败.

    :param path: wav 文件路径. 必须是 16kHz mono s16le (G1 PlayStream 契约).
      其他格式不在 runtime 做转换 — 通用重采样属于 BaseAudioStreamPlayer.resample,
      本模块不重复实现.
    :return: stream_id.
    :raises ValueError: 文件格式不匹配契约时.
    :raises FileNotFoundError / wave.Error: 文件不存在或非合法 wav.
    """
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        ch = f.getnchannels()
        sw = f.getsampwidth()
        if (sr, ch, sw) != (_SAMPLE_RATE, _CHANNELS, _SAMPLE_WIDTH_BYTES):
            raise ValueError(
                f"wav 格式不匹配 G1 PlayStream 契约: "
                f"expected 16kHz/mono/s16le, got {sr}Hz/{ch}ch/{sw*8}bit. "
                f"文件: {path}"
            )
        pcm = f.readframes(f.getnframes())
    return play_pcm(pcm)


def cancel() -> None:
    """中断当前播放. PlayStop RPC. 幂等 — 没播也调一次 OK.

    cancel 是 PCM 流 + TTS 共用的中断路径 (实测: PlayStop 能停 PlayStream 推的流,
    但能否中断 TtsMaker 合成中的语音待 6-30 后实机验证 — 大概率 TTS 不可中断,
    因为 G1 端 TTS 走的是独立播放通道, 跟 PlayStream 抢占语义可能不一样).
    TODO 实机标定 TtsMaker 是否能被 PlayStop 中断.
    """
    global _play_stop_call_count, _estimated_end_time, _error_count

    with _state_lock:
        if not _running:
            logger.debug("cancel() 在 stop 状态调用, 跳过.")
            return
        app = _app_name
        _play_stop_call_count += 1

    client = get_audio_client()
    try:
        client.PlayStop(app)
    except Exception:
        with _state_lock:
            _error_count += 1
        logger.exception("audio.cancel() PlayStop 抛异常.")
        return

    with _state_lock:
        _estimated_end_time = time.time()


def is_playing() -> bool:
    """估算: 当前是否还在播放. 基于上次推流的 PCM 时长 + TTS 字符数粗估.

    **不是物理精确信号** — G1 不上行 "播放结束" 事件, runtime 端只能算账.
    场景使用: channel 命令 "audio:speak text=..." 完成后, 上层想知道 "G1 还在说吗?
    我能不能现在再 speak / 切话题", is_playing 给个大致答案.

    若 G1 实际播放比估算长 (TTS 慢、长文本), is_playing 会过早返回 False;
    反之亦然. 精度敏感场景应该靠 cancel + 重启序列, 而不是依赖 is_playing.
    """
    with _state_lock:
        return time.time() < _estimated_end_time


# ── 公开接口: 音量 ────────────────────────────────────────────────────────

def get_volume() -> Optional[int]:
    """读 G1 当前喇叭音量. 失败返回 None.

    范围待实测 (TODO): 文档语焉不详, _archived 老代码按 0-100, 但 SDK 没强制校验.
    第一次调用时把返回值打 log, 实机校准后回填 docstring.
    """
    global _error_count

    with _state_lock:
        if not _running:
            return None

    client = get_audio_client()
    try:
        code, data = client.GetVolume()
    except Exception:
        with _state_lock:
            _error_count += 1
        logger.exception("audio.get_volume() 抛异常.")
        return None

    if code != 0 or data is None:
        logger.warning("audio.get_volume() code=%s data=%s", code, data)
        return None

    # SDK 返回 json.loads(data), 实际结构待实测. 容错: 拿 volume 字段, 没有就 None.
    if isinstance(data, dict):
        v = data.get("volume")
        if isinstance(v, int):
            return v
    logger.warning("audio.get_volume() 返回格式未识别: %r", data)
    return None


def set_volume(volume: int) -> int:
    """设 G1 喇叭音量. 范围待实测 (见 get_volume).

    :return: SetVolume RPC 的 code. 0 = 成功.
    """
    global _error_count

    with _state_lock:
        if not _running:
            return -1

    client = get_audio_client()
    try:
        return client.SetVolume(volume)
    except Exception:
        with _state_lock:
            _error_count += 1
        logger.exception("audio.set_volume() 抛异常.")
        return -2


# ── 公开接口: 健康检查 ────────────────────────────────────────────────────

def health() -> dict:
    """暴露 runtime 内部状态. 供 monitor / channel debug / 实机校验用."""
    with _state_lock:
        now = time.time()
        return {
            "running": _running,
            "app_name": _app_name,
            "is_playing_est": now < _estimated_end_time,
            "estimated_remaining_sec": max(0.0, _estimated_end_time - now),
            "last_stream_id": _last_stream_id,
            "stream_counter": _stream_counter,
            "last_play_at": _last_play_at,
            "tts_call_count": _tts_call_count,
            "play_stream_call_count": _play_stream_call_count,
            "play_stop_call_count": _play_stop_call_count,
            "error_count": _error_count,
        }
