"""Audio CLI — capability probing for audio capture, playback, TTS, ASR.

CLI 以 cli 身份声明为 matrix node (Matrix.new), 从容器查询音频抽象的注册 provider.
只调 get_provider 反映注册可用性, 不实例化实现 — 不触发重 import, 无副作用.
仅构造容器, 不 join 网络.
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave as _wave
from pathlib import Path
from typing import Optional, Type

import numpy as np
import typer

from ghoshell_container import INSTANCE

from ghoshell_moss.cli.utils import (
    echo,
    is_ai_mode,
    print_error,
    print_info,
    print_simple_panel,
    print_simple_table,
    print_success,
    print_warning,
)
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.asr import ASR
from ghoshell_moss.contracts.speech import AudioFormat, PlaybackSample, Speech, StreamAudioPlayer, TTS, TTSSpeech
from ghoshell_moss.topics.audio import AudioPlaybackTopic

audio_app = typer.Typer(
    help="Audio capability probing — capture, playback, TTS, ASR.",
    no_args_is_help=True,
)

# 核心音频抽象槽位 + 未注册时的原因说明.
_SLOTS: list[tuple[str, Type[INSTANCE], str]] = [
    ("tts", TTS, ""),
    ("speech", Speech, ""),
    ("player", StreamAudioPlayer, ""),
    ("capture", AudioCaptureSource, "in mode HOST layer only — not in project container"),
    ("asr", ASR, "no provider registered"),
]


@audio_app.command("contracts")
def contracts() -> None:
    """List the IoC provider backing each core audio abstraction (no instantiation)."""
    matrix = Matrix.new("audio_cli", category="cli")
    con = matrix.container

    rows = []
    for slot, abstract, note in _SLOTS:
        try:
            provider = con.get_provider(abstract)
        except Exception as e:
            rows.append([slot, abstract.__name__, "—", f"get_provider error: {type(e).__name__} {e}"])
            continue
        if provider is None:
            rows.append([slot, abstract.__name__, "—", note or "no provider registered"])
        else:
            rows.append([slot, abstract.__name__, type(provider).__name__, "OK"])

    print_simple_table(
        data=rows,
        headers=["Slot", "Contract", "Provider", "Status"],
        title="audio contracts",
    )
    echo("")
    print_info(
        "Result is scoped to the active mode (`moss --mode <name>`). "
        "Use `moss manifests providers` for the full provider view."
    )


# --- play — 标准 tune / wav 播放, 验证体感, 支持中断与波形 ---

_WAVE_BARS = "▁▂▃▄▅▆▇█"


def _sparkline(samples, max_chars: int = 80) -> str:
    """把 PlaybackSample 的 rms_db 映射成文本波形条. 过长时按宽度采样."""
    if not samples:
        return "(no samples)"
    dbs = [s.rms_db for s in samples]
    if len(dbs) > max_chars:
        idx = np.linspace(0, len(dbs) - 1, max_chars).astype(int)
        dbs = [dbs[i] for i in idx]
    out = []
    for db in dbs:
        level = int((max(-60.0, min(0.0, db)) + 60.0) / 60.0 * (len(_WAVE_BARS) - 0.01))
        out.append(_WAVE_BARS[level])
    return "".join(out)


def _render_frame(topic, first: bool = False) -> bool:
    """Render spectrum as horizontal bars — one row per frequency bin.

    Each row: [freq_label] [bar] dB_value.
    X axis = intensity (dB), Y axis = frequency (Hz, low→high top→bottom).
    """
    bins = topic.spectrum_bins
    if not bins:
        return first

    n_bins = len(bins)
    nyquist = topic.sample_rate / 2 if topic.sample_rate else 22050
    bin_hz = nyquist / n_bins  # Hz per bin

    bar_width = 40
    n_rows = n_bins

    if not first:
        sys.stdout.write(f"\033[{n_rows + 1}F")

    for i, db in enumerate(bins):
        freq = i * bin_hz
        if freq >= 1000:
            label = f"{freq / 1000:.1f}k".rjust(5)
        else:
            label = f"{freq:.0f}Hz".rjust(5)
        width = int((max(-60.0, min(0.0, db)) + 60.0) / 60.0 * bar_width)
        width = max(1, min(bar_width, width))
        bar = "█" * width + "░" * (bar_width - width)
        sys.stdout.write(f"\r {label} {bar} {db:+.1f}dB\n")

    sys.stdout.write(f"\r       peak {topic.peak:.2f}  rms {topic.rms_db:+.1f}dB\n")
    sys.stdout.flush()
    return False


async def _render_from_queue(queue: asyncio.Queue, first_frame_timeout: float = 2.0) -> bool:
    """Pull PlaybackSample from local queue, compute spectrum, render in real-time.

    Observer now fires at playback rate (BaseAudioStreamPlayer._wait_consumed),
    so frames arrive spaced by chunk duration — no burst, no polling hack needed.
    """
    first = True
    started = time.monotonic()
    rendered = False

    while True:
        try:
            sample = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if not rendered and time.monotonic() - started > first_frame_timeout:
                return False
            continue

        rendered = True
        bins = _spectrum_bins(sample, n_bins=16)
        frame = AudioPlaybackTopic(
            sample_rate=sample.sample_rate,
            rms_db=sample.rms_db,
            peak=sample.peak,
            spectrum_bins=bins,
            n_spectrum_bins=16,
        )
        first = _render_frame(frame, first=first)


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


def _report_playback_sample(observed, total: float) -> None:
    """报告实际播放可感知样本. 单次 add 通常 1 帧; --ai 模式最多一行."""
    if not observed:
        if is_ai_mode():
            echo(f"played {total:.2f}s — no playback samples observed")
        else:
            print_info(f"played {total:.2f}s — no playback samples observed")
        return
    s = observed[0]
    pcm_sz = len(s.pcm) if s.pcm else 0
    if is_ai_mode():
        echo(
            f"played {total:.2f}s — sample: stream={s.stream_id or '-'} fragment={s.fragment_id or '-'} "
            f"rms={s.rms_db:.1f}dB peak={s.peak:.3f} pcm={pcm_sz}B @{s.sample_rate}Hz"
        )
        return
    from rich.text import Text

    t = Text()
    t.append(f"stream={s.stream_id or '-'}  fragment={s.fragment_id or '-'}\n", style="dim")
    t.append(f"duration={s.duration:.2f}s  rms={s.rms_db:.1f}dB  peak={s.peak:.3f}\n")
    bar_w = 28
    lvl = int((max(-60.0, min(0.0, s.rms_db)) + 60.0) / 60.0 * bar_w)
    t.append("rms  " + "█" * lvl + "░" * (bar_w - lvl) + "\n", style="yellow")
    t.append(f"pcm={pcm_sz}B @{s.sample_rate}Hz", style="dim")
    print_simple_panel(t, title="playback sample")


def _spectrum_bins(sample, n_bins: int = 10) -> list[float]:
    """从 PlaybackSample.pcm 做 FFT, 返回 n_bins 个频段能量 (dB).

    消费方自行选择 bin 数——10 是频谱谱面, 20 可画波浪线.
    """
    if not sample.pcm:
        return [-96.0] * n_bins
    pcm = np.frombuffer(sample.pcm, dtype=np.int16).astype(np.float64) / 32768.0
    fft = np.abs(np.fft.rfft(pcm))
    n_fft = len(fft)
    if n_fft < n_bins * 2:
        return [float(20.0 * np.log10(max(fft.mean(), 1e-10)))] * n_bins
    bins = []
    for i in range(n_bins):
        lo = int(i * n_fft / n_bins)
        hi = int((i + 1) * n_fft / n_bins)
        db = 20.0 * np.log10(max(float(fft[lo:hi].mean()), 1e-10))
        bins.append(round(db, 1))
    return bins


def _render_spectrogram(observed, n_bins: int = 10, max_rows: int = 40) -> str:
    """N 个频段的文本频谱谱面 — 每行一个片段, 堆叠即"跳跃的波浪线".

    返回一个 Text 对象, human 模式 rich Panel 输出; --ai 模式 echo 纯文本.
    """
    if not observed:
        return "(no samples)"
    bars = "▁▂▃▄▅▆▇█"
    rows = []
    for s in observed[-max_rows:]:
        bins = _spectrum_bins(s, n_bins=n_bins)
        row = "".join(bars[max(0, min(7, int((db + 60.0) / 60.0 * 7.99)))] for db in bins)
        rows.append(row)
    return "\n".join(rows)


def _report_spectrogram(observed, total: float) -> None:
    """渲染频谱谱面 — human 模式 rich 面板, --ai 模式纯文本行."""
    spectro = _render_spectrogram(observed, n_bins=10)
    if is_ai_mode():
        echo(spectro)
        if observed:
            dbs = [s.rms_db for s in observed]
            echo(f"fragments: {len(observed)}  rms range=[{min(dbs):.1f}, {max(dbs):.1f}] dB")
        return
    from rich.text import Text

    t = Text()
    t.append(f"fragments: {len(observed)}  ", style="dim")
    t.append(f"duration: {total:.2f}s\n")
    t.append(spectro, style="yellow")
    print_simple_panel(t, title="playback spectrogram")


@audio_app.command("play")
def play(
    seconds: float = typer.Option(3.0, "--seconds", "-s", help="Standard tune duration in seconds (ignored when --file is given)."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Play a WAV file instead of the standard tune."),
    pub_topic: bool = typer.Option(False, "--pub-topic", hidden=True, help="Publish AudioPlaybackTopic via Zenoh for remote consumers."),
) -> None:
    """Play a standard tune (or a WAV file) to test audible playback feel. Ctrl+C to interrupt.

    Human 模式: 播放中实时频谱柱状图. --ai 模式: 单帧样本摘要.
    """
    matrix = Matrix.new("audio_play", category="cli")
    result = matrix.run(lambda m: _async_play(m, seconds=seconds, file=file, pub_topic=pub_topic))

    if result is None:
        return
    source, total, observed, interrupted, had_realtime = result
    if interrupted:
        print_warning("interrupted — playback stopped")
    else:
        print_success(f"playback finished: {source} {total:.2f}s")
    if had_realtime:
        return  # 已在播放期间实时渲染
    if is_ai_mode():
        _report_playback_sample(observed, total)
    else:
        _report_spectrogram(observed, total)


async def _async_play(matrix, *, seconds: float, file: Optional[Path], pub_topic: bool = False):
    """收集播放数据; 本地 queue 桥接实时渲染, --pub-topic 时额外走 Zenoh 广播."""
    player = matrix.container.get(StreamAudioPlayer)
    if player is None:
        print_error("StreamAudioPlayer not registered — run `moss audio contracts` to see what's available.")
        return None

    if file is not None:
        if not file.exists():
            print_error(f"file not found: {file}")
            return None
        pcm, rate = _read_wav(file)
        source = str(file)
    else:
        rate = 44100
        pcm = _synthesize_tune(seconds, rate)
        source = "tune"
    if len(pcm) == 0:
        print_warning("no audio to play")
        return None
    total = len(pcm) / rate

    await player.start()

    if pub_topic:
        player.enable_playback_topic()

    stream_id = f"cli-{int(time.time())}"
    observed: list = []
    interrupted = False
    had_realtime = False

    async def _feed():
        if is_ai_mode():
            player.add(
                pcm,
                audio_type=AudioFormat.PCM_S16LE,
                rate=rate,
                stream_id=stream_id,
                fragment_id="0",
            )
        else:
            for i, frag in enumerate(_fragments(pcm, rate)):
                player.add(
                    frag,
                    audio_type=AudioFormat.PCM_S16LE,
                    rate=rate,
                    stream_id=stream_id,
                    fragment_id=str(i),
                )
        await player.wait_play_done()

    try:
        if not is_ai_mode():
            # 本地桥: observer → call_soon_threadsafe → asyncio.Queue → render task
            # observer 现在以播放速率触发 (_wait_consumed), 帧按 chunk 间隔到达
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)
            try:
                feed_task = asyncio.create_task(_feed())
                render_task = asyncio.create_task(_render_from_queue(frame_q))
                done, pending = await asyncio.wait(
                    [feed_task, render_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if feed_task in done:
                    render_task.cancel()
                    had_realtime = True
                    if (exc := feed_task.exception()) is not None:
                        raise exc
                else:
                    await feed_task
            finally:
                unsub()
        else:
            await _feed()
    except asyncio.CancelledError:
        interrupted = True
        await player.clear()
    finally:
        await player.close()

    return source, total, observed, interrupted, had_realtime


# --- speak — TTS synthesis -> player playback, with optional WAV save ---


@audio_app.command("speak")
def speak(
    text: str = typer.Argument(..., help="Text to synthesize and play."),
    tone: Optional[str] = typer.Option(None, "--tone", "-t", help="TTS voice/tone preset."),
    save: Optional[Path] = typer.Option(None, "--save", "-o", help="Save synthesized audio to a WAV file."),
) -> None:
    """Synthesize text to speech and play through audio output.

    Prefers Speech (TTSSpeech) provider to test the full TTS -> Player wiring.
    Falls back to direct TTS + Player pipeline when Speech is not available
    or when --save is given (which needs direct access to audio chunks).
    """
    matrix = Matrix.new("audio_speak", category="cli")
    result = matrix.run(lambda m: _async_speak(m, text=text, tone=tone, save=save))
    if result is None:
        return
    source, total, observed, interrupted, had_realtime = result
    if interrupted:
        print_warning("interrupted — playback stopped")
    else:
        snippet = text[:50] + ("..." if len(text) > 50 else "")
        print_success(f"speak finished: \"{snippet}\" — {source} {total:.2f}s")
    if had_realtime:
        return
    if is_ai_mode():
        _report_playback_sample(observed, total)
    else:
        _report_spectrogram(observed, total)


async def _async_speak(matrix, *, text: str, tone: Optional[str], save: Optional[Path]):
    """Dispatch to Speech path (preferred) or direct TTS + Player path."""
    con = matrix.container

    player = con.get(StreamAudioPlayer)
    if player is None:
        print_error("StreamAudioPlayer not registered — run `moss audio contracts` to check.")
        return None

    speech = con.get(Speech)
    tts = con.get(TTS)

    # --save needs direct access to audio chunks; bypass Speech
    if save is not None or not isinstance(speech, TTSSpeech):
        if tts is None:
            print_error("TTS not registered — cannot synthesize speech.")
            return None
        return await _speak_direct(matrix, tts, player, text, tone, save)

    return await _speak_via_speech(matrix, speech, text, tone)


async def _speak_via_speech(matrix, speech, text: str, tone: Optional[str]):
    """Full TTS -> Player pipeline via Speech (TTSSpeech) wiring."""
    tts = speech.tts()
    player = speech.player()
    tts_info = tts.get_info()

    if tone:
        tts.use_tone(tone)

    print_info(
        f"TTS: {type(tts).__name__}  voice: {tts.current_tone()}  "
        f"rate: {tts_info.sample_rate}Hz  channels: {tts_info.channels}"
    )

    await speech.start()

    stream = speech.new_stream()
    stream.feed(text, complete=True)

    interrupted = False
    had_realtime = False
    observed: list = []
    _total = [0.0]

    def _collect(sample):
        observed.append(sample)
        _total[0] += sample.duration

    try:
        if not is_ai_mode():
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                _collect(sample)
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)
            try:
                say_task = asyncio.create_task(stream.say())
                render_task = asyncio.create_task(_render_from_queue(frame_q))
                done, pending = await asyncio.wait(
                    [say_task, render_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if say_task in done:
                    render_task.cancel()
                    had_realtime = True
                    if (exc := say_task.exception()) is not None:
                        raise exc
                else:
                    await say_task
            finally:
                unsub()
        else:
            unsub = player.observe(_collect)
            try:
                await stream.say()
            finally:
                unsub()
    except asyncio.CancelledError:
        interrupted = True
        await player.clear()
    finally:
        await speech.close()

    return f"speech:{type(tts).__name__}", _total[0], observed, interrupted, had_realtime


async def _speak_direct(matrix, tts, player, text: str, tone: Optional[str], save: Optional[Path]):
    """Direct TTS -> Player pipeline, with optional WAV save."""
    tts_info = tts.get_info()

    if tone:
        tts.use_tone(tone)

    audio_format = (
        AudioFormat(tts_info.audio_format)
        if isinstance(tts_info.audio_format, str)
        else tts_info.audio_format
    )

    print_info(
        f"TTS: {type(tts).__name__}  voice: {tts.current_tone()}  "
        f"rate: {tts_info.sample_rate}Hz  channels: {tts_info.channels}"
    )

    await tts.start()
    await player.start()

    batch = tts.new_batch(batch_id=f"cli-speak-{int(time.monotonic() * 1e9)}")
    batch.feed(text)
    batch.commit()
    await batch.start()

    stream_id = f"cli-speak-{int(time.time())}"
    all_audio: list = [] if save else None
    _total = [0.0]
    _fragment_id = [0]

    interrupted = False
    had_realtime = False
    observed: list = []

    def _collect(sample):
        observed.append(sample)

    try:
        if not is_ai_mode():
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                _collect(sample)
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)

            async def _feed():
                async for item in batch.items():
                    audio = item["audio"]
                    player.add(
                        audio,
                        audio_type=audio_format,
                        rate=tts_info.sample_rate,
                        channels=tts_info.channels,
                        stream_id=stream_id,
                        fragment_id=str(_fragment_id[0]),
                    )
                    if all_audio is not None:
                        all_audio.append(audio.copy())
                    _fragment_id[0] += 1
                await player.wait_play_done()
                _total[0] = sum(s.duration for s in observed)

            try:
                feed_task = asyncio.create_task(_feed())
                render_task = asyncio.create_task(_render_from_queue(frame_q))
                done, pending = await asyncio.wait(
                    [feed_task, render_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if feed_task in done:
                    render_task.cancel()
                    had_realtime = True
                    if (exc := feed_task.exception()) is not None:
                        raise exc
                else:
                    await feed_task
            finally:
                unsub()
        else:
            unsub = player.observe(_collect)
            try:
                async for item in batch.items():
                    audio = item["audio"]
                    player.add(
                        audio,
                        audio_type=audio_format,
                        rate=tts_info.sample_rate,
                        channels=tts_info.channels,
                        stream_id=stream_id,
                        fragment_id=str(_fragment_id[0]),
                    )
                    if all_audio is not None:
                        all_audio.append(audio.copy())
                    _fragment_id[0] += 1
                await player.wait_play_done()
                _total[0] = sum(s.duration for s in observed)
            finally:
                unsub()
    except asyncio.CancelledError:
        interrupted = True
        await player.clear()
    finally:
        await tts.close()
        await player.close()

    if all_audio is not None and save:
        combined = np.concatenate(all_audio)
        _write_wav(save, combined, tts_info.sample_rate, tts_info.channels)
        print_success(f"saved {len(combined) / tts_info.sample_rate:.2f}s audio to {save}")

    return f"tts:{type(tts).__name__}", _total[0], observed, interrupted, had_realtime


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    """Write int16 PCM data as a WAV file."""
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm.astype(np.int16).tobytes())


# --- device — list audio devices ---


@audio_app.command("device")
def device() -> None:
    """List available audio input and output devices."""
    try:
        from ghoshell_moss.depends import depend_host
        depend_host()
        import miniaudio
    except Exception:
        print_error("miniaudio not available — install ghoshell-moss[host]")
        return

    try:
        devs = miniaudio.Devices()
    except Exception as e:
        print_error(f"failed to enumerate devices: {e}")
        return

    rows = []
    for d in devs.get_captures():
        ch = max(f["channels"] for f in d["formats"]) if d["formats"] else "?"
        rows.append(["input", d["name"], str(ch)])
    for d in devs.get_playbacks():
        ch = max(f["channels"] for f in d["formats"]) if d["formats"] else "?"
        rows.append(["output", d["name"], str(ch)])

    if not rows:
        print_warning("no audio devices found")
        return

    print_simple_table(rows, headers=["Type", "Name", "Max Ch"], title="audio devices")


# --- capture — record N seconds, show waveform, optionally save ---


@audio_app.command("capture")
def capture(
    seconds: float = typer.Option(3.0, "--seconds", "-s", help="Capture duration in seconds."),
    save: Optional[Path] = typer.Option(None, "--save", "-o", help="Save captured audio to WAV file."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Capture device name pattern (empty for default)."),
) -> None:
    """Capture audio for N seconds, show waveform, optionally save to WAV."""
    matrix = Matrix.new("audio_capture", category="cli")
    result = matrix.run(lambda m: _async_capture(m, seconds=seconds, save=save, device=device))
    if result is None:
        return
    pcm, rate, total, interrupted = result
    if interrupted:
        print_warning("capture interrupted")
    else:
        print_success(f"captured {total:.2f}s @{rate}Hz")
    _report_capture_spectrogram(pcm, rate, total)


async def _async_capture(matrix, *, seconds: float, save: Optional[Path], device: Optional[str]):
    """Capture audio from the default input device."""
    con = matrix.container

    capture_source = con.get(AudioCaptureSource)
    if capture_source is None:
        print_error("AudioCaptureSource not registered")
        return None

    if device is not None:
        capture_source._config.device_pattern = device

    await capture_source.start()

    if "not started" in capture_source.device_explain():
        print_error("capture device not started — may be locked by another process")
        await capture_source.close()
        return None

    print_info(f"capturing {seconds}s from {capture_source.device_explain()}...")

    consumer = capture_source.new_sequential_consumer(max_queue_frames=256)
    await consumer.start()

    all_audio: list = []
    sample_rate = capture_source._config.sample_rate
    channels = capture_source._config.channels
    interrupted = False

    try:
        deadline = time.monotonic() + seconds
        async for chunk in consumer:
            all_audio.append(chunk.samples.copy())
            if time.monotonic() >= deadline:
                break
    except asyncio.CancelledError:
        interrupted = True
    finally:
        await consumer.close()
        await capture_source.close()

    if not all_audio:
        print_warning("no audio captured — is the device working?")
        return None

    combined = np.concatenate(all_audio)
    total = len(combined) / sample_rate

    if save:
        _write_wav(save, combined, sample_rate, channels)
        print_success(f"saved {total:.2f}s to {save}")

    return combined, sample_rate, total, interrupted


def _report_capture_spectrogram(pcm: np.ndarray, rate: int, total: float) -> None:
    """Show spectrogram of captured audio — reuse existing rendering."""
    frags = _fragments(pcm, rate, frag_ms=100)
    observed = []
    for frag in frags:
        f32 = frag.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(f32 ** 2)))
        rms_db = 20.0 * np.log10(max(rms, 1e-10))
        peak = float(np.max(np.abs(f32)))
        observed.append(PlaybackSample(
            pcm=frag.tobytes(),
            duration=len(frag) / rate,
            sample_rate=rate,
            rms_db=round(rms_db, 1),
            peak=round(peak, 3),
        ))
    if is_ai_mode():
        spectro = _render_spectrogram(observed, n_bins=10)
        echo(spectro)
        dbs = [s.rms_db for s in observed]
        echo(f"fragments: {len(observed)}  rms range=[{min(dbs):.1f}, {max(dbs):.1f}] dB")
    else:
        _report_spectrogram(observed, total)


# --- echo — capture N seconds, then play back with real-time spectrum ---


@audio_app.command("echo")
def echo_cmd(
    seconds: float = typer.Option(3.0, "--seconds", "-s", help="Capture duration in seconds."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Capture device name pattern (empty for default)."),
) -> None:
    """Capture audio then play it back immediately with real-time spectrum."""
    matrix = Matrix.new("audio_echo", category="cli")
    result = matrix.run(lambda m: _async_echo(m, seconds=seconds, device=device))
    if result is None:
        return
    total, interrupted = result
    if interrupted:
        print_warning("echo interrupted")
    else:
        print_success(f"echo finished: {total:.2f}s")


async def _async_echo(matrix, *, seconds: float, device: Optional[str]):
    """Capture audio then play back through StreamAudioPlayer."""
    cap_result = await _async_capture(matrix, seconds=seconds, save=None, device=device)
    if cap_result is None:
        return None
    pcm, rate, cap_total, interrupted = cap_result

    if interrupted:
        return cap_total, True

    player = matrix.container.get(StreamAudioPlayer)
    if player is None:
        print_error("StreamAudioPlayer not registered")
        return None

    await player.start()

    stream_id = f"cli-echo-{int(time.time())}"
    echo_interrupted = False

    try:
        if not is_ai_mode():
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)

            async def _feed():
                for i, frag in enumerate(_fragments(pcm, rate)):
                    player.add(
                        frag,
                        audio_type=AudioFormat.PCM_S16LE,
                        rate=rate,
                        stream_id=stream_id,
                        fragment_id=str(i),
                    )
                await player.wait_play_done()

            try:
                feed_task = asyncio.create_task(_feed())
                render_task = asyncio.create_task(_render_from_queue(frame_q))
                done, pending = await asyncio.wait(
                    [feed_task, render_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if feed_task in done:
                    render_task.cancel()
                    if (exc := feed_task.exception()) is not None:
                        raise exc
                else:
                    await feed_task
            finally:
                unsub()
        else:
            player.add(
                pcm,
                audio_type=AudioFormat.PCM_S16LE,
                rate=rate,
                stream_id=stream_id,
                fragment_id="0",
            )
            await player.wait_play_done()
    except asyncio.CancelledError:
        echo_interrupted = True
        await player.clear()
    finally:
        await player.close()

    return cap_total, echo_interrupted
