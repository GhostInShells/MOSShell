"""Stream-level playback perceptibility — observe() on StreamAudioPlayer.

Tests the contract promise: a global observer fires with a PlaybackSample
at actual play time, carrying raw PCM bytes + stream_id + fragment_id + text
that the sender passed — the splicing identity a consumer (e.g. speech_storage)
uses to align callbacks, and the text of the audio actually played.

Stream lifecycle is NOT owned by the player (it belongs to the governance
layer); the observer is a plain subscription with an unsubscribe handle.
"""
import asyncio

import numpy as np
import pytest

from ghoshell_moss.contracts.speech import AudioFormat, PlaybackSample
from ghoshell_moss.host.speech.player import VirtualStreamPlayer


def _make_sine(duration: float, sample_rate: int, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * (32767 * amplitude)).astype(np.int16)


@pytest.mark.asyncio
async def test_observe_fires_playback_sample_on_actual_play():
    """注册观察者后, add 的片段真正写入设备时回调 PlaybackSample."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    samples: list[PlaybackSample] = []
    player.observe(samples.append)

    player.add(
        _make_sine(0.05, 44100),
        audio_type=AudioFormat.PCM_S16LE,
        rate=44100,
        stream_id="stream-a",
        fragment_id="3",
    )
    await player.wait_play_done(timeout=2.0)

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, PlaybackSample)
    assert sample.stream_id == "stream-a"
    assert sample.fragment_id == "3"
    assert sample.duration == pytest.approx(0.05, abs=0.01)
    # 原始 PCM bytes + 响度摘要.
    assert len(sample.pcm) > 0
    assert sample.sample_rate == 44100
    assert sample.rms_db < 0.0  # 正弦波 rms < 0dBFS
    assert sample.peak > 0.0

    await player.close()


@pytest.mark.asyncio
async def test_observe_carries_splicing_identity():
    """PlaybackSample 携带发送方传入的 stream_id + fragment_id — 拼接身份."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    samples: list[PlaybackSample] = []
    player.observe(samples.append)

    # 两个 stream 交错, fragment_id 逐段自增 — 消费方据此拼接.
    for i in range(2):
        player.add(
            _make_sine(0.02, 44100),
            audio_type=AudioFormat.PCM_S16LE,
            rate=44100,
            stream_id="stream-a",
            fragment_id=str(i),
        )
        player.add(
            _make_sine(0.02, 44100),
            audio_type=AudioFormat.PCM_S16LE,
            rate=44100,
            stream_id="stream-b",
            fragment_id=str(i),
        )
    await player.wait_play_done(timeout=2.0)

    ids = [(s.stream_id, s.fragment_id) for s in samples]
    assert ("stream-a", "0") in ids
    assert ("stream-b", "1") in ids

    await player.close()


@pytest.mark.asyncio
async def test_observe_carries_text():
    """PlaybackSample 携带 add 传入的 text — 片段自解释真实播放文本."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    samples: list[PlaybackSample] = []
    player.observe(samples.append)

    player.add(
        _make_sine(0.02, 44100),
        audio_type=AudioFormat.PCM_S16LE,
        rate=44100,
        stream_id="stream-a",
        fragment_id="0",
        text="hello world",
    )
    await player.wait_play_done(timeout=2.0)

    assert len(samples) == 1
    assert samples[0].stream_id == "stream-a"
    assert samples[0].fragment_id == "0"
    assert samples[0].text == "hello world"

    await player.close()


@pytest.mark.asyncio
async def test_observe_global_fires_for_all_streams():
    """观察者是全局的 — 所有 stream 的片段都触发 (stream 身份在数据里, 不在注册里)."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    samples: list[PlaybackSample] = []
    player.observe(samples.append)

    player.add(
        _make_sine(0.02, 44100),
        audio_type=AudioFormat.PCM_S16LE,
        rate=44100,
        stream_id="stream-a",
    )
    player.add(
        _make_sine(0.02, 44100),
        audio_type=AudioFormat.PCM_S16LE,
        rate=44100,
        stream_id="stream-b",
    )
    await player.wait_play_done(timeout=2.0)

    assert len(samples) == 2
    assert {s.stream_id for s in samples} == {"stream-a", "stream-b"}

    await player.close()


@pytest.mark.asyncio
async def test_observe_unsubscribe():
    """unsubscribe 函数可移除观察者 — 之后片段不再触发回调."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    samples: list[PlaybackSample] = []
    unsubscribe = player.observe(samples.append)
    unsubscribe()

    player.add(
        _make_sine(0.02, 44100),
        audio_type=AudioFormat.PCM_S16LE,
        rate=44100,
        stream_id="stream-a",
    )
    await player.wait_play_done(timeout=2.0)

    assert samples == []

    await player.close()


@pytest.mark.asyncio
async def test_no_observer_does_not_crash():
    """无观察者时 add/wait 一切正常 — 跳过 pcm 拷贝与计算, 无副作用."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    player.add(
        _make_sine(0.02, 44100),
        audio_type=AudioFormat.PCM_S16LE,
        rate=44100,
        stream_id="stream-a",
        fragment_id="0",
    )
    done = await player.wait_play_done(timeout=2.0)
    assert done

    await player.close()


@pytest.mark.asyncio
async def test_multiple_observers_all_fire():
    """多个观察者各自收到样本."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    a: list[PlaybackSample] = []
    b: list[PlaybackSample] = []
    player.observe(a.append)
    player.observe(b.append)

    player.add(
        _make_sine(0.02, 44100),
        audio_type=AudioFormat.PCM_S16LE,
        rate=44100,
        stream_id="stream-a",
    )
    await player.wait_play_done(timeout=2.0)

    assert len(a) == 1
    assert len(b) == 1
    assert a[0].stream_id == b[0].stream_id == "stream-a"

    await player.close()
