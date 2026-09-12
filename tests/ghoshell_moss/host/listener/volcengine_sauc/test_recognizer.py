"""volcengine_sauc recognizer — on_event_creating 分派行为契约.

验证 facade 声明的两种回调形态: awaitable 回调 inline await (阻塞消费点),
sync 回调 to_thread 卸载 (非阻塞并行), 回调异常被兜住不中断.
"""
import numpy as np
import pytest

from ghoshell_moss.contracts.asr import RecognitionEvent, RecognitionPhase
from ghoshell_moss.host.listener.volcengine_sauc import VolcengineSaucASR, VolcengineSaucConfig


async def _empty_audio():
    if False:
        yield np.array([], dtype=np.int16)


def _event() -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s1", segment_id="g1",
        phase=RecognitionPhase.CLAUSE, text="你好",
    )


def test_get_info_maps_vad_end_window():
    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    asr.configure({"end_window_size": 1200})
    assert asr.get_info().vad_end_window_ms == 1200


@pytest.mark.asyncio
async def test_awaitable_callback_is_awaited():
    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    stream = asr.recognize(_empty_audio())
    seen = []

    async def cb(event):
        seen.append(event.phase)

    stream.on_event_creating(cb)
    await stream.dispatch_event_creating(_event())
    assert seen == [RecognitionPhase.CLAUSE]


@pytest.mark.asyncio
async def test_sync_callback_is_offloaded():
    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    stream = asr.recognize(_empty_audio())
    seen = []

    def cb(event):
        seen.append(event.phase)

    stream.on_event_creating(cb)
    await stream.dispatch_event_creating(_event())
    assert seen == [RecognitionPhase.CLAUSE]


@pytest.mark.asyncio
async def test_callback_exception_is_contained():
    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    stream = asr.recognize(_empty_audio())

    def bad(event):
        raise RuntimeError("boom")

    stream.on_event_creating(bad)
    await stream.dispatch_event_creating(_event())  # 不抛出
