"""NullASR 表面契约 — 降级耳朵: 空转消费音频, 不产任何 recognition event."""

import numpy as np
import pytest

from ghoshell_moss.contracts.audio import AudioChunk
from ghoshell_moss.core.asr import NullASR


@pytest.mark.asyncio
async def test_null_asr_drains_audio_produces_no_events():
    asr = NullASR()
    drained: list[int] = []

    async def gen():
        for i in range(3):
            drained.append(i)
            yield AudioChunk(seq=i, timestamp=float(i), samples=np.zeros(16, dtype=np.int16))

    events = []
    async for ev in asr.recognize(gen()):
        events.append(ev)

    assert events == []          # 不产 event
    assert drained == [0, 1, 2]  # 音频被空转消费


def test_null_asr_info_is_default():
    info = NullASR().get_info()
    assert info.sample_rate == 16000
    assert info.channel == 1
