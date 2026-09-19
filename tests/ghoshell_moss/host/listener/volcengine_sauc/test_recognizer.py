"""volcengine_sauc recognizer — on_event_creating 分派行为契约 + segment 归档契约.

验证 facade 声明的两种回调形态: awaitable 回调 inline await (阻塞消费点),
sync 回调 to_thread 卸载 (非阻塞并行), 回调异常被兜住不中断.
另验证 audio axis 归档: spec segment 带本段吐出的 clause 与 created 墙钟.
"""
import asyncio
import gzip
import json
import struct
import time

import numpy as np
import pytest
import websockets

from ghoshell_moss.contracts.asr import RecognitionEvent, RecognitionPhase, RecognitionSegment
from ghoshell_moss.contracts.audio import AudioChunk, AudioFrameMeta
from ghoshell_moss.host.listener.volcengine_sauc import VolcengineSaucASR, VolcengineSaucConfig
from ghoshell_moss.host.listener.volcengine_sauc import recognizer as sauc_recognizer

# 与 protocol._Protocol 对齐的最小常量 (替身侧只用到这两个).
_FULL_SERVER_RESPONSE = 0x09
_NEG_WITH_SEQUENCE = 0x03


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


# ── segment 归档 (audio axis 带 clause + created) ──


def _server_frame(payload: dict, *, is_last: bool = False) -> bytes:
    """服务端 full_server_response 帧 (JSON + GZIP), 布局对齐 protocol.parse_response."""
    body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    header = bytes([0x11, (_FULL_SERVER_RESPONSE << 4) | (0x02 if is_last else 0x00), 0x11, 0x00])
    return header + struct.pack(">I", len(body)) + body


class _FakeWS:
    """最小 WS 替身: send 记录并识别尾包 (负序号), recv 等尾包发完再吐预置响应帧.

    真实服务端只在收到尾包后才回响应, 这里复刻该因果 —— 否则 recv 会在 send loop
    置 ``_input_done`` 之前吐完, 会话会多跑一个 turn.
    """

    def __init__(self, frames: list[bytes]):
        self._frames = list(frames)
        self._tail_sent = asyncio.Event()
        self.sent: list[bytes] = []

    async def send(self, data: bytes) -> None:
        self.sent.append(data)
        if (data[1] & 0x0F) == _NEG_WITH_SEQUENCE:
            self._tail_sent.set()

    async def recv(self) -> bytes:
        await self._tail_sent.wait()
        if self._frames:
            return self._frames.pop(0)
        raise websockets.exceptions.ConnectionClosed(None, None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


def _connect_to(ws: _FakeWS):
    async def _connect(config, request_id: str = ""):
        return ws
    return _connect


async def _audio(*chunks: np.ndarray):
    for chunk in chunks:
        yield AudioChunk(samples=chunk)


async def _chunks(*chunks: AudioChunk):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_segment_archives_the_clauses_it_emitted(monkeypatch):
    """audio axis 归档带 clause 细节: 段携带本段吐出的 clause (text + timing), 不只在 event 里."""
    started = time.time()
    frames = [
        _server_frame({"result": {
            "text": "你好",
            "utterances": [
                {"text": "你好", "definite": True, "start_time": 100, "end_time": 600},
            ],
        }}),
        _server_frame({"result": {"text": "你好"}}, is_last=True),
    ]
    monkeypatch.setattr(sauc_recognizer, "connect", _connect_to(_FakeWS(frames)))

    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    stream = asr.recognize(_audio(np.zeros(1600, dtype=np.int16)))
    segments: list[RecognitionSegment] = []
    stream.on_segment(segments.append)

    events = [e async for e in stream]

    clauses = [e.clause for e in events if e.phase == RecognitionPhase.CLAUSE]
    assert [c.text for c in clauses] == ["你好"]
    assert len(segments) == 1
    assert [c.text for c in segments[0].clauses] == ["你好"]
    assert [(c.start_ms, c.end_ms) for c in segments[0].clauses] == [(100, 600)]

    # created 是解析时刻的墙钟: event / clause / segment 都在本次运行期间打上.
    assert all(e.created >= started for e in events)
    assert clauses[0].created >= started
    assert segments[0].created >= started


@pytest.mark.asyncio
async def test_clause_frame_emits_no_trailing_partial(monkeypatch):
    """发 clause 的帧不再发 partial: definite + 非 definite 同帧时, 只发 clause."""
    frames = [
        _server_frame({"result": {
            "text": "你好世界",
            "utterances": [
                {"text": "你好", "definite": True, "start_time": 100, "end_time": 600},
                {"text": "你好世界", "definite": False},
            ],
        }}),
        _server_frame({"result": {"text": "你好世界"}}, is_last=True),
    ]
    monkeypatch.setattr(sauc_recognizer, "connect", _connect_to(_FakeWS(frames)))

    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    stream = asr.recognize(_audio(np.zeros(1600, dtype=np.int16)))

    events = [e async for e in stream]
    phases = [e.phase for e in events]
    assert phases.count(RecognitionPhase.CLAUSE) == 1
    assert RecognitionPhase.PARTIAL not in phases


# ── 输入门控 (gate_factory → Callable[[AudioChunk], AudioChunk | None]) ──


@pytest.mark.asyncio
async def test_silence_gate_blocks_silent_stream(monkeypatch):
    """默认门控: 纯静音流被拦, 不 init 不开 WS."""
    connected: list[str] = []

    async def _connect(config, request_id: str = ""):
        connected.append(request_id)
        return _FakeWS([])

    monkeypatch.setattr(sauc_recognizer, "connect", _connect)

    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    silent = AudioChunk(
        samples=np.zeros(160, dtype=np.int16),
        meta=AudioFrameMeta(rms_db=-96.0, is_silent=True),
    )
    stream = asr.recognize(_chunks(silent))

    events = [e async for e in stream]
    assert events == []
    assert connected == []


@pytest.mark.asyncio
async def test_gate_drops_silent_then_releases_on_voice(monkeypatch):
    """门控: 静音帧丢弃不 init, 首个非静音帧放行并 init."""
    frames = [
        _server_frame({"result": {
            "text": "你好",
            "utterances": [{"text": "你好", "definite": True, "start_time": 0, "end_time": 100}],
        }}),
        _server_frame({"result": {"text": "你好"}}, is_last=True),
    ]
    ws = _FakeWS(frames)
    monkeypatch.setattr(sauc_recognizer, "connect", _connect_to(ws))

    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    silent = AudioChunk(
        samples=np.zeros(160, dtype=np.int16),
        meta=AudioFrameMeta(rms_db=-96.0, is_silent=True),
    )
    loud = AudioChunk(
        samples=np.zeros(160, dtype=np.int16),
        meta=AudioFrameMeta(rms_db=-20.0, is_silent=False),
    )

    stream = asr.recognize(_chunks(silent, loud))  # 默认门控
    events = [e async for e in stream]

    clauses = [e for e in events if e.phase == RecognitionPhase.CLAUSE]
    assert [c.clause.text for c in clauses] == ["你好"]
