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

from ghoshell_moss.contracts.asr import (
    ASRWithCorpus,
    Corpus,
    RecognitionEvent,
    RecognitionPhase,
    RecognitionSegment,
)
from ghoshell_moss.contracts.audio import AudioChunk, AudioFrameMeta
from ghoshell_moss.host.listener.volcengine_sauc import (
    VolcengineSaucASR,
    VolcengineSaucConfig,
    VolcengineSaucCorpus,
)
from ghoshell_moss.host.listener.volcengine_sauc import recognizer as sauc_recognizer
from ghoshell_moss.host.listener.volcengine_sauc.protocol import create_init_request

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


# ── ASRWithCorpus (可降级的 corpus 能力面) ──


def _decode_init(data: bytes) -> dict:
    """解 init 帧 (header4 + seq4 + len4 + gzip(json)) → {audio, request}."""
    return json.loads(gzip.decompress(data[12:]).decode("utf-8"))


def _context_from_init(data: bytes) -> dict | None:
    """从 init 帧取出 request.context (JSON 字符串) 并反序列化; 无则 None."""
    request = _decode_init(data)["request"]
    return json.loads(request["context"]) if "context" in request else None


@pytest.mark.asyncio
async def test_clause_carries_word_confidence(monkeypatch):
    """词级置信度: utterance.words[].conf 落进 clause.words (RecognitionWord)."""
    frames = [
        _server_frame({"result": {
            "text": "你好",
            "utterances": [
                {"text": "你好", "definite": True, "start_time": 0, "end_time": 100,
                 "words": [{"text": "你", "conf": 0.9}, {"text": "好", "conf": 0.3}]},
            ],
        }}),
        _server_frame({"result": {"text": "你好"}}, is_last=True),
    ]
    monkeypatch.setattr(sauc_recognizer, "connect", _connect_to(_FakeWS(frames)))

    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    stream = asr.recognize(_audio(np.zeros(1600, dtype=np.int16)))
    events = [e async for e in stream]

    clause = [e.clause for e in events if e.phase == RecognitionPhase.CLAUSE][0]
    assert [(w.text, w.conf) for w in clause.words] == [("你", 0.9), ("好", 0.3)]


def test_asr_is_corpus_capable():
    """火山 ASR 实现 ASRWithCorpus 能力面 — 泛型层靠 isinstance 判能力/降级."""
    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    assert isinstance(asr, ASRWithCorpus)


def test_corpus_tail_keeps_instruction_and_newest_lines():
    """Corpus.tail: instruction 永不参与截断, lines 只留最新的 max_lines 条, 空白不追加."""
    c = Corpus(instruction="提示", max_lines=2)
    c.tail("一")
    c.tail("二")
    c.tail("三")
    assert c.instruction == "提示"
    assert c.lines == ["二", "三"]
    c.tail("   ")
    assert c.lines == ["二", "三"]


def test_init_request_composes_instruction_first_and_lines_newest_first():
    """协议适配: instruction 打头, lines 从新到旧; 热词走厂商配置, 独立于条件文本."""
    config = VolcengineSaucConfig()
    boosting = VolcengineSaucCorpus(hotwords=["豆包"])
    context = Corpus(instruction="我是提示词", lines=["旧话", "新话"])

    data = create_init_request("u", config, boosting=boosting, context=context)
    ctx = _context_from_init(data)

    assert ctx["hotwords"] == [{"word": "豆包"}]
    assert ctx["context_type"] == "dialog_ctx"
    assert ctx["context_data"] == [
        {"text": "我是提示词"},
        {"text": "新话"},
        {"text": "旧话"},
    ]


def test_init_request_drops_oldest_lines_over_budget():
    """token 预算裁剪: instruction 优先, lines 从最旧开始丢."""
    context = Corpus(instruction="提示", lines=[f"line-{i:03d}-" + "x" * 40 for i in range(100)])
    data = create_init_request("u", VolcengineSaucConfig(), context=context)
    ctx = _context_from_init(data)
    # instruction 永在首位; lines 只保留预算内最靠前的 (最旧的被丢).
    assert ctx["context_data"][0] == {"text": "提示"}
    entries = ctx["context_data"][1:]
    assert len(entries) < 100
    kept = [int(e["text"].split("-")[1]) for e in entries]
    assert min(kept) > 0  # 最旧 (0) 被丢
    assert max(kept) == 99  # 最新 (99) 保留


@pytest.mark.asyncio
async def test_corpus_write_takes_effect_next_segment(monkeypatch):
    """corpus 不按流冻结: 一个 turn 内改 instruction, 下一个 segment 的 init 就带新值."""
    is_last = _server_frame({"result": {}}, is_last=True)
    wss: list[_FakeWS] = []

    async def _connect(config, request_id: str = ""):
        ws = _FakeWS([is_last])
        wss.append(ws)
        return ws

    monkeypatch.setattr(sauc_recognizer, "connect", _connect)

    asr = VolcengineSaucASR(config=VolcengineSaucConfig())
    loud = AudioChunk(
        samples=np.zeros(160, dtype=np.int16),
        meta=AudioFrameMeta(rms_db=-20.0, is_silent=False),
    )
    holder: dict = {}

    async def _audio():
        yield loud
        # turn1 内 (send loop 正从 generator 拉下一块): 改 corpus + commit 结束本 turn.
        asr.set_corpus_instruction("新提示词")
        holder["stream"].commit()
        yield loud
        yield loud  # turn2 的 first chunk; 之后耗尽.

    asr.set_corpus_instruction("初始提示词")
    stream = asr.recognize(_audio())
    holder["stream"] = stream

    [e async for e in stream]

    assert len(wss) == 2
    first = _context_from_init(wss[0].sent[0])
    second = _context_from_init(wss[1].sent[0])
    assert first["context_data"] == [{"text": "初始提示词"}]
    assert second["context_data"] == [{"text": "新提示词"}]
