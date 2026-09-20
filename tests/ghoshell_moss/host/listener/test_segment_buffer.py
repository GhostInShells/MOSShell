"""SegmentBuffer — 围绕 segment 的可增长 + 可拉读槽位契约.

锚定四条行为:
- 增长全文 full-replace (asr 修正覆盖), peek_current 拉当前.
- segment 签发定稿入历史, 重置 current; TAIL / 迟到事件不污染 current.
- 环形溢出计数 forgotten (模型知道上下文有 gap); resize 缩容同样计入.
- 空 segment 不入历史; clause 分解随定稿保留.
"""
import time

from ghoshell_moss.contracts.asr import (
    RecognitionClause,
    RecognitionEvent,
    RecognitionPhase,
    RecognitionSegment,
)
from ghoshell_moss.host.listener.segment_buffer import SegmentBuffer


def _event(phase: RecognitionPhase, text: str, segment_id: str = "g") -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s", segment_id=segment_id, phase=phase, text=text,
    )


def _segment(text: str, *, segment_id: str = "g", clauses: list[RecognitionClause] | None = None) -> RecognitionSegment:
    return RecognitionSegment(
        id=segment_id, stream_id="s", text=text,
        clauses=clauses or [], created=time.time(),
    )


def test_grows_current_text_full_replace():
    buf = SegmentBuffer()
    buf.on_event(_event(RecognitionPhase.FIRST, "你"))
    assert buf.peek_current().text == "你"

    buf.on_event(_event(RecognitionPhase.PARTIAL, "你好"))
    assert buf.peek_current().text == "你好"  # full-replace, 不是拼接

    buf.on_event(_event(RecognitionPhase.CLAUSE, "你好世界"))
    assert buf.peek_current().text == "你好世界"


def test_peek_current_none_before_any_event():
    buf = SegmentBuffer()
    assert buf.peek_current() is None


def test_segment_finalizes_into_recent_and_resets_current():
    buf = SegmentBuffer()
    buf.on_event(_event(RecognitionPhase.PARTIAL, "你好"))
    buf.on_segment(_segment("你好", segment_id="g"))

    assert buf.peek_current() is None  # 定稿后 current 清空
    recent = buf.peek_recent()
    assert len(recent) == 1
    assert recent[0].text == "你好"
    assert recent[0].segment_id == "g"


def test_tail_event_does_not_reopen_current():
    buf = SegmentBuffer()
    buf.on_event(_event(RecognitionPhase.PARTIAL, "你好"))
    buf.on_segment(_segment("你好", segment_id="g"))
    buf.on_event(_event(RecognitionPhase.TAIL, "你好", segment_id="g"))

    assert buf.peek_current() is None  # TAIL 是切段标记, 不更新 current


def test_stale_event_after_finalize_ignored():
    """on_segment 可能先于迟到的 text event 触发 (pump 落后), 迟到事件不重开 current."""
    buf = SegmentBuffer()
    buf.on_segment(_segment("你好", segment_id="g"))
    buf.on_event(_event(RecognitionPhase.CLAUSE, "你好", segment_id="g"))

    assert buf.peek_current() is None


def test_ring_overflow_counts_forgotten():
    buf = SegmentBuffer(history=2)
    for i, text in enumerate(["a", "b", "c"]):
        buf.on_event(_event(RecognitionPhase.PARTIAL, text, segment_id=f"g{i}"))
        buf.on_segment(_segment(text, segment_id=f"g{i}"))

    assert buf.forgotten() == 1  # "a" 被挤出
    recent = buf.peek_recent()
    assert [s.text for s in recent] == ["b", "c"]


def test_resize_shrinks_and_counts_forgotten():
    buf = SegmentBuffer(history=4)
    for i, text in enumerate(["a", "b", "c", "d"]):
        buf.on_segment(_segment(text, segment_id=f"g{i}"))

    buf.resize(2)
    assert buf.forgotten() == 2  # "a", "b" 缩容挤出
    assert [s.text for s in buf.peek_recent()] == ["c", "d"]


def test_empty_segment_not_recorded():
    buf = SegmentBuffer()
    buf.on_segment(_segment("", segment_id="g"))
    assert buf.peek_recent() == []


def test_segment_keeps_clause_breakdown():
    buf = SegmentBuffer()
    clauses = [RecognitionClause(text="第一句"), RecognitionClause(text="第二句")]
    buf.on_segment(_segment("第一句第二句", segment_id="g", clauses=clauses))

    recent = buf.peek_recent()
    assert [c.text for c in recent[0].clauses] == ["第一句", "第二句"]


def test_peek_recent_tail_n():
    buf = SegmentBuffer(history=8)
    for i, text in enumerate(["a", "b", "c"]):
        buf.on_segment(_segment(text, segment_id=f"g{i}"))

    assert [s.text for s in buf.peek_recent(2)] == ["b", "c"]
    assert len(buf.peek_recent()) == 3  # n=None → 全部


def test_to_dict_serializes_clause_text_only():
    buf = SegmentBuffer()
    clauses = [RecognitionClause(text="第一句"), RecognitionClause(text="第二句")]
    buf.on_segment(_segment("第一句第二句", segment_id="g", clauses=clauses))

    d = buf.peek_recent(1)[0].to_dict()
    assert d["segment_id"] == "g"
    assert d["text"] == "第一句第二句"
    assert d["clauses"] == ["第一句", "第二句"]  # 只留 clause text, 不含音频轴时间戳
