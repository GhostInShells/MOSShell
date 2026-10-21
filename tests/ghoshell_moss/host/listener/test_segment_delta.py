"""SegmentDelta — the delta rules the conversation etiquette depends on.

These tests pin the one invariant that makes per-clause delivery safe: **no packet ever
repeats text an earlier packet delivered**. Everything else (which phase emits, where a
segment resets) is in service of that.
"""

from __future__ import annotations

import pytest

from ghoshell_moss.contracts.asr import RecognitionClause, RecognitionEvent, RecognitionPhase
from ghoshell_moss.host.listener.segment_delta import SegmentDelta


def _event(
        phase: RecognitionPhase,
        text: str,
        clause: str | None = None,
        segment_id: str = "seg-1",
) -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="stream-1",
        segment_id=segment_id,
        phase=phase,
        text=text,
        clause=RecognitionClause(text=clause) if clause is not None else None,
    )


def _clause(text: str, clause: str, segment_id: str = "seg-1") -> RecognitionEvent:
    return _event(RecognitionPhase.CLAUSE, text, clause=clause, segment_id=segment_id)


def _tail(text: str, segment_id: str = "seg-1") -> RecognitionEvent:
    return _event(RecognitionPhase.TAIL, text, segment_id=segment_id)


class TestClausePackets:
    def test_each_clause_delivers_only_what_is_new(self):
        """两句分句: 第二包只带新的那一句, 前文不再出现."""
        delta = SegmentDelta()

        first = delta.feed(_clause("你好，我是老莫", "你好"))
        second = delta.feed(_clause("你好，我是老莫。今天吃什么", "今天吃什么"))

        assert [p.delta for p in first] == ["你好，我是老莫"]
        assert [p.delta for p in second] == ["。今天吃什么"]
        assert first[0].unit == "clause" and first[0].final is False
        assert second[0].index > first[0].index

    def test_first_and_partial_are_not_boundaries(self):
        """FIRST/PARTIAL 不产包 —— 识别器清嗓子不是模型要听的边界."""
        delta = SegmentDelta()

        assert delta.feed(_event(RecognitionPhase.FIRST, "你好")) == []
        assert delta.feed(_event(RecognitionPhase.PARTIAL, "你好，我")) == []
        assert delta.feed(_event(RecognitionPhase.PARTIAL, "你好，我是")) == []

        packets = delta.feed(_clause("你好，我是", "你好"))
        assert len(packets) == 1
        assert packets[0].delta == "你好，我是", "PARTIAL 没发过的字必须随第一个分句补上"

    def test_a_repeated_clause_emits_nothing(self):
        """同一个分句重复到达 (同段同文) 不再发 —— 重复是噪声, 不是新信息."""
        delta = SegmentDelta()

        assert len(delta.feed(_clause("你好", "你好"))) == 1
        assert delta.feed(_clause("你好", "你好")) == []


class TestSegmentPacket:
    def test_tail_never_repeats_delivered_clauses(self):
        """尾包不许包含已经发过的分句 —— 这就是「怎么聊都改不对」那个病根."""
        delta = SegmentDelta()
        delta.feed(_clause("你好，我是老莫", "你好"))
        delta.feed(_clause("你好，我是老莫。今天吃什么", "今天吃什么"))

        packets = delta.feed(_tail("你好，我是老莫。今天吃什么"))

        assert packets == [], "尾包没有新内容时, 正确的产物是沉默而不是重发全文"

    def test_tail_carries_only_the_part_no_clause_delivered(self):
        """尾包只带没发过的那一截."""
        delta = SegmentDelta()
        delta.feed(_clause("你好，我是老莫", "你好"))

        packets = delta.feed(_tail("你好，我是老莫。今天吃什么"))

        assert len(packets) == 1
        packet = packets[0]
        assert packet.unit == "segment" and packet.final is True
        assert packet.delta == "。今天吃什么"
        assert "你好" not in packet.delta


class TestRevision:
    def test_a_correction_of_delivered_text_is_flagged_and_carried(self):
        """尾包修订了已经发出去的前文: 要标记出来, 并带上更正后的全文."""
        delta = SegmentDelta()
        delta.feed(_clause("你好，我是老莫。", "你好"))
        delta.feed(_clause("你好，我是老莫。今天吃什么", "今天吃什么"))

        packets = delta.feed(_tail("您好，我是老莫。今天吃什么"))

        assert len(packets) == 1
        packet = packets[0]
        assert packet.revised is True, "前文被改过, 消费方必须知道"
        assert packet.text == "您好，我是老莫。今天吃什么", "修正后的全文才是真身"
        assert packet.delta == "您好，我是老莫。今天吃什么", "差异从第一个不同字开始, 不复述相同前缀"

    def test_delta_is_cut_by_prefix_not_by_length(self):
        """按公共前缀切, 不按长度相减 —— 长度相减会在修订后切出半句.

        已经发出去 8 个字, 修订后全文 13 个字: 从第 8 个字开始切会得到「天吃什么」这种残句,
        正确的结果是从第一个不同字开始的完整一段.
        """
        delta = SegmentDelta()
        delta.feed(_clause("你好，我是老莫。", "你好"))
        delta.feed(_clause("你好，我是老莫。今天", "今天"))

        packets = delta.feed(_tail("您好，我是老莫。今天吃什么"))

        assert packets[0].delta == "您好，我是老莫。今天吃什么"

    def test_a_packet_that_changes_nothing_is_silent(self):
        """全文没变就不发 —— 每发一次都等于叫醒模型一次."""
        delta = SegmentDelta()
        delta.feed(_clause("你好", "你好"))
        assert delta.feed(_tail("你好")) == []


class TestSegmentBoundary:
    def test_a_new_segment_does_not_carry_the_old_cursor(self):
        """换段之后从头算: 新段的第一句就是它的全部, 不跟上一段比."""
        delta = SegmentDelta()
        delta.feed(_clause("你好，我是老莫", "你好", segment_id="seg-1"))

        packets = delta.feed(_clause("吃了吗", "吃了吗", segment_id="seg-2"))

        assert len(packets) == 1
        assert packets[0].delta == "吃了吗"
        assert packets[0].segment_id == "seg-2"

    def test_index_restarts_per_segment(self):
        """序号在一段之内递增, 换段归零 —— 消费方靠它排序, 不靠它跨段对齐."""
        delta = SegmentDelta()
        first = delta.feed(_clause("你好", "你好", segment_id="seg-1"))[0]
        delta.feed(_clause("你好。再说一句", "再说一句", segment_id="seg-1"))

        second = delta.feed(_clause("新的段", "新的段", segment_id="seg-2"))[0]

        assert first.index == 1
        assert second.index == 1
