"""Segment buffer — 围绕 segment 的可增长 + 可拉读文本槽位.

语音流里除了 signal (push) 之外, 补一个模型任何时候都能 peek 的持久槽位:

- ``current``: 当前 segment 的增长全文 (FIRST/PARTIAL/CLAUSE 全量覆盖累积).
- ``recent``:  最近 n 轮已定稿的 segment (segment 签发 = commit/TAIL 切段).
- ``forgotten``: 环形溢出被挤掉的计数 (模型知道上下文有 gap).

跨 session 存续 (controller 级, 对称 g1 listener 的模块级 ``_finalized_dq``).
生命周期边界 = segment 签发, 不是 ASR VAD final — 这是与 g1 的关键差异.

只落模型可读的 text + clauses + 时间戳, 不落原始 audio (模型读不了, 也省内存).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ghoshell_moss.contracts.asr import (
    RecognitionClause,
    RecognitionEvent,
    RecognitionPhase,
    RecognitionSegment,
)

__all__ = ["HeardSegment", "SegmentBuffer"]


@dataclass
class HeardSegment:
    """一段 segment 的轻量文本视图 (定稿或正在长).

    ``text`` 是全量累计文本; ``clauses`` 是稳定分句分解 (定稿后才有, 增长阶段为空).
    """

    segment_id: str
    text: str
    clauses: list[RecognitionClause] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """channel 拉取时的 JSON 视图 (clause 只留 text, 不含音频轴时间戳)."""
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "clauses": [c.text for c in self.clauses],
            "created": self.created,
        }


class SegmentBuffer:
    """segment 槽位 — 增长全文 + 最近 n 轮 + 溢出计数, 纯内存, 拉模式读.

    ``on_event`` (text axis) 更新增长全文; ``on_segment`` (segment 签发) 定稿入历史.
    两条回调可能跑在不同 asyncio task (pump vs receive loop), ``on_segment`` 可能先于
    迟到的 text event 触发 — 靠 ``_last_finalized_id`` 拦掉已定稿 segment 的迟到事件.
    """

    def __init__(self, *, history: int = 8) -> None:
        self._history = max(1, history)
        self._recent: deque[HeardSegment] = deque(maxlen=self._history)
        self._current_segment_id: str | None = None
        self._current_text: str = ""
        self._last_finalized_id: str | None = None
        self._forgotten: int = 0

    def resize(self, history: int) -> None:
        """改环形容量 (active 礼仪切换时 history 可能变). 缩容挤掉的计入 forgotten."""
        history = max(1, history)
        if history == self._history:
            return
        overflow = max(0, len(self._recent) - history)
        self._recent = deque(self._recent, maxlen=history)
        self._forgotten += overflow
        self._history = history

    def on_event(self, event: RecognitionEvent) -> None:
        """text axis: 更新当前 segment 的增长全文 (full-replace).

        TAIL 是切段标记 (增长文本由 ``on_segment`` 定稿), 不更新 current;
        已定稿 segment 的迟到事件 (pump 落后于 on_segment) 同样忽略.
        """
        if event.phase == RecognitionPhase.TAIL:
            return
        if event.segment_id == self._last_finalized_id:
            return
        self._current_segment_id = event.segment_id
        self._current_text = event.text

    def on_segment(self, segment: RecognitionSegment) -> None:
        """segment 签发 (切段): 定稿当前 → 压入历史, 重置 current."""
        if not segment.text and not self._current_text:
            return  # 空 segment 不入历史
        heard = HeardSegment(
            segment_id=segment.id,
            text=segment.text or self._current_text,
            clauses=list(segment.clauses),
            created=segment.created,
        )
        if len(self._recent) == self._history:
            self._forgotten += 1
        self._recent.append(heard)
        self._last_finalized_id = segment.id
        self._current_segment_id = None
        self._current_text = ""

    def peek_current(self) -> HeardSegment | None:
        """当前正在长的 (未定稿) 增长全文; 无活跃 segment 则 None."""
        if self._current_segment_id is None:
            return None
        return HeardSegment(
            segment_id=self._current_segment_id,
            text=self._current_text,
        )

    def peek_recent(self, n: int | None = None) -> list[HeardSegment]:
        """最近 n 轮已定稿 segment (tail-n); n=None 返回全部."""
        items = list(self._recent)
        if n is None or n <= 0:
            return items
        return items[-n:]

    def forgotten(self) -> int:
        """环形溢出累计被挤掉的 segment 数."""
        return self._forgotten
