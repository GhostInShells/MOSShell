"""StopJudge — 智能判停状态机组件的行为测试.

验证 judge 把 recognition 事件流翻译成 commit 决策: clause 打分、activity cancel、
keyword 显式终点、clause 累积成多 content block. mock caller 驱动, 不碰真模型.
"""
import asyncio

import pytest

from ghoshell_moss.contracts.asr import RecognitionClause, RecognitionEvent, RecognitionPhase
from ghoshell_moss.host.listener.stop_judge import StopJudge, parse_stop_score
from ghoshell_moss.message import Message


class _FakeResult:
    def __init__(self, content: str):
        self.content = content


class _MockCaller:
    """记录每次 run_messages 收到的 block 数, 按序吐分; 可 gate 阻塞模拟在飞打分."""

    def __init__(self, scores=None, gate=None):
        self._scores = list(scores) if scores is not None else []
        self._gate = gate
        self.prompts: list[list[Message]] = []

    async def run_messages(self, prompt: list[Message]):
        self.prompts.append(prompt)
        if self._gate is not None:
            await self._gate.wait()
        score = str(self._scores.pop(0)) if self._scores else "0"
        return _FakeResult(score)


def _clause(text: str, segment_id: str = "g") -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s", segment_id=segment_id,
        phase=RecognitionPhase.CLAUSE, text=text,
        clause=RecognitionClause(text=text),
    )


def _partial(text: str, segment_id: str = "g") -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s", segment_id=segment_id,
        phase=RecognitionPhase.PARTIAL, text=text,
    )


async def _pump() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


def test_parse_stop_score():
    assert parse_stop_score("9") == 9
    assert parse_stop_score(" 7 ") == 7
    assert parse_stop_score("0") == 0
    assert parse_stop_score("garbage") is None
    assert parse_stop_score("") is None


@pytest.mark.asyncio
async def test_commits_when_confident():
    committed = []
    caller = _MockCaller(scores=[9])
    judge = StopJudge(caller=caller, threshold=7, commit=lambda: committed.append(1))
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()
    assert committed == [1]


@pytest.mark.asyncio
async def test_keeps_waiting_when_unsure():
    committed = []
    caller = _MockCaller(scores=[1])
    judge = StopJudge(caller=caller, threshold=7, commit=lambda: committed.append(1))
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()
    assert committed == []


@pytest.mark.asyncio
async def test_commits_on_keyword_without_scoring():
    committed = []
    caller = _MockCaller(scores=[9])
    judge = StopJudge(
        caller=caller, threshold=7, commit=lambda: committed.append(1),
        keywords=["over"],
    )
    await judge.feed(_clause("我觉得应该用长程聆听 over"))
    await _pump()
    assert committed == [1]
    assert caller.prompts == []  # keyword 命中, 没走打分


@pytest.mark.asyncio
async def test_cancels_score_on_activity():
    committed = []
    gate = asyncio.Event()
    caller = _MockCaller(scores=[9], gate=gate)
    judge = StopJudge(caller=caller, threshold=7, commit=lambda: committed.append(1))

    await judge.feed(_clause("我觉得"))
    await _pump()  # 打分 task 阻塞在 gate
    await judge.feed(_partial("我觉得应该"))  # 活动信号 → cancel 在飞打分
    await _pump()
    gate.set()  # 即便 gate 打开, 被取消的 task 不应 commit
    await _pump()
    assert committed == []


@pytest.mark.asyncio
async def test_cancelled_score_then_confident_clause_still_commits():
    committed = []
    gate = asyncio.Event()
    caller = _MockCaller(scores=[9], gate=gate)
    judge = StopJudge(caller=caller, threshold=7, commit=lambda: committed.append(1))

    await judge.feed(_clause("我觉得"))
    await _pump()
    await judge.feed(_partial("我觉得应该"))
    await _pump()
    assert committed == []

    gate.set()  # 解除 gate, 后续打分能跑通
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()
    assert committed == [1]


@pytest.mark.asyncio
async def test_accumulates_clauses_as_message_blocks():
    caller = _MockCaller(scores=[1, 9])
    judge = StopJudge(caller=caller, threshold=7, commit=lambda: None)
    await judge.feed(_clause("第一句"))
    await _pump()
    await judge.feed(_clause("第二句"))
    await _pump()

    assert len(caller.prompts) == 2
    assert len(caller.prompts[0]) == 1  # 一个 clause 一个 content block
    assert len(caller.prompts[1]) == 2  # 累积到第二个 clause


@pytest.mark.asyncio
async def test_segment_switch_resets_state():
    committed = []
    caller = _MockCaller(scores=[1, 9])
    judge = StopJudge(caller=caller, threshold=7, commit=lambda: committed.append(1))

    await judge.feed(_clause("第一段的话", segment_id="g1"))
    await _pump()
    assert committed == []  # 打分 1, 不 commit

    await judge.feed(_clause("新的一段", segment_id="g2"))
    await _pump()
    assert committed == [1]  # 新 segment 重置累积, 打分 9 才 commit
    assert len(caller.prompts[-1]) == 1  # 只累积了新段的这一句


@pytest.mark.asyncio
async def test_on_score_observation_fires():
    observations = []
    caller = _MockCaller(scores=[9])
    judge = StopJudge(
        caller=caller, threshold=7, commit=lambda: None,
        on_score=observations.append,
    )
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()

    assert len(observations) == 1
    obs = observations[0]
    assert obs.clauses == ["我觉得应该这样"]  # 请求: 被打分的 clause
    assert obs.score == 9  # 结果: 解析出的判停分
    assert obs.result.content == "9"  # 原始 LLMFuncResult 透传
