"""StopJudge — per-segment stop-detection cycle (智能判停) 的行为测试.

验证 judge 把 recognition 事件流翻译成 commit 决策: clause 打分、activity cancel、
keyword 显式终点、clause 累积成多 content block、segment_vad 兜底、双边互斥.
mock caller 驱动, 不碰真模型.
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


def _mk_judge(caller, **kwargs):
    """构造不延迟打分、兜底足够长不影响断言的 StopJudge."""
    kwargs.setdefault("judge_delay", 0)
    kwargs.setdefault("segment_vad", 60.0)
    return StopJudge(caller=caller, **kwargs)


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
    judge = _mk_judge(caller, threshold=7, commit=lambda: committed.append(1))
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()
    assert committed == [1]
    judge.close()


@pytest.mark.asyncio
async def test_keeps_waiting_when_unsure():
    committed = []
    caller = _MockCaller(scores=[1])
    judge = _mk_judge(caller, threshold=7, commit=lambda: committed.append(1))
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()
    assert committed == []
    judge.close()


@pytest.mark.asyncio
async def test_commits_on_keyword_without_scoring():
    committed = []
    caller = _MockCaller(scores=[9])
    judge = _mk_judge(
        caller, threshold=7, commit=lambda: committed.append(1),
        keywords=["over"],
    )
    await judge.feed(_clause("我觉得应该用智能判停 over"))
    await _pump()
    assert committed == [1]
    assert caller.prompts == []  # keyword 命中, 没走打分
    judge.close()


@pytest.mark.asyncio
async def test_cancels_score_on_activity():
    committed = []
    gate = asyncio.Event()
    caller = _MockCaller(scores=[9], gate=gate)
    judge = _mk_judge(caller, threshold=7, commit=lambda: committed.append(1))

    await judge.feed(_clause("我觉得"))
    await _pump()  # 打分 task 阻塞在 gate
    await judge.feed(_partial("我觉得应该"))  # 活动信号 → cancel 在飞打分
    await _pump()
    gate.set()  # 即便 gate 打开, 被取消的 task 不应 commit
    await _pump()
    assert committed == []
    judge.close()


@pytest.mark.asyncio
async def test_cancelled_score_then_confident_clause_still_commits():
    committed = []
    gate = asyncio.Event()
    caller = _MockCaller(scores=[9], gate=gate)
    judge = _mk_judge(caller, threshold=7, commit=lambda: committed.append(1))

    await judge.feed(_clause("我觉得"))
    await _pump()
    await judge.feed(_partial("我觉得应该"))
    await _pump()
    assert committed == []

    gate.set()  # 解除 gate, 后续打分能跑通
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()
    assert committed == [1]
    judge.close()


@pytest.mark.asyncio
async def test_accumulates_clauses_as_message_blocks():
    caller = _MockCaller(scores=[1, 9])
    judge = _mk_judge(caller, threshold=7, commit=lambda: None)
    await judge.feed(_clause("第一句"))
    await _pump()
    await judge.feed(_clause("第二句"))
    await _pump()

    assert len(caller.prompts) == 2
    assert len(caller.prompts[0]) == 1  # 一个 clause 一个 content block
    assert len(caller.prompts[1]) == 2  # 累积到第二个 clause
    judge.close()


@pytest.mark.asyncio
async def test_context_prepends_before_clauses():
    """context 非空时作为首消息进入 prompt, clauses 追加其后 (前缀缓存友好)."""
    caller = _MockCaller(scores=[9])
    judge = _mk_judge(
        caller, threshold=7, commit=lambda: None,
        context="user announced: stop on 'over'",
    )
    await judge.feed(_clause("okay over"))
    await _pump()

    assert len(caller.prompts) == 1
    prompt = caller.prompts[0]
    assert len(prompt) == 2  # <context> block + clause
    ctx_text = prompt[0].to_content_string()
    assert "<context>" in ctx_text
    assert "stop on 'over'" in ctx_text
    assert prompt[1].to_content_string() == "okay over"
    judge.close()


@pytest.mark.asyncio
async def test_empty_context_omits_block():
    """context 空时不发 <context> 消息 (与旧行为等价, 无冗余)."""
    caller = _MockCaller(scores=[9])
    judge = _mk_judge(caller, threshold=7, commit=lambda: None, context="")
    await judge.feed(_clause("done"))
    await _pump()

    assert len(caller.prompts[0]) == 1  # 只有 clause, 无 <context> 前缀
    judge.close()


@pytest.mark.asyncio
async def test_segment_switch_resets_state():
    committed = []
    caller = _MockCaller(scores=[1, 9])
    judge = _mk_judge(caller, threshold=7, commit=lambda: committed.append(1))

    await judge.feed(_clause("第一段的话", segment_id="g1"))
    await _pump()
    assert committed == []  # 打分 1, 不 commit

    await judge.feed(_clause("新的一段", segment_id="g2"))
    await _pump()
    assert committed == [1]  # 新 segment 重置累积, 打分 9 才 commit
    assert len(caller.prompts[-1]) == 1  # 只累积了新段的这一句
    judge.close()


@pytest.mark.asyncio
async def test_on_score_observation_fires():
    observations = []
    caller = _MockCaller(scores=[9])
    judge = _mk_judge(
        caller, threshold=7, commit=lambda: None,
        on_score=observations.append,
    )
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()

    assert len(observations) == 1
    obs = observations[0]
    assert obs.clauses == ["我觉得应该这样"]  # 请求: 被打分的 clause
    assert obs.score == 9  # 结果: 解析出的判停分
    assert obs.result.content == "9"  # 原始 LLMFuncResult 透传
    judge.close()


@pytest.mark.asyncio
async def test_judge_failure_emits_score_none():
    observations = []

    class _FailingCaller:
        async def run_messages(self, prompt):
            raise RuntimeError("boom")

    judge = StopJudge(
        caller=_FailingCaller(), threshold=7, judge_delay=0, segment_vad=60.0,
        commit=lambda: None, on_score=observations.append,
    )
    await judge.feed(_clause("我觉得"))
    await _pump()

    assert len(observations) == 1
    assert observations[0].score is None  # llm 调用失败 → 打分 None
    assert observations[0].result is None  # 无 result, 旁路能观察到失败
    judge.close()


# ── segment_vad 兜底 + 双边互斥 ──


@pytest.mark.asyncio
async def test_segment_vad_commits_when_judge_unsure():
    committed = []
    caller = _MockCaller(scores=[1])
    judge = StopJudge(
        caller=caller, threshold=7, segment_vad=0.1, judge_delay=0,
        commit=lambda: committed.append(1),
    )
    await judge.feed(_clause("我觉得"))
    assert committed == []
    await asyncio.sleep(0.2)  # 超过 segment_vad (0.1s)
    assert committed == [1]
    judge.close()


@pytest.mark.asyncio
async def test_segment_vad_renews_on_new_clause():
    committed = []
    caller = _MockCaller(scores=[0, 0])
    judge = StopJudge(
        caller=caller, threshold=7, segment_vad=0.2, judge_delay=0,
        commit=lambda: committed.append(1),
    )
    await judge.feed(_clause("第一句"))
    await asyncio.sleep(0.1)  # 未到 segment_vad
    await judge.feed(_clause("第二句"))  # 展期: 前移 last_clause_at
    await asyncio.sleep(0.1)  # 距第二句仍 < 0.2s
    assert committed == []
    await asyncio.sleep(0.15)  # 距第二句 >= 0.2s
    assert committed == [1]
    judge.close()


@pytest.mark.asyncio
async def test_judge_commit_cancels_vad_no_double_commit():
    committed = []
    caller = _MockCaller(scores=[9])
    judge = StopJudge(
        caller=caller, threshold=7, segment_vad=0.1, judge_delay=0,
        commit=lambda: committed.append(1),
    )
    await judge.feed(_clause("我觉得应该这样"))
    await _pump()
    assert committed == [1]  # judge 先 commit
    await asyncio.sleep(0.2)  # 超过 segment_vad, 但 vad 已被 cancel
    assert committed == [1]  # 仍然只 commit 一次
    judge.close()
