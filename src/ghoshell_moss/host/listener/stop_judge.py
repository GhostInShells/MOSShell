"""StopJudge — 智能判停 (长程聆听) 的状态机组件.

把 recognition 事件流翻译成 commit 决策:
- clause (决策点): 累积 clause 文本, spawn 打分 task (caller.run_messages)
- first/partial (活动信号): cancel 在飞打分 task
- 打分 >= threshold → 调 commit 回调
- keywords 命中 → 立刻 commit (显式终点, 不打分)

持 MossLLMCaller (外部装配, instruction/model/输出约束已绑定). 独立可测:
用一个 mock caller 驱动 feed, 断言 commit 时机.
"""
import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.contracts.asr import RecognitionEvent, RecognitionPhase
from ghoshell_moss.contracts.llms import LLMFuncResult, MossLLMCaller
from ghoshell_moss.message import Message

__all__ = ["STOP_JUDGE_INSTRUCTION", "StopJudge", "StopScoreObservation", "parse_stop_score"]

STOP_JUDGE_INSTRUCTION = """\
You are running a single multi-class classification task, not holding a
conversation and not using tools.

Goal: read one utterance from a live speech transcript and rate how complete
the speaker's thought is, as an integer 0-9. The score drives a turn-taking
decision: the system responds on a high score and keeps listening on a low one.

Mechanism: after the input you emit exactly ONE token — a bare digit 0-9.
The digit is read directly by a parser; any other output (a sentence, a word,
punctuation, an explanation) is a hard failure. End your reply immediately
after the digit.

Scale:
0-3 = clearly unfinished — cut mid-word or mid-phrase, ends on a trailing
      conjunction or dangling condition, or only fillers/discourse markers.
4-6 = uncertain — could honestly stop here or continue.
7-9 = clearly finished — a complete statement, an answerable question, a
      standalone greeting, or a closed short answer (yes / no / okay).

Strategy for judging a long, live session. The transcript may be Chinese,
English, or mixed — apply the same rules:
- ASR is unreliable and often renders homophones or near-sounds (谐音). Judge
  by meaning and intent, never by surface spelling.
- If the <context> declares an explicit end signal (for example the speaker
  ends each turn with "over"), hearing that signal is strong positive
  evidence of completion.
- Fillers and trailing phrases split into three states:
  * Still composing — "um…", "you know…", "how should I put it…", 嗯…, 啊…,
    那个…, 就是…, 怎么说呢…, 然后… The thought has not landed. Score 0-3.
  * Wants brief acknowledgment — "you understand?", "right?", "okay?",
    你明白吗?, 是吧?, 对吧? The point is made; it only asks for a nod, not a
    full reply. Score 7-9.
  * Wants an answer now — "what do you think?", a direct question, 你觉得呢?,
    怎么办?, 对不对? Clearly finished and awaiting a reply. Score 7-9.
- A trailing conjunction or open condition (because…, if…, 如果…的话, 因为…,
  虽然…) means the sentence is grammatically open: score 0-3.

Behavior:
- This is classification, not deliberation. Trust your first impression.
- Output only one integer 0-9, no punctuation, no prose.

Input format: you receive one message per clause of the live transcript, in
order. An optional first message wrapped in <context>…</context> carries
prior turns or a listening etiquette; clause messages themselves are raw
text. After the last clause, output nothing but your score.\
"""


def parse_stop_score(raw: str) -> int | None:
    """从原始输出提取 0-9 判停分. 无数字返回 None."""
    s = raw.strip()
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return None


@dataclass
class StopScoreObservation:
    """一次判停打分的观测 — 请求 (clauses) + 结果 (score + LLMFuncResult).

    供旁路监控 (测试 node / 日志) 观察 judge 实际发了什么、模型回了什么分、
    花了多久 (``result.cast``) 与多少 token (``result.usage``).
    """

    clauses: list[str]
    score: int | None
    result: LLMFuncResult


class StopJudge:
    """智能判停状态机 — 判停输入是 in-flight 未 commit 的累积 clause, 不是 segment 全文.

    commit 才生产 segment. 每次 clause 到来累积一条, 打分看的是「当前累积到哪为止
    说话人有没有把意思讲完」; 活动信号 (first/partial) 说明还在说, 上一次判断作废.

    ``feed`` 是 async 但绝不 inline await 打分 — clause 只 spawn 打分 task, 立即返回
    (on_event_creating 是 inline await 挂载点, 阻塞一次就堵死收包). 打分在后台 task
    里跑, 用 epoch 计数防止被取代/迟到的结果二次 commit.
    """

    def __init__(
            self,
            *,
            caller: MossLLMCaller,
            threshold: int = 7,
            commit: Callable[[], None],
            keywords: Sequence[str] | None = None,
            context: str = "",
            on_score: Callable[[StopScoreObservation], None] | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self._caller = caller
        self._threshold = threshold
        self._commit = commit
        self._keywords = list(keywords) if keywords else []
        self._context = context
        self._on_score = on_score
        self._logger = logger or logging.getLogger("moss")
        self._segment_id: str | None = None
        self._clauses: list[str] = []
        self._task: asyncio.Task | None = None
        self._epoch = 0
        self._committed = False

    async def feed(self, event: RecognitionEvent) -> None:
        """喂一个 recognition 事件 (FIRST/PARTIAL/CLAUSE), 驱动状态机. 非阻塞."""
        if event.segment_id != self._segment_id:
            self._segment_id = event.segment_id
            self._clauses = []
            self._committed = False
            self._cancel_task()
        if event.phase == RecognitionPhase.CLAUSE:
            self._on_clause(event)
        elif event.phase in (RecognitionPhase.FIRST, RecognitionPhase.PARTIAL):
            self._cancel_task()

    def close(self) -> None:
        """取消在飞打分 task (session 结束时的清理)."""
        self._cancel_task()

    def _on_clause(self, event: RecognitionEvent) -> None:
        if self._committed:
            return
        text = event.clause.text if event.clause else event.text
        if self._keywords and any(k in text for k in self._keywords):
            self._committed = True
            self._cancel_task()
            self._commit()
            return
        self._clauses.append(text)
        self._cancel_task()
        self._task = asyncio.create_task(self._score(list(self._clauses), self._epoch))

    def _cancel_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._epoch += 1

    async def _score(self, clauses: list[str], epoch: int) -> None:
        try:
            score = await self._judge(clauses)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("stop judge failed")
            return
        if epoch != self._epoch:
            return
        if score is not None and score >= self._threshold:
            self._committed = True
            self._commit()

    async def _judge(self, clauses: list[str]) -> int | None:
        result = await self._caller.run_messages(self._build_messages(clauses))
        score = parse_stop_score(result.content or "")
        if self._on_score is not None:
            self._on_score(StopScoreObservation(clauses=list(clauses), score=score, result=result))
        return score

    def _build_messages(self, clauses: list[str]) -> list[Message]:
        """累积 clause → 逐条 content block (一个 clause 一个 block, 前缀缓存命中)."""
        messages: list[Message] = []
        if self._context:
            messages.append(Message.new().with_content(f"<context>\n{self._context}\n</context>"))
        for clause in clauses:
            messages.append(Message.new().with_content(clause))
        return messages
