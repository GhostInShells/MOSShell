"""StopJudge — per-segment stop-detection cycle (出口位点的判停单元).

这是出口位点上一个 **openbox 降级实现**, 不是默认实现: 默认出口只有 segment_vad /
keywords, LLM 打分件必须显式声明 (``StopSpec.judge``) 并且环境能注入 caller 才组装得起来.
将来专用端点模型进来时替换掉的就是本类所在的这一格, 槽位形状不变.

Translates the recognition event stream into commit decisions:
- clause: accumulate, spawn the debounced llm judge, arm the segment_vad timer
- first/partial: cancel the in-flight judge
- judge score >= threshold → commit (early); segment_vad expiry → commit (fallback)
- keyword hit → commit immediately (explicit endpoint, no scoring)

Holds a MossLLMCaller (externally assembled — the model dependency). Independently
testable: drive ``feed`` with a mock caller and assert commit timing.
"""
import asyncio
import logging
import time
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
    result: LLMFuncResult | None


class StopJudge:
    """Per-segment stop-detection cycle — segment_vad timer + llm judge, mutually exclusive.

    One cycle per segment (turn). Two commit paths race to end the turn:
    - ``segment_vad`` timer: commits `segment_vad` seconds after the last clause
      (renewed by each new clause — 展期). This is the baseline fallback.
    - llm judge: debounced by ``judge_delay`` (安全期) so a clause superseded within
      that window never costs a call; commits early when score >= ``threshold``.

    Whichever fires first commits exactly once — a single ``_committed`` flag plus
    mutual task cancellation, with no ``await`` between flag-set and commit. A new
    segment tears the cycle down (cancels both tasks, resets accumulation).

    ``feed`` is async but never awaits scoring inline — a clause only spawns the judge
    task and returns (``on_event_creating`` is the inline-await mount point; blocking
    there blocks the receive loop).
    """

    def __init__(
            self,
            *,
            caller: MossLLMCaller | None = None,
            judge: bool = True,
            threshold: int = 7,
            segment_vad: float = 3.0,
            judge_delay: float = 0.3,
            commit: Callable[[], None],
            keywords: Sequence[str] | None = None,
            context: str = "",
            on_score: Callable[[StopScoreObservation], None] | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self._caller = caller
        self._judge = judge
        self._threshold = threshold
        self._segment_vad = segment_vad
        self._judge_delay = judge_delay
        self._commit = commit
        self._keywords = list(keywords) if keywords else []
        self._context = context
        self._on_score = on_score
        self._logger = logger or logging.getLogger("moss")
        self._segment_id: str | None = None
        self._clauses: list[str] = []
        self._committed = False
        self._last_clause_at: float | None = None
        self._judge_task: asyncio.Task | None = None
        self._vad_task: asyncio.Task | None = None
        self._epoch = 0

    async def feed(self, event: RecognitionEvent) -> None:
        """Drive the cycle with a recognition event (FIRST/PARTIAL/CLAUSE). Non-blocking."""
        if event.segment_id != self._segment_id:
            self._teardown()
            self._segment_id = event.segment_id
            self._clauses = []
            self._committed = False
            self._last_clause_at = None
        if self._committed:
            return
        if event.phase == RecognitionPhase.CLAUSE:
            self._on_clause(event)
        elif event.phase in (RecognitionPhase.FIRST, RecognitionPhase.PARTIAL):
            self._cancel_judge()

    def close(self) -> None:
        """Cancel in-flight stop tasks (session teardown)."""
        self._teardown()

    # ── commit paths ──

    def _on_clause(self, event: RecognitionEvent) -> None:
        text = event.clause.text if event.clause else event.text
        if self._keywords and any(k in text for k in self._keywords):
            self._try_commit()
            return
        self._clauses.append(text)
        if self._segment_vad <= 0:
            self._try_commit()  # segment_vad=0 → 首个 clause 即端点, 不起定时器
            return
        self._last_clause_at = time.monotonic()
        self._start_vad()
        if self._judge:
            self._start_judge()

    def _start_vad(self) -> None:
        self._cancel_vad()
        self._vad_task = asyncio.create_task(self._vad_loop())

    async def _vad_loop(self) -> None:
        # deadline = last_clause_at + segment_vad; each new clause moves it forward (展期).
        while True:
            await asyncio.sleep(0.05)
            if self._committed or self._last_clause_at is None:
                return
            if time.monotonic() - self._last_clause_at >= self._segment_vad:
                self._try_commit()
                return

    def _start_judge(self) -> None:
        self._cancel_judge()
        epoch = self._epoch
        self._judge_task = asyncio.create_task(self._judge_loop(list(self._clauses), epoch))

    async def _judge_loop(self, clauses: list[str], epoch: int) -> None:
        try:
            await asyncio.sleep(self._judge_delay)  # 安全期: superseded → cancelled, no call spent
            result = await self._caller.run_messages(self._build_messages(clauses))
            score = parse_stop_score(result.content or "")
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("stop judge failed")
            self._emit_score(clauses, None, None)  # surface the failure to observers
            return
        self._emit_score(clauses, score, result)
        if epoch != self._epoch or self._committed:
            return
        if score is not None and score >= self._threshold:
            self._try_commit()

    def _emit_score(self, clauses: list[str], score: int | None, result: LLMFuncResult | None) -> None:
        """Emit a score observation; a throwing observer must not kill the commit."""
        if self._on_score is None:
            return
        try:
            self._on_score(StopScoreObservation(clauses=list(clauses), score=score, result=result))
        except Exception:
            self._logger.exception("stop judge on_score observer failed")

    def _try_commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        self._cancel_vad()
        self._cancel_judge()
        self._commit()

    # ── internals ──

    def _cancel_judge(self) -> None:
        if self._judge_task is not None and not self._judge_task.done():
            self._judge_task.cancel()
        self._judge_task = None
        self._epoch += 1

    def _cancel_vad(self) -> None:
        if self._vad_task is not None and not self._vad_task.done():
            self._vad_task.cancel()
        self._vad_task = None

    def _teardown(self) -> None:
        self._cancel_vad()
        self._cancel_judge()

    def _build_messages(self, clauses: list[str]) -> list[Message]:
        """One content block per clause — the accumulated prefix hits the prompt cache."""
        messages: list[Message] = []
        if self._context:
            messages.append(Message.new().with_content(f"<context>\n{self._context}\n</context>"))
        for clause in clauses:
            messages.append(Message.new().with_content(clause))
        return messages
