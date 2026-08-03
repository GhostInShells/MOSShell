from typing import Callable
from ghoshell_moss.core.concepts.qa import (
    QAManager, QAMeta, Question, Answer, Watcher, Asker, QA
)
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from threading import Lock
import janus


class JanusQA(QA):

    def __init__(
            self,
            question: Question,
            issuer: str,
            namespace: str,
            reply_queue: janus.Queue,
    ):
        if question.meta is None:
            raise ValueError(f"question.meta cannot be None")
        self._question = question
        self._issuer = issuer
        self._namespace = namespace
        self._reply_queue = reply_queue
        self._done_event = ThreadSafeEvent()
        self._answer: Answer | None = None
        self._replied: Answer | None = None
        self._answer_callbacks: list[Callable[[Answer], None]] = []
        self._cancel_callbacks: list[Callable[[Question], None]] = []
        self._result_locker = Lock()

    @property
    def question(self) -> Question:
        return self._question

    def answer_meta(self) -> QAMeta:
        return self._question.meta.new_reply(self._issuer)

    def on_answer(self, callback: Callable[[Answer], None]) -> None:
        self._answer_callbacks.append(callback)

    def on_cancel(self, callback: Callable[[Question], None]) -> None:
        self._cancel_callbacks.append(callback)

    @property
    def answer(self) -> Answer | None:
        return self._answer

    def set_answer(self, answer: Answer) -> None:
        """private setter for QA"""
        answer.match_question(self._question)
        self._answer = answer
        self._done_event.set()
        if len(self._answer_callbacks) > 0:
            for callback in self._answer_callbacks:
                callback(self._answer)

    def set_cancel(self, reason: str = '') -> None:
        """private setter for QA"""
        question = self._question
        question.canceled = reason
        self._done_event.set()
        for callback in self._cancel_callbacks:
            callback(self._question)

    def done(self) -> bool:
        return self._done_event.is_set()

    def replied(self) -> Answer | None:
        return self._replied

    def cancel(self, reason: str = '') -> None:
        if not self.owned():
            raise ValueError(f"cannot cancel question {self._question.meta}")
        elif self.done():
            raise ValueError(f"cannot cancel question {self._question.meta} which is done")
        for callback in self._cancel_callbacks:
            callback(self._question)

    def owned(self) -> bool:
        return self._question.meta.issuer == self._issuer

    def canceled(self) -> bool:
        return len(self._question.canceled) > 0

    def reply(self, answer: Answer) -> None:
        if self._replied is not None:
            raise ValueError(f"reply cannot be set twice")
        elif self.done():
            raise ValueError(f"reply cannot be set twice")

        with self._result_locker:
            self._replied = answer
            if self.owned():
                self.set_answer(answer)
            else:
                self._reply_queue.sync_q.put_nowait(answer)

    async def wait(self) -> None:
        await self._done_event.wait()


class JanusAsker(Asker):

    def __init__(
            self,
            issuer: str,
            namespace: str,
            broadcaster: Callable[[QA], None],
    ):
        self._issuer = issuer
        self._namespace = namespace

    @property
    def identifier(self) -> str:
        return self._issuer

    @property
    def namespace(self) -> str:
        return self._namespace

    def broadcast_question(self, question: Question) -> QA:
        if question.meta is None:
            raise ValueError(f"question.meta cannot be None")


class JanusQAManager(QAManager):

    def __init__(
            self,
            issuer: str,
    ):
        self._issuer = issuer

    @property
    def issuer(self) -> str:
        return self._issuer

    def asker(self, namespace: str) -> Asker:
        pass

    def watch(self, namespace: str) -> Watcher:
        pass
