"""In-process QA exchange backed by janus queues.

Pushes all communication through janus queues — broadcast, reply, verdict,
and cancel all travel through queues consumed by independent asyncio tasks.
Callbacks fire in the consumer's context, never inline on the issuer or
responder stack.

Why janus.Queue (not asyncio.Queue):
    reply() may be called from any thread (TUI input, sync callback, …).
    janus bridges sync-q puts and async-q gets without forcing the caller
    into an event loop.  The owner's consumer task drains the async side.

Why ThreadSafeEvent (not asyncio.Event):
    done-event may be waited from asyncio (``await qa.wait()``) and set
    from a sync context (cancel, owner's reply consumer receiving via
    janus async_q).  ThreadSafeEvent is the standard MOSS primitive for
    this crossing (see core/helpers/asyncio_utils.py).

Flow::

    Asker                         Distributor              Watcher(s)
    -----                         -----------              ----------
    broadcast_question --sync_q-> broadcast_q --async_q-> copy to watcher_q
                                  (namespace task)

    Owner QA                      Distributor              Watcher
    --------                      -----------              -------
    _consume_replies <-async_q-- reply_queue <-sync_q-- reply(answer)
    _accept_answer -> verdict -sync_q-> broadcast_q --> copy to watcher_q
    cancel() -> cancel -sync_q-> broadcast_q --> copy to watcher_q
"""

from __future__ import annotations

import asyncio
import janus
from threading import Lock
from typing import Callable

from ghoshell_moss.core.concepts.qa import (
    QAManager, Asker, Watcher, QA, QAMeta, Question, Answer,
)
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

# --- broadcast queue payload tags ---

_BQ_QUESTION = 'question'
_BQ_VERDICT = 'verdict'
_BQ_CANCEL = 'cancel'

# Bounded queue sizes.  Low enough to back-pressure without stalling the
# event loop; high enough that a single namespace with a few watchers
# never drops a message under normal load.
_REPLY_QUEUE_SIZE = 16
_BROADCAST_QUEUE_SIZE = 64
_WATCHER_QUEUE_SIZE = 32


# ============================================================
#  JanusQA
# ============================================================

class JanusQA(QA):
    """In-process QA handle backed by janus queues.

    Owner copy (owned=True):
        Spawns a consumer task that drains *reply_queue* and calls
        :meth:`_accept_answer` on the first-arriving reply.  Subsequent
        replies are silently dropped because the done-event is already set.

    Responder copy (owned=False):
        :meth:`reply` locks locally then pushes the answer into the shared
        *reply_queue* for the owner to consume.  The responder never sets
        the done event — it waits for the verdict broadcast instead.
    """

    def __init__(
        self,
        question: Question,
        identifier: str,
        reply_queue: janus.Queue,
        owned: bool,
    ) -> None:
        self._question = question
        self._identifier = identifier
        self._reply_queue = reply_queue
        self._owned = owned

        self._done_event = ThreadSafeEvent()
        self._answer: Answer | None = None
        self._replied: Answer | None = None
        self._reply_lock = Lock()

        self._answer_callbacks: list[Callable[[Answer], None]] = []
        self._cancel_callbacks: list[Callable[[Question], None]] = []

        # Owned QAs spawn a background consumer task that drains the
        # reply_queue and accepts the first answer.  Unowned copies
        # receive the verdict through the distribute → watcher path.
        self._reply_consumer: asyncio.Task | None = None
        if self._owned:
            self._reply_consumer = asyncio.ensure_future(self._consume_replies())

    # -- internal setters ------------------------------------------------
    # These are public-internal: called by JanusWatcher / JanusAsker to
    # apply broadcast verdicts onto local copies.  They are NOT part of
    # the QA ABC, but they are the sanctioned extension point that
    # concrete implementations expose to their owning runtime.

    def _accept_answer(self, answer: Answer) -> None:
        """Owner-side: validate, record the accepted answer, finalise."""
        with self._reply_lock:
            if self._done_event.is_set():
                return
            answer.match_question(self._question)
            self._answer = answer
            self._replied = answer
            self._done_event.set()
        self._fire_answer(answer)

    def _apply_verdict(self, answer: Answer) -> None:
        """Responder-side: apply final answer from verdict broadcast.

        Does NOT re-validate — the owner already validated via
        match_question in _accept_answer.
        """
        with self._reply_lock:
            if self._done_event.is_set():
                return
            self._answer = answer
            self._done_event.set()
        self._fire_answer(answer)

    def _apply_cancel(self, reason: str) -> None:
        """Any copy: apply a cancel verdict from broadcast."""
        with self._reply_lock:
            if self._done_event.is_set():
                return
            self._question.canceled = reason
            self._done_event.set()
        for cb in self._cancel_callbacks:
            try:
                cb(self._question)
            except Exception:
                pass

    # -- private ---------------------------------------------------------

    async def _consume_replies(self) -> None:
        try:
            while True:
                answer: Answer = await self._reply_queue.async_q.get()
                self._accept_answer(answer)
        except (asyncio.CancelledError, RuntimeError):
            pass

    def _fire_answer(self, answer: Answer) -> None:
        for cb in self._answer_callbacks:
            try:
                cb(answer)
            except Exception:
                pass

    # -- read-only (QA contract) ----------------------------------------

    @property
    def question(self) -> Question:
        return self._question

    def answer_meta(self) -> QAMeta:
        return self._question.meta.new_reply(self._identifier)

    @property
    def answer(self) -> Answer | None:
        return self._answer

    def done(self) -> bool:
        return self._done_event.is_set()

    def replied(self) -> Answer | None:
        return self._replied

    def owned(self) -> bool:
        return self._owned

    def canceled(self) -> bool:
        return bool(self._question.canceled)

    # -- callbacks (QA contract) ----------------------------------------

    def on_answer(self, callback: Callable[[Answer], None]) -> None:
        self._answer_callbacks.append(callback)

    def on_cancel(self, callback: Callable[[Question], None]) -> None:
        self._cancel_callbacks.append(callback)

    # -- writes (QA contract) -------------------------------------------

    def cancel(self, reason: str = '') -> None:
        if not self._owned:
            raise ValueError("only owner can cancel")
        with self._reply_lock:
            if self._done_event.is_set():
                return
            self._question.canceled = reason
            self._done_event.set()
        for cb in self._cancel_callbacks:
            try:
                cb(self._question)
            except Exception:
                pass
        if self._reply_consumer is not None:
            self._reply_consumer.cancel()

    def reply(self, answer: Answer) -> None:
        if self._replied is not None:
            raise ValueError("already replied")
        if self._done_event.is_set():
            raise ValueError("already done")

        # Validate and stamp meta before acquiring the lock so that
        # malformed answers fail-fast without touching shared state.
        answer.match_question(self._question)
        answer.meta = self.answer_meta()

        with self._reply_lock:
            if self._replied is not None:
                raise ValueError("already replied")
            self._replied = answer

        # Push through janus sync side — thread-safe, does not require
        # the caller to be inside an event loop.
        self._reply_queue.sync_q.put_nowait(answer)

    async def wait(self) -> None:
        await self._done_event.wait()


# ============================================================
#  JanusAsker
# ============================================================

class JanusAsker(Asker):
    """Asker that broadcasts questions via a namespace-scoped janus queue.

    Each issued question gets its own reply queue (janus.Queue) for the
    answer round-trip.  When the owner accepts an answer or cancels, the
    verdict is pushed back to the broadcast queue so all watcher copies
    converge.
    """

    def __init__(
        self,
        issuer: str,
        namespace: str,
        broadcast_queue: janus.Queue,
    ) -> None:
        self._issuer = issuer
        self._namespace = namespace
        self._broadcast_queue = broadcast_queue
        self._owned_qas: dict[str, QA] = {}

    @property
    def issuer(self) -> str:
        return self._issuer

    @property
    def namespace(self) -> str:
        return self._namespace

    def undone(self) -> list[QA]:
        return [qa for qa in self._owned_qas.values() if not qa.done()]

    def broadcast_question(self, question: Question) -> QA:
        reply_queue: janus.Queue = janus.Queue(maxsize=_REPLY_QUEUE_SIZE)

        qa = JanusQA(question, self._issuer, reply_queue, owned=True)
        qid = question.meta.id
        self._owned_qas[qid] = qa

        # When the owner accepts, broadcast the verdict so every watcher
        # copy learns the final answer (KD8: 先到先得, KD1: requester 持真相).
        def _broadcast_verdict(answer: Answer) -> None:
            self._broadcast_queue.sync_q.put_nowait(
                (_BQ_VERDICT, qid, answer),
            )

        def _broadcast_cancel(question: Question) -> None:
            self._broadcast_queue.sync_q.put_nowait(
                (_BQ_CANCEL, qid, question.canceled),
            )

        qa.on_answer(_broadcast_verdict)
        qa.on_cancel(_broadcast_cancel)

        self._broadcast_queue.sync_q.put_nowait(
            (_BQ_QUESTION, question, reply_queue),
        )

        return qa


# ============================================================
#  JanusWatcher
# ============================================================

class JanusWatcher(Watcher):
    """Watcher that receives questions via a janus inbound queue.

    A background asyncio task drains *inbound_queue* and creates responder
    QA copies when a new question arrives, then fires :meth:`on_question`
    callbacks.  Verdict / cancel broadcasts are also consumed here and
    applied to the local copy via ``_apply_verdict`` / ``_apply_cancel``.
    """

    def __init__(
        self,
        namespace: str,
        identifier: str,
        inbound_queue: janus.Queue,
    ) -> None:
        self._namespace = namespace
        self._identifier = identifier
        self._inbound_queue = inbound_queue
        self._qas: dict[str, JanusQA] = {}
        self._on_question_cbs: list[Callable[[QA], None]] = []
        self._consumer: asyncio.Task | None = None
        self._start()

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def identifier(self) -> str:
        return self._identifier

    # -- lifecycle -------------------------------------------------------

    def _start(self) -> None:
        self._consumer = asyncio.ensure_future(self._consume())

    async def _consume(self) -> None:
        try:
            while True:
                kind, *payload = await self._inbound_queue.async_q.get()
                if kind == _BQ_QUESTION:
                    question, reply_queue = payload
                    qa = JanusQA(question, self._identifier, reply_queue, owned=False)
                    self._qas[question.meta.id] = qa
                    for cb in self._on_question_cbs:
                        try:
                            cb(qa)
                        except Exception:
                            pass
                elif kind == _BQ_VERDICT:
                    qid, answer = payload
                    qa = self._qas.get(qid)
                    if qa is not None:
                        qa._apply_verdict(answer)
                elif kind == _BQ_CANCEL:
                    qid, reason = payload
                    qa = self._qas.get(qid)
                    if qa is not None:
                        qa._apply_cancel(reason)
        except (asyncio.CancelledError, RuntimeError):
            pass

    def stop(self) -> None:
        if self._consumer is not None:
            self._consumer.cancel()

    # -- Watcher ABC -----------------------------------------------------

    def questions(self, *, answered: bool = False) -> list[QA]:
        if answered:
            return [qa for qa in self._qas.values() if qa.done()]
        return list(self._qas.values())

    def on_question(self, callback: Callable[[QA], None]) -> None:
        self._on_question_cbs.append(callback)


# ============================================================
#  JanusQAManager
# ============================================================

class JanusQAManager(QAManager):
    """In-process QA manager that routes messages via janus queues.

    One broadcast queue per namespace.  A background distributor task
    reads from the broadcast queue and copies each message to every
    registered watcher inbound queue in that namespace.

    Construct one instance per process / cell and call :meth:`asker` /
    :meth:`watch` to obtain role-bound actors (KD3).
    """

    def __init__(self, issuer: str) -> None:
        self._issuer = issuer
        self._broadcast_queues: dict[str, janus.Queue] = {}
        self._watcher_queues: dict[str, list[janus.Queue]] = {}
        self._distributor_tasks: dict[str, asyncio.Task] = {}

    @property
    def issuer(self) -> str:
        return self._issuer

    # -- namespace management --------------------------------------------

    def _ensure_namespace(self, namespace: str) -> janus.Queue:
        if namespace not in self._broadcast_queues:
            broadcast_q = janus.Queue(maxsize=_BROADCAST_QUEUE_SIZE)
            self._broadcast_queues[namespace] = broadcast_q
            self._watcher_queues[namespace] = []
            self._distributor_tasks[namespace] = asyncio.ensure_future(
                self._run_distributor(namespace),
            )
        return self._broadcast_queues[namespace]

    async def _run_distributor(self, namespace: str) -> None:
        """Distribute broadcast-queue messages to all watcher queues."""
        broadcast_q = self._broadcast_queues[namespace]
        try:
            while True:
                kind, *payload = await broadcast_q.async_q.get()
                for watcher_q in self._watcher_queues.get(namespace, []):
                    if kind == _BQ_QUESTION:
                        question, reply_queue = payload
                        watcher_q.sync_q.put_nowait(
                            (_BQ_QUESTION, question, reply_queue),
                        )
                    elif kind == _BQ_VERDICT:
                        qid, answer = payload
                        watcher_q.sync_q.put_nowait(
                            (_BQ_VERDICT, qid, answer),
                        )
                    elif kind == _BQ_CANCEL:
                        qid, reason = payload
                        watcher_q.sync_q.put_nowait(
                            (_BQ_CANCEL, qid, reason),
                        )
        except (asyncio.CancelledError, RuntimeError):
            pass

    # -- QAManager ABC ---------------------------------------------------

    def asker(self, namespace: str) -> Asker:
        broadcast_q = self._ensure_namespace(namespace)
        return JanusAsker(self._issuer, namespace, broadcast_q)

    def watch(self, namespace: str) -> Watcher:
        self._ensure_namespace(namespace)
        watcher_q = janus.Queue(maxsize=_WATCHER_QUEUE_SIZE)
        self._watcher_queues[namespace].append(watcher_q)
        return JanusWatcher(namespace, self._issuer, watcher_q)
