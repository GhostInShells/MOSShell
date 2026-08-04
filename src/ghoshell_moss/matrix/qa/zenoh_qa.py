"""Cross-process QA exchange backed by zenoh pub/sub.

Keyexpr layout (prefix is injected at construction, typically
``{network_ns}/qa``)::

    {prefix}/questions/{ns}   — question broadcast (Asker pub, Watcher sub)
    {prefix}/replies/{ns}     — answer submissions  (Watcher pub, Asker sub)
    {prefix}/verdicts/{ns}    — accepted answer / cancel (Asker pub, Watcher sub)
    {prefix}/query/{ns}       — late-join queryable (Asker responds with undone)

All payloads are JSON-serialised pydantic models.  The qid is carried in
``QAMeta.refer_to`` (answers / verdicts point back to their question), never
in the keyexpr.

Lifecycle
---------
:class:`ZenohQAManager` is an async context manager.  Subscribers and
queryables are declared lazily when the first Asker / Watcher is created
for a namespace and undeclared on ``__aexit__``.

Asker / Watcher are plain sync factories — they do not expose their own
context manager.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from threading import Lock
from typing import Callable

import janus

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.core.concepts.qa import (
    QAManager, Asker, Watcher, QA, QAMeta, Question, Answer,
)
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.depends import depend_matrix

depend_matrix()
import zenoh  # noqa: E402

TaskSpawner = Callable[[Coroutine], asyncio.Task]

_ASKR_QUEUE_SIZE = 16
_WATCHER_QUEUE_SIZE = 64


def _short_qid(qid: str) -> str:
    return qid[:8]


# ============================================================
#  ZenohQA
# ============================================================

class ZenohQA(QA):
    """Cross-process QA handle.

    Owner copy (owned=True):
        The owning Asker's reply subscriber calls :meth:`_accept_answer` when
        an answer arrives on ``{prefix}/replies/{ns}``.

    Responder copy (owned=False):
        :meth:`reply` locks locally then calls the *reply_publisher* callback
        injected by the Watcher that created this copy.  The callback does
        ``session.put(reply_keyexpr, answer_json)``.
    """

    def __init__(
        self,
        question: Question,
        identifier: str,
        owned: bool,
        *,
        reply_publisher: Callable[[Answer], None] | None = None,
        logger: LoggerItf | None = None,
    ) -> None:
        self._question = question
        self._identifier = identifier
        self._owned = owned
        self._reply_publisher = reply_publisher

        self._logger = logger or get_moss_logger()
        self._log_prefix = f"[ZenohQA qid={_short_qid(question.meta.id)}]"

        self._done_event = ThreadSafeEvent()
        self._answer: Answer | None = None
        self._replied: Answer | None = None
        self._reply_lock = Lock()

        self._answer_callbacks: list[Callable[[Answer], None]] = []
        self._cancel_callbacks: list[Callable[[Question], None]] = []

    # -- internal setters ------------------------------------------------

    def _accept_answer(self, answer: Answer) -> None:
        """Owner-side: validate, record the accepted answer, finalise."""
        with self._reply_lock:
            if self._done_event.is_set():
                self._logger.debug(
                    "%s _accept_answer skipped — already done", self._log_prefix,
                )
                return
            answer.match_question(self._question)
            self._answer = answer
            self._replied = answer
            self._done_event.set()
        self._logger.info(
            "%s answer accepted rejected=%s", self._log_prefix, answer.rejected,
        )
        self._fire_answer(answer)

    def _apply_verdict(self, answer: Answer) -> None:
        """Responder-side: apply final answer from verdict broadcast."""
        with self._reply_lock:
            if self._done_event.is_set():
                return
            self._answer = answer
            self._done_event.set()
        self._logger.info(
            "%s verdict applied (responder copy)", self._log_prefix,
        )
        self._fire_answer(answer)

    def _apply_cancel(self, reason: str) -> None:
        """Any copy: apply a cancel verdict from broadcast."""
        with self._reply_lock:
            if self._done_event.is_set():
                return
            self._question.canceled = reason
            self._done_event.set()
        self._logger.info(
            "%s cancel applied reason=%r", self._log_prefix, reason,
        )
        for cb in self._cancel_callbacks:
            try:
                cb(self._question)
            except Exception:
                self._logger.exception(
                    "%s on_cancel callback failed", self._log_prefix,
                )

    # -- private ---------------------------------------------------------

    def _fire_answer(self, answer: Answer) -> None:
        for cb in self._answer_callbacks:
            try:
                cb(answer)
            except Exception:
                self._logger.exception(
                    "%s on_answer callback failed", self._log_prefix,
                )

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
        self._logger.info(
            "%s cancelled reason=%r", self._log_prefix, reason,
        )
        for cb in self._cancel_callbacks:
            try:
                cb(self._question)
            except Exception:
                self._logger.exception(
                    "%s on_cancel callback failed", self._log_prefix,
                )

    def reply(self, answer: Answer) -> None:
        if self._replied is not None:
            raise ValueError("already replied")
        if self._done_event.is_set():
            raise ValueError("already done")

        answer.match_question(self._question)
        answer.meta = self.answer_meta()

        with self._reply_lock:
            if self._replied is not None:
                raise ValueError("already replied")
            self._replied = answer

        if self._reply_publisher is None:
            raise RuntimeError("reply_publisher not set — Watcher must inject one")
        self._reply_publisher(answer)

    async def wait(self) -> None:
        await self._done_event.wait()


# ============================================================
#  ZenohAsker
# ============================================================

class ZenohAsker(Asker):
    """Asker that broadcasts questions and receives replies via zenoh.

    Subscribes to ``{prefix}/replies/{ns}`` to receive answers from
    watchers.  Zenoh callbacks run on zenoh's I/O thread and only
    deserialise + enqueue; a consumer task on the event loop drains
    the queue and dispatches to the matching owner QA.
    """

    def __init__(
        self,
        issuer: str,
        namespace: str,
        *,
        session: zenoh.Session,
        questions_key: str,
        replies_key: str,
        verdicts_key: str,
        query_key: str,
        logger: LoggerItf | None = None,
    ) -> None:
        self._issuer = issuer
        self._namespace = namespace
        self._session = session
        self._questions_key = questions_key
        self._replies_key = replies_key
        self._verdicts_key = verdicts_key
        self._query_key = query_key
        self._logger = logger or get_moss_logger()
        self._log_prefix = f"[ZenohAsker ns={namespace}]"

        self._owned_qas: dict[str, ZenohQA] = {}
        self._reply_subscriber: zenoh.Subscriber | None = None
        self._queryable: zenoh.Queryable | None = None

        # janus queue + consumer — offloads dispatch from zenoh I/O thread
        # to the event loop, matching the janus QA isolation pattern.
        self._event_queue: janus.Queue | None = None
        self._consumer: asyncio.Task | None = None

    @property
    def issuer(self) -> str:
        return self._issuer

    @property
    def namespace(self) -> str:
        return self._namespace

    def undone(self) -> list[QA]:
        return [qa for qa in self._owned_qas.values() if not qa.done()]

    # -- lifecycle (called by QAManager) ---------------------------------

    def _start(self, spawn: TaskSpawner) -> None:
        """Declare zenoh subscribers, queryables, and consumer task."""
        self._event_queue = janus.Queue(maxsize=_ASKR_QUEUE_SIZE)
        self._consumer = spawn(self._consume())

        self._reply_subscriber = self._session.declare_subscriber(
            self._replies_key,
            self._on_reply,
        )
        self._queryable = self._session.declare_queryable(
            self._query_key,
            self._on_query,
        )
        self._logger.info(
            "%s started replies_sub=%s query=%s",
            self._log_prefix, self._replies_key, self._query_key,
        )

    def _stop(self) -> None:
        """Undeclare zenoh resources, cancel consumer, close queue."""
        for resource, name in [
            (self._reply_subscriber, "reply_subscriber"),
            (self._queryable, "queryable"),
        ]:
            if resource is not None:
                try:
                    resource.undeclare()
                except RuntimeError:
                    pass
        self._reply_subscriber = None
        self._queryable = None

        if self._consumer is not None:
            self._consumer.cancel()
            self._consumer = None
        if self._event_queue is not None:
            self._event_queue.close()
            self._event_queue = None

        self._logger.info("%s stopped", self._log_prefix)

    # -- consumer ---------------------------------------------------------

    async def _consume(self) -> None:
        """Drain the event queue on the event loop, dispatch replies."""
        self._logger.info("%s consumer started", self._log_prefix)
        try:
            while True:
                answer = await self._event_queue.async_q.get()
                try:
                    qid = answer.meta.refer_to if answer.meta else None
                    if qid is None:
                        self._logger.warning(
                            "%s reply with no refer_to — dropped", self._log_prefix,
                        )
                        continue
                    qa = self._owned_qas.get(qid)
                    if qa is None or qa.done():
                        self._logger.debug(
                            "%s reply for unknown/done qid=%s",
                            self._log_prefix, _short_qid(qid),
                        )
                        continue
                    qa._accept_answer(answer)
                except Exception:
                    self._logger.exception(
                        "%s dispatch reply failed", self._log_prefix,
                    )
        except asyncio.CancelledError:
            pass
        except janus.QueueClosedError:
            pass
        self._logger.info("%s consumer stopped", self._log_prefix)

    # -- zenoh callbacks -------------------------------------------------

    def _on_reply(self, sample: zenoh.Sample) -> None:
        """Deserialise reply on zenoh thread, enqueue for consumer."""
        try:
            data = json.loads(sample.payload.to_bytes())
            answer = Answer.model_validate(data)
        except Exception:
            self._logger.exception(
                "%s failed to deserialise reply", self._log_prefix,
            )
            return
        try:
            self._event_queue.sync_q.put(answer)
        except janus.QueueClosedError:
            self._logger.debug("%s reply queue closed — dropped", self._log_prefix)

    def _on_query(self, query: zenoh.Query) -> None:
        """Respond to late-join query with list of undone questions."""
        undone = self.undone()
        if not undone:
            return
        payload = json.dumps(
            [qa.question.model_dump() for qa in undone],
        )
        query.reply(self._query_key, payload)

    # -- Asker ABC -------------------------------------------------------

    def broadcast_question(self, question: Question) -> QA:
        qid = question.meta.id

        qa = ZenohQA(
            question, self._issuer, owned=True, logger=self._logger,
        )
        self._owned_qas[qid] = qa

        # Verdict broadcast callbacks
        def _broadcast_verdict(answer: Answer) -> None:
            self._logger.info(
                "%s broadcasting verdict qid=%s rejected=%s",
                self._log_prefix, _short_qid(qid), answer.rejected,
            )
            payload = json.dumps({
                'type': 'verdict',
                'qid': qid,
                'answer': answer.model_dump(),
            })
            self._session.put(self._verdicts_key, payload)

        def _broadcast_cancel(question: Question) -> None:
            self._logger.info(
                "%s broadcasting cancel qid=%s reason=%r",
                self._log_prefix, _short_qid(qid), question.canceled,
            )
            payload = json.dumps({
                'type': 'cancel',
                'qid': qid,
                'reason': question.canceled,
            })
            self._session.put(self._verdicts_key, payload)

        qa.on_answer(_broadcast_verdict)
        qa.on_cancel(_broadcast_cancel)

        # Publish question
        self._session.put(self._questions_key, question.model_dump_json())
        self._logger.info(
            "%s question broadcast qid=%s kind=%s",
            self._log_prefix, _short_qid(qid), question.kind,
        )

        return qa


# ============================================================
#  ZenohWatcher
# ============================================================

class ZenohWatcher(Watcher):
    """Watcher that receives questions and verdicts via zenoh.

    Subscribes to ``{prefix}/questions/{ns}`` and ``{prefix}/verdicts/{ns}``.
    Zenoh callbacks run on zenoh's I/O thread and only deserialise + enqueue;
    a consumer task on the event loop drains the queue, creates responder
    :class:`ZenohQA` copies, and fires :meth:`on_question` callbacks.
    """

    def __init__(
        self,
        namespace: str,
        identifier: str,
        *,
        session: zenoh.Session,
        questions_key: str,
        replies_key: str,
        verdicts_key: str,
        logger: LoggerItf | None = None,
    ) -> None:
        self._namespace = namespace
        self._identifier = identifier
        self._session = session
        self._questions_key = questions_key
        self._replies_key = replies_key
        self._verdicts_key = verdicts_key
        self._logger = logger or get_moss_logger()
        self._log_prefix = f"[ZenohWatcher ns={namespace}]"

        self._qas: dict[str, ZenohQA] = {}
        self._on_question_cbs: list[Callable[[QA], None]] = []
        self._question_subscriber: zenoh.Subscriber | None = None
        self._verdict_subscriber: zenoh.Subscriber | None = None

        # janus queue + consumer — offloads dispatch from zenoh I/O thread
        # to the event loop, matching the janus QA isolation pattern.
        self._event_queue: janus.Queue | None = None
        self._consumer: asyncio.Task | None = None

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def identifier(self) -> str:
        return self._identifier

    # -- lifecycle (called by QAManager) ---------------------------------

    def _start(self, spawn: TaskSpawner) -> None:
        self._event_queue = janus.Queue(maxsize=_WATCHER_QUEUE_SIZE)
        self._consumer = spawn(self._consume())

        self._question_subscriber = self._session.declare_subscriber(
            self._questions_key,
            self._on_question_sample,
        )
        self._verdict_subscriber = self._session.declare_subscriber(
            self._verdicts_key,
            self._on_verdict_sample,
        )
        self._logger.info(
            "%s started questions_sub=%s verdicts_sub=%s",
            self._log_prefix, self._questions_key, self._verdicts_key,
        )

    def _stop(self) -> None:
        for resource, name in [
            (self._question_subscriber, "question_subscriber"),
            (self._verdict_subscriber, "verdict_subscriber"),
        ]:
            if resource is not None:
                try:
                    resource.undeclare()
                except RuntimeError:
                    pass
        self._question_subscriber = None
        self._verdict_subscriber = None

        if self._consumer is not None:
            self._consumer.cancel()
            self._consumer = None
        if self._event_queue is not None:
            self._event_queue.close()
            self._event_queue = None

        self._logger.info("%s stopped", self._log_prefix)

    # -- consumer ---------------------------------------------------------

    async def _consume(self) -> None:
        """Drain the event queue on the event loop, dispatch questions/verdicts."""
        self._logger.info("%s consumer started", self._log_prefix)
        try:
            while True:
                kind, payload = await self._event_queue.async_q.get()
                try:
                    if kind == 'question':
                        question = payload
                        qa = ZenohQA(
                            question, self._identifier, owned=False,
                            reply_publisher=self._make_reply_publisher(),
                            logger=self._logger,
                        )
                        self._qas[question.meta.id] = qa
                        self._logger.info(
                            "%s received question qid=%s kind=%s",
                            self._log_prefix, _short_qid(question.meta.id), question.kind,
                        )
                        for cb in self._on_question_cbs:
                            try:
                                cb(qa)
                            except Exception:
                                self._logger.exception(
                                    "%s on_question callback failed", self._log_prefix,
                                )
                    elif kind == 'verdict':
                        data = payload
                        qid = data.get('qid')
                        qa = self._qas.get(qid)
                        if qa is None:
                            self._logger.debug(
                                "%s verdict for unknown qid=%s",
                                self._log_prefix, _short_qid(qid or '?'),
                            )
                            continue
                        vtype = data.get('type')
                        if vtype == 'verdict':
                            try:
                                answer = Answer.model_validate(data['answer'])
                            except Exception:
                                self._logger.exception(
                                    "%s failed to parse verdict answer qid=%s",
                                    self._log_prefix, _short_qid(qid),
                                )
                                continue
                            qa._apply_verdict(answer)
                        elif vtype == 'cancel':
                            qa._apply_cancel(data.get('reason', ''))
                except Exception:
                    self._logger.exception(
                        "%s dispatch event failed", self._log_prefix,
                    )
        except asyncio.CancelledError:
            pass
        except janus.QueueClosedError:
            pass
        self._logger.info("%s consumer stopped", self._log_prefix)

    # -- reply publisher factory -----------------------------------------

    def _make_reply_publisher(self) -> Callable[[Answer], None]:
        """Return a callback that publishes an answer to the replies keyexpr."""
        replies_key = self._replies_key

        def _publish(answer: Answer) -> None:
            self._session.put(replies_key, answer.model_dump_json())

        return _publish

    # -- zenoh callbacks (zenoh I/O thread — deserialise + enqueue only) --

    def _on_question_sample(self, sample: zenoh.Sample) -> None:
        try:
            question = Question.model_validate_json(
                sample.payload.to_string(),
            )
        except Exception:
            self._logger.exception(
                "%s failed to deserialise question", self._log_prefix,
            )
            return
        try:
            self._event_queue.sync_q.put(('question', question))
        except janus.QueueClosedError:
            self._logger.debug(
                "%s question queue closed — dropped", self._log_prefix,
            )

    def _on_verdict_sample(self, sample: zenoh.Sample) -> None:
        try:
            data = json.loads(sample.payload.to_bytes())
        except Exception:
            self._logger.exception(
                "%s failed to deserialise verdict", self._log_prefix,
            )
            return
        try:
            self._event_queue.sync_q.put(('verdict', data))
        except janus.QueueClosedError:
            self._logger.debug(
                "%s verdict queue closed — dropped", self._log_prefix,
            )

    # -- Watcher ABC -----------------------------------------------------

    def questions(self, *, answered: bool = False) -> list[QA]:
        if answered:
            return [qa for qa in self._qas.values() if qa.done()]
        return list(self._qas.values())

    def on_question(self, callback: Callable[[QA], None]) -> None:
        self._on_question_cbs.append(callback)


# ============================================================
#  ZenohQAManager
# ============================================================

class ZenohQAManager(QAManager):
    """Cross-process QA manager backed by zenoh pub/sub.

    One set of keyexprs per namespace.  Subscribers and queryables are
    created lazily when the first Asker / Watcher is created for a
    namespace and cleaned up on ``__aexit__``.

    Construct one instance per cell and use as an async context manager::

        async with ZenohQAManager(
            issuer=address, prefix='MOSS/matrix/scopes/local/qa',
            session=zenoh_session,
        ) as mgr:
            asker = mgr.asker('safemode')
            watcher = mgr.watch('safemode')
            qa = asker.ask_approval('…')
            await qa.wait()
    """

    def __init__(
        self,
        *,
        issuer: str,
        prefix: str,
        session: zenoh.Session,
        logger: LoggerItf | None = None,
    ) -> None:
        self._issuer = issuer
        self._prefix = prefix.rstrip('/')
        self._session = session
        self._logger = logger or get_moss_logger()
        self._log_prefix = "[ZenohQAManager]"

        self._askers: dict[str, ZenohAsker] = {}
        self._watchers: dict[str, list[ZenohWatcher]] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def issuer(self) -> str:
        return self._issuer

    # -- keyexpr builders ------------------------------------------------

    def _questions_key(self, namespace: str) -> str:
        return f"{self._prefix}/questions/{namespace}"

    def _replies_key(self, namespace: str) -> str:
        return f"{self._prefix}/replies/{namespace}"

    def _verdicts_key(self, namespace: str) -> str:
        return f"{self._prefix}/verdicts/{namespace}"

    def _query_key(self, namespace: str) -> str:
        return f"{self._prefix}/query/{namespace}"

    # -- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> ZenohQAManager:
        self._logger.info("%s entering prefix=%s", self._log_prefix, self._prefix)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._logger.info(
            "%s exiting — stopping %d asker(s), %d watcher group(s)",
            self._log_prefix, len(self._askers), len(self._watchers),
        )
        for asker in self._askers.values():
            asker._stop()
        self._askers.clear()
        for watchers in self._watchers.values():
            for watcher in watchers:
                watcher._stop()
        self._watchers.clear()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        self._logger.info("%s exited", self._log_prefix)

    # -- task tracking ----------------------------------------------------

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # -- QAManager ABC ---------------------------------------------------

    def asker(self, namespace: str) -> Asker:
        if namespace in self._askers:
            return self._askers[namespace]
        asker = ZenohAsker(
            issuer=self._issuer,
            namespace=namespace,
            session=self._session,
            questions_key=self._questions_key(namespace),
            replies_key=self._replies_key(namespace),
            verdicts_key=self._verdicts_key(namespace),
            query_key=self._query_key(namespace),
            logger=self._logger,
        )
        asker._start(self._spawn)
        self._askers[namespace] = asker
        return asker

    def watch(self, namespace: str) -> Watcher:
        watcher = ZenohWatcher(
            namespace=namespace,
            identifier=self._issuer,
            session=self._session,
            questions_key=self._questions_key(namespace),
            replies_key=self._replies_key(namespace),
            verdicts_key=self._verdicts_key(namespace),
            logger=self._logger,
        )
        watcher._start(self._spawn)
        self._watchers.setdefault(namespace, []).append(watcher)
        return watcher
