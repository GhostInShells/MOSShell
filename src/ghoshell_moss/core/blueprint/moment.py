import dataclasses
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Any, Tuple, Callable
from typing_extensions import Self
from pydantic import BaseModel, Field, AwareDatetime

from ghoshell_moss.message import Message, WithAdditional
from ghoshell_moss.message import unique_id
from datetime import datetime
from dateutil import tz
import logging

__all__ = [
    'Echoes',
    'Moment',
    'Moments',
    'Observer',
    'BaseMomentsObserver',
    'Logos',
    'Disposer',
    'Epoch',
]

Logos = str


class Echoes(BaseModel, WithAdditional):
    """
    The echoes that return from executing the previous round's logos — body
    feedback, thought steps, tool calls, and so on. Echoes need not be a
    completed outcome; they may be partial, in-progress, or a stop signal.

    Because no current model supports full-duplex interaction, this adapter
    layer is still needed to stitch the pieces back into one exchange.
    """

    moment_id: str = Field(
        default_factory=unique_id,
        description="The id of the preceding Moment.",
    )
    executed_logos: Logos = Field(
        default='',
        description="The logos the system actually executed in the previous round. "
                    "They drive the body and tools. Here logos carries the sense of "
                    "symbol/logic/command/path/law-of-reality — the Dao spoken into words. "
                    "The system-executed logos is not identical to the model-generated logos.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="The internal (body) feedback received during or after executing the logos — "
                    "the echoes in the thinking cave.",
    )
    need_observe: bool = Field(
        default=False,
        description="Whether the next observation is required.",
    )
    stop_reason: str = Field(
        default='',
        description="If this is an unfinished Moment, it can record its state here.",
    )

    def is_empty(self) -> bool:
        """Whether no echo messages were received this round."""
        if len(self.messages) == 0:
            return True
        for message in self.messages:
            if not message.is_empty():
                return False
        return True

    def add_echoes(self, messages: list[Message | str], need_observe: bool = False) -> list[Message]:
        """Normalize and append non-empty echoes, setting need_observe if any appended.

        Returns the normalized Messages that were actually appended (input strings are
        converted to Messages here), so observers always receive Messages — matching the
        echo-added callback signature exactly.
        """
        appended: list[Message] = []
        for msg in messages:
            message = None
            if isinstance(msg, str):
                message = Message.new().with_content(msg)
            elif isinstance(msg, Message):
                message = msg
            if message and not message.is_empty():
                self.messages.append(message)
                appended.append(message)
        # need_observe 参数无条件生效, 与是否 append 到消息无关 — 空消息也可能
        # 只发 observe 信号不带内容 (见 _notify_moments_need_observe).
        if need_observe:
            self.need_observe = True
        return appended

    def new_moment(
            self,
            *,
            dynamic_context: dict[str, list[Message]] | None = None,
            percepts: dict[str, list[Message]] | None = None,
            command_logos: str = '',
            hint: str = '',
    ) -> "Moment":
        """
        Build the next Moment from this round's echoes.

        hint only takes effect in this round and is not persisted.
        """
        return Moment(
            previous=self,
            percepts=percepts or {},
            dynamic_context=dynamic_context or {},
            hint=hint,
            command_logos=command_logos,
        )


class Moment(BaseModel, WithAdditional):
    """
    A keyframe of the agent's context-aware perception — "an instant".

    Each keyframe stitches the previous round's output into the current round's
    input position. The split points are each round's output (logos) and the
    observe action.
    """

    id: str = Field(
        default_factory=unique_id,
        description="A unique id for this observation.",
    )

    # --- stitching the previous round's info --- #
    previous: Echoes | None = Field(
        default=None,
        description="The stitched echoes from the previous round — the seam connecting "
                    "this frame to the prior one.",
    )

    # --- the inputs for the new round --- #

    dynamic_context: dict[str, list[Message]] = Field(
        default_factory=dict,
        description="A dynamic context snapshot produced at the instant this Moment is "
                    "generated, merging the different context types.",
    )
    percepts: dict[str, list[Message]] = Field(
        default_factory=dict,
        description="This round's external inputs, keyed by source. Each source overwrites, "
                    "so duplicates are naturally removed. Aligned with dynamic_context: within "
                    "one frame, only the latest value per source is kept. Moment instances are "
                    "independent across turns and do not accumulate across frames. After JSON "
                    "deserialization the dict order is lost — the storage layer must rebuild it "
                    "in insertion order (not persisted yet, reserved).",
    )
    hint: str = Field(
        default='',
        description="A hint that only takes effect in this round.",
    )
    command_logos: Logos = Field(
        default='',
        description="Per system convention, the logos to execute before the brain starts "
                    "thinking. Consider how to surface them to the model.",
    )
    logos: Logos = Field(
        default='',
        description="The logos generated by the decision module for this keyframe — "
                    "usually from the model.",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="The time this conversation was created.",
    )

    def to_dict(self) -> dict[str, Any]:
        """View the moment data as a dict; more behaviors are on BaseModel."""
        return self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            mode='json',
        )

    def for_saving(self) -> 'Moment':
        """Prepare a copy for persistence — dynamic context and hint are transient and excluded."""
        return self.model_copy(
            update={'dynamic_context': {}, 'hint': ''},
        )

    def to_json(
            self,
            *,
            exclude_dynamic_context: bool = True,
            exclude_hint: bool = True,
            indent: int = 0,
    ) -> str:
        """The canonical serialization form, convenient for storage."""
        exclude: set[str] = set()
        if exclude_dynamic_context:
            exclude.add('dynamic_context')
        if exclude_hint:
            exclude.add('hint')
        return self.model_dump_json(
            exclude=exclude or None,
            indent=indent,
            ensure_ascii=False,
            exclude_none=True,
            exclude_defaults=True,
        )

    def new_echoes_container(self) -> Echoes:
        """Create the Echoes container that will receive the next round's echoes."""
        return Echoes(
            moment_id=self.id,
        )

    def previous_stop_reason(self) -> str | None:
        if self.previous:
            return self.previous.stop_reason or None
        return None

    def previous_executed_logos(self) -> str:
        if self.previous is None:
            return ''
        return self.previous.executed_logos

    def with_dynamic_context(self, key: str, messages: list[Message]) -> Self:
        """Merge a dynamic-context type. The key carries no content; it only dedups repeated updates."""
        self.dynamic_context[key] = messages
        return self

    def with_percepts(self, source: str, messages: list[Message]) -> Self:
        """Write source-keyed percepts. Same source overwrites, dedup naturally."""
        messages = [msg for msg in messages if not msg.is_empty()]
        if messages:
            self.percepts[source] = messages
        return self

    def percepts_messages(self) -> Iterable[Message]:
        """Flatten all sources' percepts in insertion order."""
        for messages in self.percepts.values():
            for msg in messages:
                if not msg.is_empty():
                    yield msg

    def percepts_texts(self) -> list[str]:
        """Flatten all percepts and extract plain text, for test assertions and debugging."""
        return [msg.to_content_string() for msg in self.percepts_messages()]

    def last_moment_id(self) -> str | None:
        if self.previous is None:
            return None
        return self.previous.moment_id

    # --- code-as-prompt: how the various fields combine --- #

    def dynamic_context_messages(self) -> Iterable[Message]:
        """
        The keyframe messages handed to the thinking module for this instant.

        This is a cognitive sliding window — like a screen for a human: only the
        latest frame is fed in.
        """
        if len(self.dynamic_context) == 0:
            yield from []
            return
        for messages in self.dynamic_context.values():
            yield from messages

    def previous_echoes_messages(self) -> Iterable[Message]:
        if self.previous is None:
            yield from []
            return
        result = self.previous
        if len(result.messages) > 0:
            yield from result.messages
        if result.stop_reason:
            yield Message.new(tag='stop_reason').with_content(result.stop_reason)

    def is_empty(self) -> bool:
        return self.is_echoes_empty() and self.is_percepts_empty()

    def is_percepts_empty(self) -> bool:
        return all(len(v) == 0 for v in self.percepts.values())

    def is_echoes_empty(self) -> bool:
        return self.previous is None or len(self.previous.messages) == 0

    def inputs_messages(
            self,
            with_hint: bool = True,
            with_command_executing: bool = True,
    ) -> Iterable[Message]:
        """Alias so that percepts read naturally as the agent's input messages."""
        yield from self.percepts_messages()
        if with_command_executing and self.command_logos:
            yield Message.new(tag='executing').with_content(self.command_logos)
        if with_hint and self.hint:
            yield Message.new(tag='hint').with_content(self.hint)

    def full_moment_messages(
            self,
            *,
            with_dynamic_context: bool = True,
            with_hint: bool = True,
            with_command_executing: bool = True,
            with_percepts: bool = True,
    ) -> list[Message]:
        """
        Fold the whole keyframe into a single moment message.

        Wraps this frame's echoes / dynamic_context / percepts / executing / hint
        into one ``<moment moment_id=...>`` message carrying the moment id, so it can be
        inserted as a single tool-result or user message. The transient sub-blocks
        (dynamic_context / command_logos / hint) belong to the current frame only;
        history should drop them via ``as_history_messages``.

        ``with_percepts=False`` folds the context half only (echoes / dynamic_context /
        command_logos), leaving percepts for the separate input message.
        """
        messages: list[Message] = []
        echoes = list(self.previous_echoes_messages())
        if echoes:
            messages.append(Message.new(tag='echoes').with_messages(*echoes))

        if with_dynamic_context:
            dynamic_messages = list(self.dynamic_context_messages())
            if dynamic_messages:
                messages.append(
                    Message.new(tag='dynamic_context').with_messages(*dynamic_messages)
                )
        if with_percepts:
            percepts = list(self.percepts_messages())
            if percepts:
                percepts_message = Message.new(tag='percepts').with_messages(*percepts)
                messages.append(
                    percepts_message
                )
        if with_command_executing and self.command_logos:
            messages.append(
                Message.new(tag='executing').with_content(self.command_logos)
            )
        if with_hint and self.hint:
            messages.append(
                Message.new(tag='hint').with_content(self.hint)
            )
        if not messages:
            return []
        return messages

    def as_moment_message(
            self,
            *,
            always_return: bool = True,
            with_moment_id: bool = True,
            with_percepts: bool = True,
            with_hint: bool = True,
    ) -> Message | None:
        """Fold the keyframe into one ``<moment moment_id=...>`` message.

        ``with_percepts=False`` / ``with_hint=False`` drop the input half (percepts / hint),
        leaving the context half (echoes / dynamic_context / command_logos). Callers that
        want the input half as a separate message assemble it from ``percepts_messages()``
        and ``hint`` themselves.
        """
        messages = self.full_moment_messages(with_percepts=with_percepts, with_hint=with_hint)
        if messages or always_return:
            attributes = {'moment_id': self.id} if with_moment_id else {}
            return Message.new(tag='moment', attributes=attributes).with_messages(*messages)
        return None

    def as_history_messages(self) -> Iterable[Message]:
        """When used as history, this drops all dynamic messages. A code-as-prompt."""
        yield from self.previous_echoes_messages()
        yield from self.percepts_messages()

    @classmethod
    def to_history_turns(
            cls,
            moments: Iterable['Moment'],
    ) -> Iterable[Tuple[list[Message], Logos | None]]:
        """
        Turn a coherent run of moments into turn-based history.

        The model-generated logos divide the turns. If a round produced no model
        logos but still executed something, the executed logos are stitched in.
        """
        buffered_messages = []
        last_moment_has_logos = False
        for moment in moments:
            # If the previous round had no model-generated logos but there are
            # executed logos, stitch them into history. A Ghost may execute a
            # command where thinking is disallowed, yielding no logos. If the
            # executed logos are not recorded, they become pure hallucination.
            if not last_moment_has_logos:
                if executed_logos := moment.previous_executed_logos():
                    buffered_messages.append(Message.new(tag='executed').with_content(executed_logos))
            # Load the history messages bridging the previous response and this input.
            buffered_messages.extend(moment.as_history_messages())
            # Does this round carry logos?
            this_moment_has_logos = len(moment.logos) > 0
            if buffered_messages or this_moment_has_logos:
                # A round only splits a turn when it carries logos; otherwise it
                # just keeps buffering history. An observation may legitimately
                # hold empty data, so an empty buffer with logos still yields.
                if this_moment_has_logos:
                    yield buffered_messages, moment.logos
                    buffered_messages = []
            last_moment_has_logos = this_moment_has_logos
        if buffered_messages:
            yield buffered_messages, None


Disposer = Callable[[], None]


@dataclasses.dataclass(frozen=True)
class Epoch:
    id: str  # uuid
    index: int  # 生成的第几轮.
    recap: list[Message]  # 前情提要数据.


class Moments(ABC):
    """
    The observation container for moments (in memory).

    Unlike a data store, new moment data is produced only when an 'observe'
    action happens.
    """

    @property
    @abstractmethod
    def epoch(self) -> Epoch:
        """The recap of the moments after compaction."""
        ...

    @abstractmethod
    def with_epoch_recap(self, key: str, recap_func: Callable[[], list[Message]]) -> Disposer:
        ...

    @abstractmethod
    def on_epoch_created(self, callback: Callable[[Epoch], None]) -> Disposer:
        """当新的 epoch 创建时, 追加 recap 数据"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear held data and reset state."""
        ...

    @abstractmethod
    def moments(self, peek: bool = False) -> list[Moment]:
        """All produced moments. Usually capped by maxsize; older ones are discarded beyond it."""
        ...

    @abstractmethod
    def peek(self) -> Moment:
        """The current Moment without it entering the moments history."""
        ...

    @abstractmethod
    def with_dynamic_context_func(self, key: str, func: Callable[[], Iterable[Message]]) -> Disposer:
        """Register a dynamic-context function, evaluated each time a Moment is generated."""
        ...

    @abstractmethod
    def with_percepts_buffer(self, key: str, drain: Callable[[], Iterable[Message]]) -> Disposer:
        """Register a percepts drain function, called whenever a Moment is generated."""
        ...

    @abstractmethod
    def with_echoes_drain(self, key: str, drain: Callable[[], tuple[list[Message], bool]]) -> Disposer:
        ...

    @abstractmethod
    def inject_percepts(self, *messages: Message | str) -> None:
        """Add new percepts, to be produced on the next observe."""
        ...

    def turns(self, peek: bool = False) -> Iterable[tuple[list[Message], Logos | None]]:
        """Organize all moment data as (inputs, logos) turns; excludes recap."""
        yield from Moment.to_history_turns(self.moments(peek=peek))

    @abstractmethod
    def add_executed_logos(self, logos: str) -> None:
        ...

    @abstractmethod
    def add_echoes(self, result: list[Message | str], need_observe: bool = False) -> None:
        """Add echoes, which appear in the next Moment, and mark whether to observe."""
        ...

    @abstractmethod
    def on_moment_created(self, callback: Callable[[Moment], None]) -> Disposer:
        """Register a moment callback, invoked each time a Moment is produced.

        Useful for storage queues or building a multi-sided observer.
        """
        ...

    @abstractmethod
    def on_echoes_add(self, callback: Callable[[list[Message], bool], None]) -> Disposer:
        """Callback when echoes are added.

        The observer must react to the observe signal; it cannot poll, so it is
        notified through this method.
        """
        ...


class Observer(Moments, ABC):
    """An observer over the Moments history; it must have exactly one user."""

    @abstractmethod
    def need_observe(self) -> bool:
        """Whether observation is needed."""
        ...

    @abstractmethod
    def observe(self) -> Moment:
        """Produce a Moment at the instant of observation; only observe generates Moments."""
        ...

    @abstractmethod
    def new_epoch(
            self,
            recap: list[Message],
            end_moment_id: str | None = None,
    ) -> Epoch:
        """生成新的 epoch"""
        ...


class BaseMomentsObserver(Observer):

    def __init__(self, max_size: int, logger: logging.Logger | None = None) -> None:
        self._moments: deque[Moment] = deque()
        self._max_moments_size = max_size
        self._recap: list[Message] = []
        self._echoes = Echoes()
        self._moment_created_callbacks: set[Callable[[Moment], None]] = set()
        self._echoes_added_callbacks: set[Callable[[list[Message], bool], None]] = set()
        self._epoch_created_callbacks: set[Callable[[Epoch], None]] = set()

        self._dynamic_context_funcs: dict[str, Callable[[], Iterable[Message]]] = dict()
        self._percepts_drain_funcs: dict[str, Callable[[], Iterable[Message]]] = dict()
        self._drain_new_echoes_funcs: dict[str, Callable[[], tuple[list[Message], bool]]] = dict()
        self._epoch_recap_funcs: dict[str, Callable[[], list[Message]]] = dict()

        self._buffered_drained_percepts = {}
        self._injected_percepts = []
        self._logger: logging.Logger = logger or logging.getLogger(__name__)
        self._current_epoch: Epoch | None = None

    @property
    def epoch(self) -> Epoch:
        if self._current_epoch is None:
            return self.new_epoch([])
        return self._current_epoch

    def new_epoch(self, recap: list[Message], end_moment_id: str | None = None) -> Epoch:
        index = 0
        if self._current_epoch is not None:
            index = self._current_epoch.index
        self.compact(end_moment_id)
        index += 1
        uid = unique_id()
        epoch = Epoch(
            id=uid,
            index=index,
            recap=recap,
        )
        self._current_epoch = epoch
        if len(self._epoch_recap_funcs) > 0:
            for func in self._epoch_recap_funcs.values():
                try:
                    messages = func()
                    if messages:
                        epoch.recap.extend(messages)
                except Exception as e:
                    self._logger.exception(e)

        if len(self._epoch_created_callbacks) > 0:
            for callback in self._epoch_created_callbacks:
                try:
                    callback(epoch)
                except Exception as e:
                    self._logger.exception(e)
        return epoch

    def compact(self, end_moment_id: str | None) -> None:
        while len(self._moments) > 0:
            moment = self._moments.popleft()
            if moment.id == end_moment_id:
                break

    def with_epoch_recap(self, key: str, recap_func: Callable[[], list[Message]]) -> Disposer:
        self._epoch_recap_funcs[key] = recap_func

        def _dispose() -> None:
            value = self._epoch_recap_funcs.get(key)
            if value is recap_func:
                self._epoch_recap_funcs.pop(key)

        return _dispose

    def on_epoch_created(self, callback: Callable[[Epoch], None]) -> Disposer:
        self._epoch_created_callbacks.add(callback)

        def _dispose() -> None:
            self._epoch_created_callbacks.discard(callback)

        return _dispose

    def moments(self, peek: bool = False) -> list[Moment]:
        result = list(self._moments)
        if peek:
            result.append(self._peek_moment())
        return result

    def need_observe(self) -> bool:
        return self._echoes.need_observe

    def peek(self) -> Moment:
        return self._peek_moment()

    def _peek_moment(self) -> Moment:
        moment = self._echoes.new_moment()
        if len(self._drain_new_echoes_funcs) > 0:
            for key, func in self._drain_new_echoes_funcs.items():
                try:
                    messages, observe = func()
                    # Echoes 自己持有 drain: 直接喂给 moment.previous (即 self._echoes),
                    # 无需 observer 级缓冲. 空消息也会置位 need_observe (add_echoes 已处理).
                    moment.previous.add_echoes(messages, observe)
                except Exception as e:
                    self._logger.error(e)
        if len(self._dynamic_context_funcs) > 0:
            for key, func in self._dynamic_context_funcs.items():
                try:
                    messages = func()
                    moment.with_dynamic_context(key, list(messages))
                except Exception as e:
                    self._logger.error(e)
        if len(self._percepts_drain_funcs) > 0:
            for key, func in self._percepts_drain_funcs.items():
                try:
                    messages = func()
                    if messages:
                        buffer = self._buffered_drained_percepts.get(key, [])
                        buffer.extend(messages)
                        moment.with_percepts(key, buffer)
                except Exception as e:
                    self._logger.error(e)
        if len(self._injected_percepts) > 0:
            moment.with_percepts("MomentsInjectedPercepts", self._injected_percepts)
        return moment

    def observe(self) -> Moment:
        moment = self._peek_moment()
        self._echoes = moment.new_echoes_container()
        self._buffered_drained_percepts = {}
        self._injected_percepts = []
        self._moments.append(moment)
        while len(self._moments) > self._max_moments_size:
            self._moments.popleft()

        if len(self._moment_created_callbacks) > 0:
            for callback in self._moment_created_callbacks:
                callback(moment)
        return moment

    def clear(self) -> None:
        self._recap = []
        self._moments.clear()
        self._echoes = Echoes()
        self._buffered_drained_percepts = {}
        self._injected_percepts = []

    def on_moment_created(self, callback: Callable[[Moment], None]) -> Disposer:
        self._moment_created_callbacks.add(callback)

        def _disposer():
            if callback in self._moment_created_callbacks:
                self._moment_created_callbacks.discard(callback)

        return _disposer

    def on_echoes_add(self, callback: Callable[[list[Message], bool], None]) -> Disposer:
        self._echoes_added_callbacks.add(callback)

        def _disposer():
            if callback in self._echoes_added_callbacks:
                self._echoes_added_callbacks.discard(callback)

        return _disposer

    def with_dynamic_context_func(self, key: str, func: Callable[[], Iterable[Message]]) -> Disposer:
        self._dynamic_context_funcs[key] = func

        def _disposer():
            if key in self._dynamic_context_funcs:
                value = self._dynamic_context_funcs.get(key)
                if value is func:
                    self._dynamic_context_funcs.pop(key)

        return _disposer

    def with_echoes_drain(self, key: str, drain: Callable[[], tuple[list[Message], bool]]) -> Disposer:
        self._drain_new_echoes_funcs[key] = drain

        def _disposer():
            if key in self._drain_new_echoes_funcs:
                value = self._drain_new_echoes_funcs.get(key)
                if value is drain:
                    self._drain_new_echoes_funcs.pop(key)

        return _disposer

    def with_percepts_buffer(self, key: str, drain: Callable[[], Iterable[Message]]) -> Disposer:
        self._percepts_drain_funcs[key] = drain

        def _disposer():
            if key in self._percepts_drain_funcs:
                value = self._percepts_drain_funcs.get(key)
                if value is drain:
                    self._percepts_drain_funcs.pop(key)

        return _disposer

    def add_echoes(self, result: list[Message | str], need_observe: bool = False) -> None:
        appended = self._echoes.add_echoes(result, need_observe)
        if appended and len(self._echoes_added_callbacks) > 0:
            for callback in self._echoes_added_callbacks:
                try:
                    callback(appended, need_observe)
                except Exception as e:
                    self._logger.error(e)

    def add_executed_logos(self, logos: str) -> None:
        self._echoes.executed_logos += logos

    def inject_percepts(self, *messages: Message | str) -> None:
        percepts = []
        for message in messages:
            if isinstance(message, str):
                percepts.append(Message.new().with_content(message))
            elif isinstance(message, Message):
                percepts.append(message)
        self._injected_percepts.extend(percepts)
