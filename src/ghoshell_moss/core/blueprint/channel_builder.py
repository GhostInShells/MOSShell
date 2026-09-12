"""
How to build a Channel — the entry point for channel construction in MOSShell.
module path: `ghoshell_moss.core.blueprint.channel_builder`

A Channel is the capability container a model drives. This module is the verb side:
``Builder`` / ``new_channel`` construct one, ``CommandUtil`` is what a command calls,
``ChannelMeta`` (in `core.concepts.channel`) is the noun side and the authority on what
the runtime model sees. Build against this surface; do not read the implementation
(`core.py_channel`) unless you hit a bug.

Three things to keep straight while building:

- direction: a command's result, progress and signal reach the model in three
  directions. See ``CommandUtil``.
- tier: what a channel exposes to the model is cold (``instruction``), warm
  (``notice``) or hot (``context``). Each decorator below marks the tier it feeds.
- channel path: a channel may be mounted on different trees and renamed — never restate
  your own name, CTML tag or channel path in an ``instruction``, a command docstring, or
  a returned message; name commands, not paths.

The runtime model drives a channel through CTML: `moss ctml read`.
"""

from abc import ABC, abstractmethod

from PIL import Image
from typing import Union, Callable, Coroutine, Any, Optional, TypeVar, AsyncIterable, Type

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss.core import ChannelRuntime
from ghoshell_moss.message import Message, Base64Image
from ghoshell_moss.core.concepts.command import Command, Observe, ObserveError
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import TOPIC_MODEL, Publisher, Subscriber
from ghoshell_moss.core.blueprint.mindflow import Signal
import asyncio

Facade = ABC
"""Facade marks a "consumption surface" abstraction — you hold it and call it, you do not inherit it."""

__all__ = [
    "Channel", "ChannelFactory",
    "CommandFunction", "MessageFunction", "StringType", "LifecycleFunction",
    "Message",
    "MessageType",
    "Builder",
    "MutableChannel",
    "new_channel", "new_command",
    "CommandUtil",
    "Observe", "ObserveError",

    # Implementation styles kept as examples.
    "ChannelInterface", "ChannelCreator",
]

ChannelFactory = Callable[[IoCContainer], Channel | None]

CommandFunction = Union[Callable[..., Coroutine], Callable[..., Any]]
"""
A local python function (or class method) that can be registered on a Channel and
become a command.
"""

MacroFunction = Union[Callable[..., Coroutine[None, None, str]], Callable[..., str]]
"""A macro: a function that generates new Command Token syntax strings."""

MessageType = Message | str | Image.Image
MessageFunction = Union[
    Callable[[], Coroutine[None, None, list[MessageType]]],
    Callable[[], list[MessageType]],
]
"""
Functions that produce message bodies. Registered on a Channel, they generate Context
Messages and Memory Messages dynamically: over the duplex channel, at the instant of each
keyframe's thought, the AI pulls the corresponding message body and substitutes it into
the context.
"""

# The Memory Messages field already exists on `ChannelMeta.memory`, but the Builder has
# no mount point for it yet.


StringType = Union[
    str,
    Callable[[], str],
    Callable[[], Coroutine[None, None, str]],
]

LifecycleFunction = Union[Callable[..., Coroutine[None, None, None]], Callable[..., None]]
"""A local python function (or class method) that defines a lifecycle behavior of the
Channel itself.

The runtime lifecycle of a Channel is:

- [on startup]: when the channel starts
- [on idle]: while idle, with no command input
- [on close]: when the channel closes
- [on running]: start < running < close

``refresh_meta`` is a separate hook — a refresh rhythm, not a runtime lifecycle: it runs
before metas are regenerated on each refresh cycle. See Builder.refresh_meta.

A typical example: a digital human plays its locomotion animation while an animation
command runs; when the command finishes and there is no input, it returns to a breathing
idle effect (on_idle).
"""

_ChannelName = str

INSTANCE = TypeVar("INSTANCE", bound=object)


class CommandUtil:
    """
    Tools used inside a Command, available ONLY while a Command or Channel Lifecycle
    Function is executing. The caller's capabilities are obtained through the
    contextlib ctx. Collects the common APIs a command function needs.
    """

    @classmethod
    def force_get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        """
        force get contract from ioc Container.
        raise Error if the contract is not registered.
        combine with moss manifests to know existing contracts
        """
        # dig deeper only when necessary
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        return ChannelCtx.get_contract(contract)

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE | None:
        """
        if contract is not registered, return None
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        return runtime.container.get(contract)

    @classmethod
    def runtime(cls) -> ChannelRuntime:
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        if runtime is None:
            cls.raise_observe('CommandUtil is not in the runtime context')
        return runtime

    @classmethod
    def enabled(cls) -> bool:
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        return runtime is not None

    @classmethod
    def topic_publisher(cls, model: Type[TOPIC_MODEL]) -> Publisher[TOPIC_MODEL]:
        """
        return a topic publisher, use it in with statement:

        async with CommandUtil.topic_publisher(model) as publisher:
            await publisher.publish(topic_model: TOPIC_MODEL)

        or managed within startup -> close lifecycle of the channel
        """
        runtime = cls.runtime()
        return runtime.topic_publisher(model)

    @classmethod
    def topic_subscriber(cls, model: Type[TOPIC_MODEL], *, maxsize: int = 1000) -> Subscriber[TOPIC_MODEL]:
        """
        return a topic subscriber, use it in with statement:

        async with CommandUtil.topic_subscriber(model) as subscriber:
            model = await subscriber.poll()

        or managed within startup -> close lifecycle of the channel
        """
        runtime = cls.runtime()
        return runtime.topic_subscriber(model, maxsize=maxsize)

    @classmethod
    def logger(cls):
        """Return the logging.Logger module, keeping only the basic recording functions."""
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        from ghoshell_moss.contracts import LoggerItf, get_moss_logger
        return ChannelCtx.container().get(LoggerItf) or get_moss_logger()

    @classmethod
    def set_progress(cls, progress: str) -> None:
        """set progress of the current command task (ONLY when needed to do so)"""
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        if task is not None:
            task.set_progress(progress)

    @classmethod
    def observe(cls, value: str) -> 'str | Observe':
        """Return information that must be observed immediately. It actually returns an
        Observe object, but a command may declare its return type as str."""
        return Observe(messages=[Message.new().with_content(value)])

    @classmethod
    def observe_image(cls, text: str, image: Image.Image, *, format: str = "JPEG") -> 'Observe':
        """Return a text + image observation that must be observed immediately.

        Like ``observe``, it returns an ``Observe`` while the command may declare its
        return type as ``str``. The image is embedded as a ``Base64Image`` content part.
        ``format`` is the PIL save format (default JPEG keeps frames compact).
        """
        return Observe(messages=[
            Message.new().with_content(text, Base64Image.from_pil_image(image, format=format))
        ])

    @classmethod
    def raise_observe(cls, value: str) -> None:
        """Raise an observation that interrupts other logic running in the command."""
        raise cls.observe_error(value)

    @classmethod
    def observe_error(cls, value: str) -> 'ObserveError':
        from ghoshell_moss.core.concepts.command import ObserveError
        return ObserveError(value)

    @classmethod
    def reraise_stopped(cls, message: str) -> None:
        """Report progress when a command is cancelled/interrupted: rewrite the cancel
        into a STOPPED (301) CommandError.

        A command function calls this inside its ``except asyncio.CancelledError``
        branch, raising a STOPPED error that carries the message. It is recorded by
        is_notifiable as a model-readable message, but code < 400 triggers no observe and
        does not interrupt interpretation. This method never returns.
        """
        raise CommandErrorCode.STOPPED.error(message)

    @classmethod
    def send_signal(cls, signal: Signal) -> None:
        """
        Send a signal to your own brain from inside a command, closing the self-driven
        loop. To send other kinds of Signal, see the SignalMeta protocol from service
        discovery.
        """
        from ghoshell_moss.core.blueprint.session import Session
        session = cls.force_get_contract(Session)
        if isinstance(signal, Signal):
            session.add_signal(signal)
        else:
            raise TypeError(f"only Signal or str is accepted")

    @classmethod
    def create_task(cls, coroutine) -> asyncio.Task:
        """
        create an asyncio task in channel lifecycle.
        useful for some task going on after command itself done
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        return runtime.create_asyncio_task(coroutine)

    @classmethod
    def is_task_done(cls) -> bool:
        """
        Whether the task that triggered the current command has already finished.
        Convenient for state cleanup inside a synchronous function.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        return task.done()

    @classmethod
    def get_task_context(cls) -> dict[str, Any]:
        """
        Return the arguments passed in from the environment when the task was created.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        return task.context

    @classmethod
    def send_input_signal(cls, content: str, *, description: str = '') -> None:
        """Send a standard request signal to the ghost."""
        from ghoshell_moss.core.blueprint.session import Session
        session = cls.force_get_contract(Session)
        session.add_input_signal(content, description=description)

    @classmethod
    async def create_signal_task(
            cls,
            *,
            closure: Callable[[], Coroutine[None, None, Signal | str]],
    ) -> None:
        """
        Create an async Signal callback task inside a Command without blocking the
        Command's return. When the closure finishes asynchronously, the resulting Signal
        is sent to the ghost.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        caller = task.caller_name()
        task_ctx = task.context

        async def _send_signal_after_task_done() -> None:
            nonlocal closure, task_ctx, caller
            signal = await closure()
            if isinstance(signal, Signal):
                cls.send_signal(signal)
            elif isinstance(signal, str):
                cls.send_input_signal(signal)
            else:
                cls.logger().error(
                    "signal task returns invalid signal type: %s, task %s, task context %s,",
                    signal, caller, task_ctx
                )

        runtime = ChannelCtx.runtime()
        runtime.create_asyncio_task(_send_signal_after_task_done())


def new_command(
        func: CommandFunction,
        *,
        name: str = "",
        doc: Optional[StringType] = None,
        comments: Optional[StringType] = None,
        interface: Optional[StringType | Callable[[...], Coroutine[None, None, Any]]] = None,
        available: Optional[Callable[[], bool]] = None,
        # --- advanced parameters --- #
        blocking: bool = True,
        call_soon: bool = False,
        priority: int = 0,
        always_observe: bool = False,
        timeout: Optional[float] = None,
        visible: bool = True,
        macro: bool = False,
) -> Command:
    """
    Define a Command. Same logic as Builder.command.

    Prefer this when reflecting a python function into a Command in scenarios like
    own_commands (equivalent to PyCommand, but the implementation is replaceable).
    """
    from ghoshell_moss.core.concepts.command import PyCommand
    return PyCommand(
        func=func,
        name=name,
        doc=doc,
        comments=comments,
        interface=interface,
        available=available,
        blocking=blocking,
        call_soon=call_soon,
        priority=priority,
        always_observe=always_observe,
        timeout=timeout,
        visible=visible,
        macro=macro,
    )


class Builder(Facade):
    """
    The general interface for dynamically building a Channel.

    A Builder has a unique id and is side-effecting. One instance should be used only
    once.
    """

    # ---- decorators ---- #
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def available(self, func: Callable[[], bool]) -> Callable[[], bool]:
        """
        decorator
        Register a function that dynamically produces the whole Channel's available
        state. The Channel reads from it on every state refresh; defaults to True
        otherwise.
        >>> async def building(chan: MutableChannel) -> None:
        >>>     chan.build.available(lambda: True)
        """
        pass

    @abstractmethod
    def instruction(self, func: StringType) -> StringType:
        """
        decorator
        Register a string or a function that produces this channel's instruction /
        system prompt. Generated once.

        Cold data: sent once, never re-sent.

        Red line: never restate which commands this channel has — command signatures are
        already reflected to the model by interface (Code as Prompt), and a hand-written
        list drifts into a lie as the code changes. Write only what interface cannot
        express: overall usage, collaboration conventions, state semantics.

        The full contract of what the channel exposes to the model is authoritative in
        ChannelMeta (ghoshell_moss.core.concepts.channel:ChannelMeta).

        Note: a Channel should provide instruction only when genuinely necessary. Most
        channels need none at all.
        """
        # Channel as Context Components 思想:
        #     直接将 Channel 作为上下文的组件, 提供模块化的上下文讯息.
        #     讯息应该足够简洁, 高效, 同时注意 token 用量. 具体裁剪和压缩由 Agent 工程决定.
        #     由于 Channel 持有的 Command 可以影响自身的运行时状态, 所以 Channel 提供了完整的上下文反身性.
        #     结合后续的 StatefulChannel 实现, 同时提供渐进式披露的能力.
        pass

    @abstractmethod
    def context_messages(self, func: MessageFunction, reset: bool = False) -> MessageFunction:
        """
        decorator
        Register a context generator function that produces the channel's runtime
        dynamic context. For example, a vision module can supply the frame it currently
        sees, plus a short description, as context messages.

        Hot data: regenerated on every refresh.

        Usually only perception modules need dynamic context messages.

        >>> async def building(chan: MutableChannel) -> None:
        >>>     async def context() -> list[Message]:
        >>>         return [
        >>>             Message.new().with_content("dynamic information")
        >>>         ]
        >>>     chan.build.context_messages(context)
        """
        pass

    @abstractmethod
    def notice(self, func: StringType) -> StringType:
        """
        decorator
        Register a function that produces the channel's warm notice description.

        notice describes what this channel currently exposes — distinct from description
        (static identity) and context_messages (hot state data). It is evaluated on every
        meta refresh and rendered before the command interface.

        Warm data: re-sent only when it changes.

        Red line: notice answers "what can it do"; context answers "what is it now".
        """
        pass

    def content_command(
            self,
            func: Callable[[AsyncIterable[str]], Coroutine[None, None, None | str]],
            doc: Optional[str] = None,
            override: bool = True,
    ) -> Command[None | str]:
        """
        Register a special function as the channel's content method.
        """
        name = ChannelRuntime.__content__.__name__
        return self.command(
            name=name,
            # Allows overriding the description.
            doc=doc,
            # use __content__ as interface, override the docstring if need.
            interface=ChannelRuntime.__content__,
            override=override,
            return_command=True,
        )(func)

    @abstractmethod
    def add_command(
            self,
            command: Command,
            *,
            override: bool = True,
            name: Optional[str] = None,
    ) -> None:
        """
        Add an existing Command object.
        """
        pass

    @abstractmethod
    def command(
            self,
            *,
            name: str = "",
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[list[str]] = None,
            interface: Optional[StringType | Callable[[...], Coroutine[None, None, Any]]] = None,
            available: Optional[Callable[[], bool]] = None,
            override: bool = True,
            # --- advanced parameters --- #
            blocking: bool = True,
            call_soon: bool = False,
            priority: int = 0,
            return_command: bool = False,
            always_observe: bool = False,
            timeout: float | None = None,
            visible: bool = True,
            macro: bool = False,
    ) -> Callable[[CommandFunction], CommandFunction | Command]:
        """
        decorator
        Register a python function or class method on the Channel as one of its Commands.
        The function's signature is reflected automatically as information for the model
        to read. The model sees only the signature and docstring — never the source code.

        Design the docstring for the runtime model, not a human developer: this text IS
        the runtime model's prompt (Code as Prompt). Make it readable at a glance — clear
        parameter semantics, concrete examples, no developer jargon.

        :param name: if non-empty, rename the function.
        :param doc: override the function docstring. If a function is passed, it is called
                on every refresh to generate the docstring dynamically.
        :param comments: override the function body with a comment-form string. Every line
                is automatically prefixed with '#'. The most direct use is writing usage
                examples, explanations and execution logic to help the model understand.

        :param interface: the function-code form the model sees. Once set, doc, name and
                comments all become inert. Three ways to pass it:
                - str: define the model-visible signature directly as a string.
                    It must be written in Python async form:
                    async def foo(...) -> ...:
                      '''docstring'''
                      # comments
                - callalble[[], str]: a function that generates the model-visible signature
                - async function: reflect this function directly to generate the signature
                    string. A virtual function may be defined as the interface.
        :param override: override an existing one
        :param tags: tag the function's category, for users to filter and select.
        :param available: define this command's state through an available function. When
                it returns False, the Command dynamically becomes unavailable. This can
                combine with state-machine logic to define a Channel's available
                functions dynamically.
        :param blocking: whether this function blocks the channel. None follows the
                channel default. blocking = True blocks subsequent Commands until it
                finishes, typical for robots that need temporal planning.
                blocking = False runs concurrently; use it for tools with no ordering
                constraint.
        :param call_soon: whether the function executes the moment it enters the track
                (without waiting for scheduling) or waits its turn in the queue. If
                (blocking and call_soon) == True, it clears the queue on enqueue.

        :param priority: command priority. < 0: cancelled as soon as a new command joins.
                > 0: every earlier command with lower priority is cancelled immediately.

        :param return_command: if true, returns a Command object instead of the original
                function. Typically used in tests.
        :param always_observe: if True, no explicit Observe return is needed; the command's
                return value always marks the next round as needing observation.
        :param timeout: if not None, set default timeout for the command.
        :param visible: whether the command is visible to the model. Invisible is typical
                when the command is part of the model's cognitive protocol and can be
                omitted.
        :param macro: the command's return value is used to generate a Command syntax
                string, e.g. a CTML string.

        Best practice for a CommandFunction:
        >>> # Prefer async, so it can block the Channel for the real time it takes.
        >>> # Give parameters and return values explicit types — the types are prompt too.
        >>> # Use serializable objects as inputs and outputs.
        >>> # Define a sync function only for logic that depends on thread safety.
        >>> async def func(arg: type) -> Any:
        >>>     '''A clear docstring'''
        >>>     try
        >>>         # Run logic. No thread blocking, or it blocks everything else.
        >>>         # CommandUtil.create_task
        >>>         ...
        >>>         # return None   # merely done, nothing to observe
        >>>         # return Any    # feedback into context, no Re-Act (or set always_observe to trigger thought)
        >>>         # return CommandUtil.observe('xxx')  # a result the model must observe and think about; triggers Re-Act
        >>>         # raise CommandUtil.raise_observe(...)  # interrupt all action, think now
        >>>     except asyncio.CancelledError:
        >>>         # The scheduler may cancel a command normally; the model can cancel a
        >>>         # running Command at any time.
        >>>         ...
        >>>     except Exception as e:
        >>>         # Handle exceptions correctly.
        >>>         ...
        >>>     finally:
        >>>         # Run-end logic.
        >>>         ...
        """
        # An async function supports the cancel lifecycle, so a command is a unit with full
        # done / cancel / exception semantics. A sync function must manage mid-interruption
        # and state cleanup through CommandUtil.is_task_done, so that blocking behavior does
        # not conflict with the sync function not having finished.
        # `priority` is an advanced feature: do not change it if you do not understand it.
        pass

    @abstractmethod
    def idle(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        decorator
        Register a lifecycle function, executed when the Channel runs its policy.

        Best practice for a lifecycle function:

        >>> # Prefer async, so it can block the Channel for the real time it takes.
        >>> async def func() -> None:
        >>>     # You can get the real runtime executing this command.
        >>>     try
        >>>         # Get dependencies from the global IoC container to obtain runtime
        >>>         # dependency injection.
        >>>         contract = CommandUtil.force_get_contract(...)
        >>>         ...
        >>>     except asyncio.CancelledError:
        >>>         # A lifecycle function can be cancelled by the Channel Runtime at any time.
        >>>         ...
        >>>     except Exception as e:
        >>>         # Handle exceptions correctly.
        >>>         ...
        >>>     finally:
        >>>         # Run-end logic.
        >>>         ...
        """
        pass

    @abstractmethod
    def startup(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        Lifecycle function executed on startup.
        """
        pass

    @abstractmethod
    def close(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        Lifecycle function executed on close.
        """
        pass

    @abstractmethod
    def running(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        Logic that runs for the whole time the Channel Runtime is_running. Called only
        once. Note it runs in parallel with idle / executing.
        """
        pass

    @abstractmethod
    def refresh_meta(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        decorator
        Register an async callback, called before metas are regenerated on each refresh
        cycle.

        Use it for I/O needed at refresh time — alive_cells queries, proxy cache updates,
        external state sync. Combine with get_virtual_children(): update the cache here
        (async), and have get_virtual_children() return the cache (sync, fast).

        >>> async def func() -> None:
        >>>     alive = await matrix.alive_cells()
        >>>     ...  # update the proxy cache for the next get_virtual_children() call
        """
        pass

    @abstractmethod
    def virtual_children(self, func: Callable[[], dict[str, Channel]]) -> Callable[[], dict[str, Channel]]:
        """
        decorator
        Register a callback whose result is merged in when virtual children are fetched at
        runtime.
        """
        ...

    @abstractmethod
    def with_binding(self, contract: type[INSTANCE], instance: INSTANCE) -> Self:
        """
        Register a dependency, injected into the IoC container when the Channel is
        instantiated. Retrieve it via CommandUtil.get_contract.
        """
        # Dependency injection is entirely optional; module instantiation or global
        # factories are a valid substitute.
        pass

    @abstractmethod
    def with_contract_factory(
            self,
            contract: type[INSTANCE],
            factory: Callable[[...], INSTANCE],
            *,
            singleton: bool = True,
            override: bool = False,
    ) -> Self:
        """
        Register a factory for a dependency. When the Channel instantiates its Runtime,
        the factory is registered on the IoC container for that contract.
        """
        pass

    @abstractmethod
    def import_channels(
            self,
            *children: Channel | ChannelFactory | tuple[Channel | ChannelFactory, _ChannelName],
    ) -> Self:
        """
        add sustain channels to the channel.
        """
        pass


class MutableChannel(Channel, Facade):
    """
    A convention for a Channel that supports dynamic construction.
    """

    def import_channels(
            self,
            *children: Channel | ChannelFactory | tuple[Channel | ChannelFactory, _ChannelName],
    ) -> Self:
        """
        Add a child Channel to the current Channel, forming a tree. Similar in effect to
        python's `import module as name`.
        """
        self.build.import_channels(*children)
        return self

    @property
    @abstractmethod
    def build(self) -> Builder:
        """
        Supports dynamically building a Channel through a Builder.
        """
        pass

    @abstractmethod
    def children(self) -> dict[_ChannelName, Channel]:
        """
        return all the static imported channel
        """
        pass

    @abstractmethod
    def virtual_children(self) -> dict[_ChannelName, Channel]:
        """
        return the virtual children channels
        """
        pass


def new_channel(name: str, description: str = "", uid: str | None = None) -> MutableChannel:
    """
    Create a new Mutable/Stateful Channel object with builder.
    Able to define all kinds of channels.
    Use this tool to build your own channel object.
    """
    from ghoshell_moss.core.py_channel import PyChannel
    # if uid is None, auto generate uuid for it.
    return PyChannel(name=name, description=description, uid=uid)


class ChannelCreator(ABC):
    """
    Example of the object-oriented style, if you do not use new_channel and the
    composition-oriented style.
    """

    @classmethod
    @abstractmethod
    def factory(cls, container: IoCContainer) -> Channel:
        """The function creating the channel can itself be registered on a channel."""
        pass


class ChannelInterface(ChannelCreator, ABC):
    """
    A more aggressive object-oriented Channel style: it defines the factory and also
    declares the functions up front.
    """

    @classmethod
    @abstractmethod
    def new(cls, container: IoCContainer) -> Self:
        """Ensure it can instantiate itself from the IoC container."""
        pass

    @abstractmethod
    def as_channel(self) -> Channel:
        """Once instantiated, the object can produce a channel object."""
        pass

    @classmethod
    def factory(cls, container: IoCContainer) -> Channel:
        self_instance = cls.new(container)
        return self_instance.as_channel()


if __name__ == "__build_channel_by_channel_interface_example__":
    # Building an object-oriented abstraction/implementation through the ChannelCreator style.
    main = new_channel(name="__main__")


    class FooInterface(ChannelInterface):

        @abstractmethod
        async def foo(self) -> str:
            pass

        @classmethod
        @abstractmethod
        def new(cls, container: IoCContainer) -> Self:
            pass

        def as_channel(self) -> Channel:
            channel = new_channel(name="foo")
            channel.build.command(
                interface=FooInterface.foo,
            )(self.foo)
            return channel


    class FooImpl(FooInterface):

        def __init__(self, c: IoCContainer):
            self.c = c

        async def foo(self) -> str:
            return self.c.name

        @classmethod
        def new(cls, container: IoCContainer) -> Self:
            return cls(container)


    # Inject the factory method directly into the channel.
    main.import_channels(FooImpl.factory)


async def test_channel(
        *channels: Channel,
        ctml: str,
        timeout: float | None = None,
):
    """Convenience wrapper around ctml_shell_test for app channel testing.

    This is a baseline helper — it covers the common case of sending CTML to a
    single channel and checking results.  Before using it you should understand
    CTML syntax::

        moss ctml read

    For more complex scenarios (scopes, until=any/all, observe, cancel, nested
    channels) this wrapper is not enough.  Read the source of ``ctml_shell_test``,
    search for how it is used in existing tests, and consult the CTML syntax
    (``moss ctml read``).

    Usage in app tests::

        from ghoshell_moss.core.blueprint.channel_builder import new_channel, test_channel

        chan = new_channel(name="my_app")

        @chan.build.command()
        async def greet(name: str) -> str:
            return f"Hello, {name}"

        tasks = await test_channel(chan, ctml='<apps.my_app:greet name="world" />')
        assert len(tasks) == 1
        assert await tasks[0] == "Hello, world"
    """
    from ghoshell_moss.core.ctml import ctml_shell_test
    return await ctml_shell_test(*channels, ctml=ctml, timeout=timeout)
