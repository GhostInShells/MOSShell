import inspect
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, Protocol

from ghoshell_moss.core.concepts.channel import ChannelFullPath, ChannelMeta
from ghoshell_moss.core.concepts.interpreter import Interpretation
from ghoshell_moss.core.concepts.shell import InterpreterKind, MOSSShell
from ghoshell_moss.message import Message

__all__ = [
    "ChatTextStreamer",
    "ShellTurnRequest",
    "ShellTurnResult",
    "run_shell_turn",
    "shell_turn",
    "with_shell",
]


class ChatTextStreamer(Protocol):
    """Provider-agnostic streaming chat completion function.

    The implementation should:
    - accept `messages` as ghoshell `Message` objects
    - return an async iterable that yields plain text deltas
    """

    def __call__(
        self,
        messages: list[Message],
        /,
        **kwargs: Any,
    ) -> AsyncIterable[str]: ...


@dataclass(frozen=True, slots=True)
class ShellTurnRequest:
    """Inputs for running a single shell turn.

    Mainly used by the decorator-style API (`shell_turn`).
    """

    inputs: list[Message]
    history: list[Message] | None = None
    llm_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ShellTurnResult:
    """Result of a single turn that couples LLM output with shell execution."""

    model_messages: list[Message]
    assistant_text: str
    executed_tokens: str
    interpretation: Interpretation


async def run_shell_turn(
    shell: MOSSShell,
    *,
    llm: ChatTextStreamer,
    inputs: list[Message],
    history: list[Message] | None = None,
    kind: InterpreterKind = "clear",
    stream_id: str | None = None,
    config: dict[ChannelFullPath, ChannelMeta] | None = None,
    ignore_wrong_command: bool = False,
    prepare_timeout: float = 2.0,
    llm_kwargs: dict[str, Any] | None = None,
) -> ShellTurnResult:
    """Run a single (turn-based) LLM response and execute produced commands.

    This is a provider-agnostic driver that relies on `llm` to stream plain text
    deltas. It manages interpreter lifecycle and returns execution artifacts.
    """

    if history is None:
        history = []
    if llm_kwargs is None:
        llm_kwargs = {}

    interpreter = await shell.interpreter(
        kind=kind,
        stream_id=stream_id,
        config=config,
        prepare_timeout=prepare_timeout,
        ignore_wrong_command=ignore_wrong_command,
    )

    async with interpreter:
        # Use interpreter's canonical message merge rules.
        model_messages = interpreter.merge_messages(history, inputs)

        assistant_chunks: list[str] = []
        text_stream = llm(model_messages, **llm_kwargs)
        try:
            async for delta in text_stream:
                if delta:
                    assistant_chunks.append(delta)
                    interpreter.feed(delta)
        finally:
            interpreter.commit()

        interpretation = await interpreter.wait_stopped()
        assistant_text = "".join(assistant_chunks)

        return ShellTurnResult(
            model_messages=model_messages,
            assistant_text=assistant_text,
            executed_tokens=interpreter.executed_tokens(),
            interpretation=interpretation,
        )


def shell_turn(
    shell: MOSSShell,
    *,
    llm: ChatTextStreamer,
    kind: InterpreterKind = "clear",
    stream_id: str | None = None,
    config: dict[ChannelFullPath, ChannelMeta] | None = None,
    ignore_wrong_command: bool = False,
    prepare_timeout: float = 2.0,
):
    """Decorator factory that binds `shell + llm` into a turn-based runner.

    The decorated function can return either:
    - `ShellTurnRequest`
    - `list[Message]` (treated as `inputs`, with empty history)
    """

    def _decorator(func):
        async def _wrapped(*args: Any, **kwargs: Any) -> ShellTurnResult:
            value = func(*args, **kwargs)
            if inspect.isawaitable(value):
                value = await value

            if isinstance(value, ShellTurnRequest):
                inputs = value.inputs
                history = value.history
                llm_kwargs = value.llm_kwargs
            elif isinstance(value, list) and all(isinstance(m, Message) for m in value):
                inputs = value
                history = None
                llm_kwargs = None
            else:
                raise TypeError("decorated function must return ShellTurnRequest or list[Message]")

            return await run_shell_turn(
                shell,
                llm=llm,
                inputs=inputs,
                history=history,
                kind=kind,
                stream_id=stream_id,
                config=config,
                ignore_wrong_command=ignore_wrong_command,
                prepare_timeout=prepare_timeout,
                llm_kwargs=llm_kwargs,
            )

        return _wrapped

    return _decorator


def with_shell(
    shell: MOSSShell,
    *,
    llm: ChatTextStreamer,
    kind: InterpreterKind = "clear",
    stream_id: str | None = None,
    config: dict[ChannelFullPath, ChannelMeta] | None = None,
    ignore_wrong_command: bool = False,
    prepare_timeout: float = 2.0,
    default_llm_kwargs: dict[str, Any] | None = None,
):
    """Bind a shell + llm into a directly callable turn runner.

    This is a syntax sugar over `run_shell_turn`.

    Example:

        runner = with_shell(shell, llm=litellm_text_streamer())
        result = await runner([Message.new(role="user").with_content("hi")])
    """

    if default_llm_kwargs is None:
        default_llm_kwargs = {}

    async def _run(
        inputs: list[Message] | ShellTurnRequest,
        *,
        history: list[Message] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> ShellTurnResult:
        if isinstance(inputs, ShellTurnRequest):
            request = inputs
            _inputs = request.inputs
            _history = request.history
            _llm_kwargs = request.llm_kwargs
        else:
            _inputs = inputs
            _history = history
            _llm_kwargs = llm_kwargs

        merged_llm_kwargs = dict(default_llm_kwargs)
        if _llm_kwargs:
            merged_llm_kwargs.update(_llm_kwargs)

        return await run_shell_turn(
            shell,
            llm=llm,
            inputs=_inputs,
            history=_history,
            kind=kind,
            stream_id=stream_id,
            config=config,
            ignore_wrong_command=ignore_wrong_command,
            prepare_timeout=prepare_timeout,
            llm_kwargs=merged_llm_kwargs,
        )

    return _run
