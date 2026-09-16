"""
Streaming interpreter: interprets tokens from a model's output into the runtime
topology of Commands and dispatches them immediately.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Iterable, AsyncIterable, AsyncIterator
from typing_extensions import Self
from ghoshell_moss.core.concepts.errors import CommandErrorCode, InterpretError
from ghoshell_moss.core.concepts.command import CommandTask, CommandToken
from ghoshell_moss.core.concepts.channel import ChannelFullPath, ChannelMeta, Channel
from ghoshell_moss.core.concepts.tools import CommandAsTool
from ghoshell_moss.message import Message
from ghoshell_common.contracts import LoggerItf
from pydantic import BaseModel, Field, AwareDatetime
from datetime import datetime
from dateutil import tz
import queue

__all__ = [
    "CommandTaskCallback",
    "CommandTokenParser",
    "CommandTokenCallback",
    "TextTokenParser",
    "Interpreter",
    "Interpretation",
]

CommandTokenCallback = Callable[[CommandToken | None], None]
CommandTaskCallback = Callable[[CommandTask | None], None]

# 宏展开的递归深度上限. 对齐 loop 原语 100 次上限: 用宏实现的 loop, N 次迭代即深度 N.
MAX_MACRO_DEPTH = 100


class TextTokenParser(ABC):
    """
    parse from string stream into command tokens
    """

    @abstractmethod
    def with_callback(self, *callbacks: CommandTokenCallback) -> None:
        """
        Register callbacks that receive generated command tokens.
        Each produced command token is sent to these callbacks.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Whether this parser is done parsing."""
        pass

    @abstractmethod
    def feed(self, delta: str) -> None:
        """feed this parser with the stream delta"""
        pass

    @abstractmethod
    def commit(self) -> None:
        """notify the parser that the stream is done"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop parsing immediately. Does not raise.
        """
        pass

    @abstractmethod
    def buffered(self) -> str:
        """
        Return the input text after coalesced buffering.
        """
        pass

    @abstractmethod
    def parsed(self) -> Iterable[CommandToken]:
        """Return the command tokens generated so far."""
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the parser's manual usage context.
        """
        pass


class CommandTokenParser(ABC):
    """
    CommandTaskElement works like AST but in realtime.
    It accepts command token from a stream, and generate command task concurrently.

    The keypoint is, the command tokens are organized in the recursive pattern,
    that one command can embrace many children command within it, and handle them by its own means,
    just like a function call other functions inside it.

    So we need an Element Tree to parse the tokens into command tasks, and send the tasks immediately
    """

    @abstractmethod
    def on_token(self, token: CommandToken | None) -> list[CommandTask] | None:
        """
        Accept one command token.
        :param token: if None, the command token stream has ended.
        """
        pass

    @abstractmethod
    def is_end(self) -> bool:
        """Whether parsing is finished."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    @abstractmethod
    def destroy(self) -> None:
        """Manually release internal data structures to speed up garbage collection and avoid memory leaks."""
        pass


_TaskId = str
_TaskCaller = str


class Interpretation(BaseModel):
    """
    The result of a single Interpreter run.
    """

    done: bool = Field(default=False, description="Whether the run has finished.")
    id: str = Field(description="interpretation id")
    observe: bool = Field(
        default=False,
        description="Whether this run result needs to be observed by the AI.",
    )
    feed_inputs: list[str] = Field(default_factory=list, description="Text fed into the interpreter via feed.")
    command_tokens: list[CommandToken] = Field(
        default_factory=list,
        description="Command tokens generated during interpretation.",
    )
    executed_inputs: list[str] = Field(default_factory=list, description="Input text that has been executed.")

    compiled_tasks: dict[_TaskId, _TaskCaller] = Field(default_factory=dict,
                                                       description="Compiled tasks: cid => task caller.")
    task_done_at: dict[_TaskId, float] = Field(
        default_factory=dict,
    )
    pending_tasks: dict[_TaskId, _TaskCaller] = Field(default_factory=dict,
                                                      description="Not-yet-finished tasks: cid => task caller.")
    cancelled_tasks: dict[_TaskId, _TaskCaller] = Field(
        default_factory=dict,
        description="Cancelled tasks: cid => task caller.",
    )
    failed_tasks: dict[_TaskId, _TaskCaller] = Field(
        default_factory=dict,
        description="Failed tasks: cid => task caller.",
    )
    success_tasks: dict[_TaskId, _TaskCaller] = Field(
        default_factory=dict, description="Succeeded tasks: cid => task caller."
    )
    output: list[Message] = Field(default_factory=list, description="Messages to output from the run result.")
    messages: list[Message] = Field(default_factory=list, description="Messages to observe from the run result.")
    interrupted: bool = Field(default=False, description="Whether the run was forcibly interrupted.")
    exception: str = Field(
        default="",
        description="Exception raised during the run, if any.",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="The Interpretation creation time."
    )
    started_at: float = Field(
        default_factory=time.time,
    )

    def executed_logos(self) -> str:
        return "".join(self.executed_inputs)

    def on_task_compiled(self, task: CommandTask | None) -> None:
        """Record a task's compiled state."""
        if task is None or task.meta.name.startswith("_"):
            return
        self.compiled_tasks[task.cid] = task.caller_name()
        self.pending_tasks[task.cid] = task.caller_name()

    def on_done_task(self, task: CommandTask) -> None:
        """Handle a done task: record its state and merge its result."""
        if not task.done():
            return
        task_id = task.cid
        self.task_done_at[task_id] = time.time()
        if self.done:
            return
        if task_id in self.pending_tasks:
            self.pending_tasks.pop(task_id)
        # 注册执行成功的 tokens.
        if task.success():
            # 宏任务自身的 tokens 不进 executed_inputs —— 它被展开成 body, 摊平轨迹只含展开产物,
            # 否则含 <macro/> 的轨迹存成新宏 body 会自指.
            if not task.meta.macro:
                self.executed_inputs.append(task.tokens)
            self.success_tasks[task_id] = task.caller_name()
        # 记录 cancel 类别的.
        elif CommandErrorCode.is_cancelled(task.errcode):
            self.cancelled_tasks[task_id] = task.caller_name()
        # 记录异常的.
        else:
            self.failed_tasks[task_id] = task.caller_name()

        # 合并 task 运行结果.
        result = task.task_result()
        # 根据协议判定要 observe.
        if result.observe or CommandErrorCode.is_critical(task.errcode):
            self.observe = True
        if len(result.output) > 0:
            self.output.extend(result.output)
        result_messages = result.as_messages()
        if len(result_messages) > 0:
            self.messages.extend(result_messages)

    def output_messages(self) -> list[Message]:
        """
        Messages to be output to the client.
        """
        return self.output.copy()

    def state(self) -> str:
        state = 'running'
        if self.done:
            if self.exception:
                state = 'error'
            elif self.interrupted:
                state = 'interrupted'
            else:
                state = 'done'
        return state

    def status_messages(self) -> list[Message]:
        """A description of the current run state."""
        state = self.state()
        status_message = Message.new(
            tag='shell',
            timestamp=False,
            attributes={'state': state, 'run_at': self.created},
        )
        lines = []
        if len(self.compiled_tasks) > 0:
            lines.append("tasks: %d" % len(self.compiled_tasks))
        if len(self.success_tasks) > 0:
            lines.append("completed: %d" % len(self.success_tasks))
        if len(self.cancelled_tasks) > 0:
            lines.append("cancelled: %d" % len(self.cancelled_tasks))
        if len(self.failed_tasks) > 0:
            lines.append("failed: %s" % self._status_task_expression(list(self.failed_tasks.values())))
        if self.exception:
            lines.append("Stop at Exception: %s." % self.exception)
        if len(self.pending_tasks) > 0:
            lines.append("ongoing: %s" % ",".join(self.pending_tasks.values()))
        status_message.with_content("\n".join(lines))
        return [status_message]

    def _status_task_expression(self, tasks: list[str]) -> str:
        count = len(tasks)
        if count == 0:
            return '0'
        last = tasks[-1]
        return "%d, last is %s" % (count, last)

    def executed_messages(self) -> list[Message]:
        """The messages describing the run result."""
        messages = self.messages.copy()
        return messages

    def as_messages(self) -> list[Message]:
        messages = self.status_messages()
        messages.extend(self.executed_messages())
        return messages


class Interpreter(ABC):
    """
    Command interpreter: parses command tokens from a text stream.
    It also parses the streaming command tokens into streaming command tasks and calls back into the executor.

    It can be regarded as a key-frame of the Shell runtime state.
    The Shell creates only one stateful Interpreter at a time; if the previous one has not finished, it will be interrupted.

    There are two interruption modes, clear / append:
    clear wipes all state of the previous Interpreter.
    append only interrupts the previous Interpreter's run.

    The previous interpreter is only temporarily interrupted; its run result is passed to the next interpreter.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """each time stream interpretation has a unique id"""
        pass

    @property
    @abstractmethod
    def kind(self) -> str:
        pass

    @property
    @abstractmethod
    def logger(self) -> LoggerItf:
        pass

    @abstractmethod
    def previews(self) -> Interpretation | None:
        """
        The interpretation result of the previous run that was interrupted.
        """
        pass

    @abstractmethod
    def interpretation(self) -> Interpretation:
        """
        Return the current interpretation.
        It may still be running and continuously adding new information.
        """
        pass

    @abstractmethod
    def channels(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        Return all channels of the current interpreter.
        """
        pass

    @abstractmethod
    def meta_instruction(self) -> str:
        """
        Meta rules for a model to use MOSS.
        Each interpreter may define its own rules.
        For example, CTMLInterpreter defines the CTML rules.
        """
        pass

    @abstractmethod
    def static_messages(self) -> str:
        """
        The complete prompt for the channels under the current interpreter state, presented to the model.
        """
        pass

    def instruction(self, prompts: list[str] | None = None) -> str:
        """
        The default system prompt of the MOSS architecture.
        """
        instructions = [self.meta_instruction()]
        channel_instructions = self.static_messages()
        instructions.append(channel_instructions)
        if prompts:
            instructions.extend(prompts)
        return '\n'.join(instructions)

    @abstractmethod
    def dynamic_messages(self) -> list[Message]:
        """
        Return the dynamic context the interpreter obtains as a snapshot.
        """
        pass

    def merge_messages(self, history: list[Message | dict], inputs: list[Message | dict]) -> list[Message]:
        """
        Merge messages following the system rules to build a model context.
        This also illustrates how to use the interpreter to define context.

        In the model context conversation history, the simplest context topology is:

        - instructions: prompts and instructions. Change as little as possible and keep them merged.
        - conversations: the conversation history.
        - last turn: the input and output messages of the previous turn.
        - context: the current state, the mutable part. The model should understand that this part changes at any time.
        + new turn:
            - inputs: this turn's input for a turn-based model.
            - recall: recall generated automatically from the context
            - reasoning: the reasoning process
            - actions: the action process.
            - outputs: output
            - observation: messages that need to be observed.
        """
        instructions = self.instruction()
        messages = [Message.new(tag="").with_content(instructions)]
        messages.extend(history)
        messages.extend(self.dynamic_messages())
        messages.extend(inputs)
        return messages

    @abstractmethod
    def feed(self, delta: str, throw: bool = True) -> bool:
        """
        Submit a text fragment to the interpreter. The interpreter parses these input streams asynchronously and runs its dispatch logic.
        >>> async def run_interpreter(interpreter: Interpreter, items: AsyncIterable[str]):
        >>>     async with interpreter:
        >>>         async for item in items:
        >>>             interpreter.feed(item)
        >>>         interpreter.commit()

        :param delta: the text fragment to transfer.
        :param throw: when True, a parsing error raises. Can be used to trigger interruption.
        :raise InterpretError:
        :return: True if the state is normal and the submission succeeded, otherwise False.
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        Mark all input as finished. Later feeds have no effect.
        Note that the interpreter's parsing and execution may not be complete yet.
        """
        pass

    async def interpret(self, deltas: AsyncIterable[str]) -> None:
        """
        Syntactic sugar: a complete interpretation pass that includes feed and commit.
        """
        async for delta in deltas:
            if not self.feed(delta):
                break
        self.commit()

    @abstractmethod
    def on_task_compiled(self, *callbacks: CommandTaskCallback) -> None:
        """
        Register callbacks invoked when a task is compiled.
        """
        pass

    @abstractmethod
    def on_task_done(self, *callbacks: CommandTaskCallback) -> None:
        """
        Register callbacks invoked when a task finishes running.
        """
        pass

    @abstractmethod
    def text_token_parser(self) -> TextTokenParser:
        """
        The token parser held by the interpreter. It parses text input into command tokens, and command tokens into command tasks.
        Command tasks automatically call back into the interpreter for execution.

        >>> async def example(interpreter: Interpreter, deltas: AsyncIterable[str]) -> None:
        >>>     with interpreter.text_token_parser() as parser:
        >>>         async for delta in deltas:
        >>>             parser.feed(delta)

        Note the parser is synchronous and blocking, so the correct approach is to use the interpreter's own feed function for non-blocking behavior.
        The parser usually runs in a separate thread pool.
        """
        pass

    @abstractmethod
    def command_token_parser(self) -> CommandTokenParser:
        """
        The Element object the current interpreter uses for tree-shaped command token parsing. For debugging.
        Usually runs in a separate thread pool.
        """
        pass

    @abstractmethod
    def parsed_tokens(self) -> Iterable[CommandToken]:
        """
        Command tokens parsed and generated so far.
        """
        pass

    @abstractmethod
    def received_text(self) -> str:
        """
        Return the text that has been fully fed in. Input must go through feed.
        """
        pass

    @abstractmethod
    def compiled_tasks(self) -> dict[str, CommandTask]:
        """
        Tasks compiled so far.
        """
        pass

    @abstractmethod
    def managing_tasks(self) -> dict[str, CommandTask]:
        """
        Tasks under management, possibly including ones from the previous run.
        """
        pass

    def done_tasks(self) -> list[CommandTask]:
        """
        Return tasks that have finished executing, including cancelled or failed ones.
        """
        tasks = self.managing_tasks().copy()
        executed = []
        for task in tasks.values():
            if not task.done():
                continue
            executed.append(task)
        return executed

    def incomplete_tasks(self) -> list[CommandTask]:
        """
        Return tasks that compiled successfully but have not finished executing.
        """
        tasks = self.managing_tasks().copy()
        pending = []
        for task in tasks.values():
            if not task.done():
                pending.append(task)
        return pending

    def executed_tokens(self) -> str:
        """
        Return the tokens that have finished executing.
        """
        tokens = []
        for task in self.done_tasks():
            tokens.append(task.tokens)
        return "".join(tokens)

    @abstractmethod
    async def close(
            self,
            cancel_executing: bool = True,
    ) -> Interpretation | None:
        """
        Stop the interpretation.
        :param cancel_executing: whether to also clear the parsed tasks. If not cleared, the tasks themselves are not interrupted.
        :return: if an unfinished Interpreter was interrupted, return its executed interpretation state; if it already finished, return None.
        """
        pass

    @abstractmethod
    def is_stopped(self) -> bool:
        """
        Whether the interpretation process has stopped.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        pass

    @abstractmethod
    def progresses(self) -> dict[_TaskId, str]:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Whether it is running: between start and stop.
        """
        pass

    @abstractmethod
    def is_interrupted(self) -> bool:
        """
        Whether the interpretation process was interrupted.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        Enter the interpreter as an async context manager.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def exception(self) -> Optional[Exception]:
        """
        Return the exception raised during the run, if any.
        """
        pass

    def raise_exception(self):
        if exp := self.exception():
            raise exp

    @abstractmethod
    async def wait_compiled(self, timeout: float | None = None, throw: bool = True) -> None:
        """
        Wait for the interpretation to complete. Completion has two cases:
        1. Input is complete.
        2. It was interrupted.
        """
        pass

    @abstractmethod
    async def wait_stopped(self) -> Interpretation:
        """
        Block until the run finishes or the system is interrupted, then return the interpretation.
        It does not mean all generated tasks have finished executing.
        """
        pass

    @abstractmethod
    async def wait_tasks(
            self,
            timeout: float | None = None,
            *,
            return_when: str = asyncio.ALL_COMPLETED,
            throw: bool = True,
            throw_task_error: bool = False,
            clear_undone: bool = True,
    ) -> dict[str, CommandTask]:
        """
        Block until all generated tasks complete, returning by the return_when rule. Usually used for debugging.
        :param timeout: timeout for the wait.
        :param throw: whether to raise an exception, or just return the tasks as they were when interrupted.
        :param throw_task_error: whether to re-raise when a task hits an exception.
        :param return_when: when to exit the wait-for-execution-done.
        :param clear_undone: whether to mark unfinished Tasks as Cleared when this function exits.
        """
        pass

    @abstractmethod
    async def parse_macro_logos(
            self,
            logos: str,
            *,
            root_channel: ChannelFullPath = '',
            macro_id: str = '',
            caller: str = '',
            lineage: str = '',
    ) -> list[CommandToken]:
        """Parse the given string within the interpreter's lifecycle to generate CommandTokens.
        lineage: the stream_id lineage chain of the expanded tokens, used for cid uniqueness (tasks rely on cid for de-duplication in several places). If empty, the implementation generates it."""
        ...

    # --- tools 兼容.  --- #

    @abstractmethod
    def tools(self) -> Iterable[CommandAsTool]:
        """
        openai & anthropic & pydantic ai compatible tool
        """
        pass

    # --- interpreter 的无状态解析函数 --- #

    async def aparse_text_to_command_tokens(
            self,
            texts: AsyncIterable[str],
            *,
            stopped: Callable[[], bool] | None = None,
    ) -> AsyncIterable[CommandToken]:
        """
        Wrap a synchronous function as an async one while still raising exceptions correctly.
        """
        text_queue = queue.Queue()
        token_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def callback(token: CommandToken | None) -> None:
            loop.call_soon_threadsafe(token_queue.put_nowait, token)

        def real_stop():
            """
            Determine whether a forced stop is in effect.
            """
            nonlocal stop_event
            if stop_event.is_set():
                return True
            if stopped and stopped():
                return True
            return False

        async def consume():
            """
            Consume the incoming texts.
            """
            nonlocal texts
            async for text in texts:
                text_queue.put(text)
            text_queue.put(None)

        cor = asyncio.to_thread(self.parse_text_to_command_tokens, text_queue, callback, stopped=real_stop)
        parsing_task = asyncio.create_task(cor)

        async def read_from():
            """
            Read messages.
            """
            while not real_stop():
                item = await token_queue.get()
                if item is None:
                    break
                yield item
            await parsing_task

        consume_task = asyncio.create_task(consume())
        try:
            async for got in read_from():
                yield got
        except asyncio.CancelledError:
            raise
        except Exception as e:
            text_queue.put(None)
            stop_event.set()
            self.logger.exception(
                "[Interpreter][%s] failed parsing text into command tokens: %r", self.__class__.__name__, e
            )
            raise e
        finally:
            # 冗余的回收.
            if not parsing_task.done():
                parsing_task.cancel()
            if not consume_task.done():
                consume_task.cancel()

    async def parse_tokens_to_command_tasks(
            self,
            tokens_queue: asyncio.Queue[CommandToken | None],
            task_callback: Callable[[CommandTask | None], None],
            *,
            stopped: Callable[[], bool] | None = None,
            run_macro: bool = True,
    ):
        """
        Can run in a coroutine. Parse the input token stream and produce Command Tasks. Uses a poison pill as the end marker.
        When a task with `meta.macro` is hit, it stops pulling -> awaits the macro task -> re-roots and parses the return value's (MacroResult | str) logos
        into a token stream, feeding it back to the same parser for in-place expansion. Recursive expansion is bounded by MAX_MACRO_DEPTH.
        raise InterpretError
        """
        parser = self.command_token_parser()
        # parser.with_callback(task_callback)
        if stopped is None:
            def empty_stopped():
                return False

            stopped = empty_stopped
        # 单次解释自增: 展开批次 id + lineage (stream_id) 来源, 保证展开 token 的 cid 独立.
        macro_counter = 0

        async def expand_macro(task: CommandTask, depth: int) -> None:
            nonlocal macro_counter
            if stopped():
                return
            # await 宏任务. 0.2s wait_for 轮询, stopped() 可打断.
            while not stopped():
                try:
                    await asyncio.wait_for(task.wait(throw=False), 0.2)
                    break
                except asyncio.TimeoutError:
                    continue
            if stopped():
                return
            if not task.success():
                if task.cancelled() or CommandErrorCode.is_cancelled(task.errcode):
                    # 取消: 非 interpreter error, 跳过展开.
                    return
                err = task.exception() or RuntimeError(f"macro command failed: {task.caller_name()}")
                # 宏任务执行失败 (非取消) → interpreter error.
                raise InterpretError(f"macro `{task.caller_name()}` failed: {err}")
            tr = task.task_result()
            if tr is None:
                return
            # 归一化为 logos 串: MacroResult.logos 优先, 否则裸 str.
            if tr.logos is not None:
                logos = tr.logos
            elif isinstance(tr.result, str):
                logos = tr.result
            else:
                return  # 非 logos 承载, 不展开.
            if not logos:
                return  # 空串 void 宏, 不展开.
            if depth >= MAX_MACRO_DEPTH:
                # 自引用宏靠深度上限兜底.
                raise InterpretError(f"macro recursion depth exceeded: {depth} >= {MAX_MACRO_DEPTH}")
            macro_counter += 1
            macro_id = macro_counter
            # 换根解析 (阻塞, 内部卸载到线程). 解析失败已在实现侧归属 interpreter error.
            tokens = await self.parse_macro_logos(
                logos,
                root_channel=task.chan,
                macro_id=str(macro_id),
                caller=task.caller_name(),
                lineage=f"m{macro_id}",
            )
            # 注入: 展开 token 在流序上先于后续模型 token (循环此刻持有 queue, 天然不拉取).
            for token in tokens:
                tasks = parser.on_token(token)
                if tasks is not None:
                    for t in tasks:
                        t.macro_id = str(macro_id)
                        t.from_macro_id = task.macro_id
                        t.on_compiled()
                        task_callback(t)
                        if t.meta.macro:
                            await expand_macro(t, depth + 1)
                await asyncio.sleep(0.0)

        try:
            with parser:
                while not stopped() and not parser.is_end():
                    try:
                        item = await asyncio.wait_for(tokens_queue.get(), 0.2)
                    except asyncio.TimeoutError:
                        continue
                    if item is None:
                        break
                    tasks = parser.on_token(item)
                    if tasks is not None:
                        for task in tasks:
                            task.on_compiled()
                            task_callback(task)
                            if run_macro and task.meta.macro:
                                await expand_macro(task, 0)
                    await asyncio.sleep(0.0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception(
                "[Interpreter][%s] failed parsing tokens into command tasks: %r", self.__class__.__name__, e
            )
            raise e
        finally:
            task_callback(None)
            parser.destroy()

    async def run(self, logos: str) -> dict[str, CommandTask]:
        """
        Syntactic sugar, usually for debugging or unit tests.
        """
        async with self as itp:
            itp.feed(logos)
            itp.commit()
            await itp.wait_stopped()
            itp.raise_exception()
            return itp.managing_tasks()

    def parse_text_to_command_tokens(
            self,
            text_queue: queue.Queue[str | None],
            command_token_callback: Callable[[CommandToken | None], None],
            *,
            stopped: Callable[[], bool] | None = None,
    ):
        """
        Usually runs in a separate thread. Parse the input Text stream and produce a Command Token stream. Uses a poison pill as the end marker.
        raise InterpretError
        """
        text_token_parser = self.text_token_parser()
        text_token_parser.with_callback(command_token_callback)
        if stopped is None:
            def empty_stopped():
                return False

            stopped = empty_stopped
        with text_token_parser:
            while not text_token_parser.is_done():
                if stopped():
                    text_token_parser.stop()
                    break
                try:
                    # check every 0.1 second if the loop is stopped.
                    item = text_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    text_token_parser.commit()
                    break
                text_token_parser.feed(item)
