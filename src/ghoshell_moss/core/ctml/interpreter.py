import asyncio
import logging
from typing import Optional, ClassVar, Callable, Coroutine, Iterable
from typing_extensions import Self

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import Timeleft, uuid
from ghoshell_moss.core.concepts.channel import ChannelFullPath, ChannelMeta
from ghoshell_moss.core.concepts.command import Command, CommandTask, CommandToken
from ghoshell_moss.core.concepts.errors import CommandErrorCode, InterpretError
from ghoshell_moss.core.concepts.interpreter import (
    CommandTaskCallback,
    CommandTokenParser,
    TextTokenParser,
    Interpreter,
    Interpretation,
)
from ghoshell_moss.contracts.speech import Speech
from ghoshell_moss.core.concepts.tools import CommandAsTool
from ghoshell_moss.core.ctml.elements import CommandTaskElementContext
from ghoshell_moss.core.ctml.meta import get_moss_ctml_meta_instruction
from ghoshell_moss.core.ctml.token_parser import CTML2CommandTokenParser, AttrWithTypeSuffixParser, ctml_default_parsers
from ghoshell_moss.core.ctml.v1_0.prompts import make_static_messages, make_dynamic_messages, make_interfaces
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.message import Message
import queue

__all__ = [
    "DEFAULT_META_PROMPT",
    "CTMLInterpreter",
]

DEFAULT_META_PROMPT = get_moss_ctml_meta_instruction()

_Title = str
_Description = str
_Interface = str


class CTMLInterpreter(Interpreter):
    instances_count: ClassVar[int] = 0

    def __init__(
            self,
            kind: str,
            *,
            interrupted: Interpretation | None = None,
            undone_tasks: list[CommandTask] | None = None,
            commands: dict[ChannelFullPath, dict[str, Command]],
            speech: Speech,
            stream_id: Optional[str] = None,
            callback: Optional[CommandTaskCallback] = None,
            root_tag: str = "ctml",
            tokens_replacement: Optional[dict[str, str]] = None,
            logger: Optional[LoggerItf] = None,
            on_startup: Optional[Callable[[], Coroutine[None, None, None]]] = None,
            moss_meta_instruction: Optional[str] = None,
            channel_metas: Optional[dict[ChannelFullPath, ChannelMeta]] = None,
            ignore_wrong_command: bool = False,
            clear_after_exit: bool | None = None,
            ctml_attr_parser: Optional[AttrWithTypeSuffixParser] = None,
            moss_static: str | None = None,
            moss_dynamic: list[Message] | None = None,
    ):
        """
        :param commands: 所有 interpreter 可以使用的命令. key 是 channel path, value 是这个 channel 可以用的 commands.
        :param speech: 用来发送所有模型非 command 输出的内容.
        :param stream_id: 让 interpreter 有一个唯一的 id.
        :param callback: command task callback
        :param root_tag: 决定生成 command token 的起始和结尾标记. 通常没有功能性.
        :param tokens_replacement: 如果传入, 在解析时会把 输出的 key token 转换成 value token 然后解析. 用来做快速匹配.
        :param logger: 日志.
        :param on_startup: 可以定义额外的启动函数.
        :param moss_meta_instruction: MOSS 解释器的基础语法规则, 如果为空则使用默认的.
        :param channel_metas: 用来定义当前所拥有的 channels 信息, 用来提供给大模型.
        :param ignore_wrong_command: 是否忽略不存在的 command.
        :param clear_after_exit: clear undone tasks after exit.
        :param moss_static: 静态讯息.
        :param moss_dynamic: 动态生成的讯息.
        """
        # 生成 stream id.
        self._id = stream_id or uuid()
        self._kind: str = kind
        self._previews_interrupted_interpretation: Interpretation | None = interrupted
        self._meta_instruction: str | None = moss_meta_instruction
        self._channel_metas = channel_metas or {}
        if clear_after_exit is None:
            clear_after_exit = False
        self._clear_after_exit = clear_after_exit
        # 准备日志.
        self._logger = logger or logging.getLogger("CTMLInterpreter")
        self._log_prefix = "[CTMLInterpreter %s] " % self.id
        # 可用的 task 回调.
        self._on_task_created_callbacks: list[CommandTaskCallback] = []
        self._on_task_done_callbacks: list[CommandTaskCallback] = []
        self._ctml_attr_parser = ctml_attr_parser or ctml_default_parsers
        if callback is not None:
            self._on_task_created_callbacks.append(callback)
        # 启动时执行的命令.
        self._on_startup = on_startup

        # commands map, key is unique name of the command
        self._channel_command_map = commands
        self._commands_map: dict[str, Command] = {}
        for channel_path, channel_commands in commands.items():
            for command_name, command in channel_commands.items():
                if not command.is_available():
                    # 不加入不可运行的指令.
                    continue
                unique_name = Command.make_unique_name(channel_path, command_name)
                self._commands_map[unique_name] = command

        self._root_tag = root_tag
        self._token_replacement = tokens_replacement or {}
        self._stopped_event = ThreadSafeEvent()
        self._closed = False
        self._parsing_exception: Optional[InterpretError] = None
        self._ignore_wrong_command = ignore_wrong_command

        # output related
        self._speech = speech
        self._outputted: Optional[list[str]] = None
        # 用线程安全队列就可以. 考虑到队列可能不是在同一个 loop 里添加
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_deltas_queue: queue.Queue[str | None] = queue.Queue()
        # 内部传输 tokens 的通道.
        self._text_to_parsed_tokens_queue: asyncio.Queue[CommandToken | None] = asyncio.Queue()

        # create task element
        self._managing_tasks: dict[str, CommandTask] = {}  # 解析生成的 tasks.
        self._compiled_tasks: dict[str, CommandTask] = {}

        # input buffer
        self._interpretation = Interpretation(
            id=self._id,
        )
        self._moss_static: str | None = moss_static
        self._moss_dynamic: list[Message] | None = moss_dynamic
        if undone_tasks is not None and len(undone_tasks) > 0:
            for task in undone_tasks:
                # 分享 task 和 task done.
                self._managing_tasks[task.cid] = task
                task.add_done_callback(self._task_done_callback)

        #  --- runtime --- #
        self._main_parsing_loop_task: Optional[asyncio.Task] = None  # 解析的主循环.
        self._tasks_done_then_stop_task: Optional[asyncio.Task] = None
        self._wait_interpreter_stop_task: Optional[asyncio.Task] = None
        self._started = False
        self._committed = False
        self._interrupted = False
        self._task_sent_done = False
        self._parsing_loop_done = asyncio.Event()  # 标记解析完成.
        self._destroyed = False
        CTMLInterpreter.instances_count += 1

    def _set_interpreter_error(self, error: InterpretError) -> None:
        if self._stopped_event.is_set():
            return
        if self._parsing_exception is not None:
            return
        self._parsing_exception = error
        self._interpretation.observe = True
        self._interpretation.exception = str(error)
        self._stopped_event.set()
        for task in self._managing_tasks.values():
            if not task.done():
                task.cancel("interpret error")

    @property
    def id(self) -> str:
        return self._id

    @property
    def kind(self) -> str:
        return self._kind

    def tools(self) -> Iterable[CommandAsTool]:
        for channel_path, meta in self._channel_metas.items():
            commands = self._commands_map.get(channel_path, None)
            if commands is None:
                continue
            for command_meta in meta.commands:
                unique_name = Command.make_unique_name(channel_path, command_meta.name)
                if unique_name in commands:
                    command = commands[unique_name]
                    yield CommandAsTool(command, channel_path=channel_path, task_callback=self._send_command_task)

    @property
    def logger(self) -> LoggerItf:
        return self._logger

    def previews(self) -> Interpretation | None:
        return self._previews_interrupted_interpretation

    def interpretation(self) -> Interpretation:
        return self._interpretation

    def managing_tasks(self) -> dict[str, CommandTask]:
        return self._managing_tasks

    def _receive_command_token(self, token: CommandToken | None) -> None:
        """将 token 记录到解析后的 tokens 中."""
        if self._stopped_event.is_set():
            return
        if token is not None:
            self._interpretation.command_tokens.append(token)
        self._text_to_parsed_tokens_queue.put_nowait(token)

    def _send_command_task(self, task: CommandTask | None) -> None:
        try:
            if self._task_sent_done or self._stopped_event.is_set():
                if task is not None and not task.done():
                    task.cancel("interpreter stopped")
                return
            # 只发送一次 None 作为毒丸.
            if task is not None:
                # 添加新的 task.
                self._managing_tasks[task.cid] = task
                # 生成的 task
                self._compiled_tasks[task.cid] = task
                self._interpretation.on_task_compiled(task)
                # 注册 task 的回调, 如果出了异常就干脆中断整个流程, 也别解析了.
                task.add_done_callback(self._task_done_callback)

            if len(self._on_task_created_callbacks) > 0:
                for callback in self._on_task_created_callbacks:
                    try:
                        callback(task)
                    except Exception as exc:
                        self._logger.exception(
                            "%s on task creation callback %s exception: %s",
                            self._log_prefix,
                            task,
                            exc,
                        )
            self._task_sent_done = task is None
        except Exception as e:
            err = InterpretError(f"Send command failed: {e}")
            self._set_interpreter_error(err)
            self._logger.exception("%s Send command task %s failed: %s", self._log_prefix, task, e)
        finally:
            self._logger.debug("%s Send command task %s", self._log_prefix, task)

    def _task_done_callback(self, command_task: CommandTask) -> None:
        if not command_task.done():
            self._logger.error(
                "%s Command task is not done but send to interpreter on task %s done",
                self._log_prefix,
                command_task,
            )
            command_task.cancel("system error")
        self._interpretation.on_done_task(command_task)
        if self._stopped_event.is_set():
            # 生命周期已经移交了.
            return
        # 发现任何任务出错超出预期.
        if self._interpretation.observe:
            if self._clear_after_exit:
                # 中断所有的运行.
                tasks = self._managing_tasks.values()
                for task in tasks:
                    if not task.done():
                        task.cancel("interpreter stopped for observe")
            self._stopped_event.set()

        if len(self._on_task_done_callbacks) > 0:
            for callback in self._on_task_done_callbacks:
                try:
                    callback(command_task)
                except Exception as e:
                    self._logger.exception(
                        "%s call command task done callback %s failed: %s",
                        self._log_prefix,
                        callback,
                        e,
                    )

    def meta_instruction(self) -> str:
        if self._meta_instruction is None:
            self._meta_instruction = get_moss_ctml_meta_instruction()
        return self._meta_instruction

    def channels(self) -> dict[str, ChannelMeta]:
        return self._channel_metas

    def static_messages(self) -> str:
        if self._moss_static is None:
            self._moss_static = make_static_messages(self._channel_metas)
        return self._moss_static

    def dynamic_messages(self) -> list[Message]:
        if self._moss_dynamic is None:
            self._moss_dynamic = make_dynamic_messages(self._channel_metas)
        return self._moss_dynamic

    def feed(self, delta: str, throw: bool = False) -> bool:
        if not isinstance(delta, str):
            raise ValueError("delta must be a string")
        if self._committed:
            if throw:
                raise InterpretError(f"interpreter already committed ")
            return False

        if self._closed:
            if throw:
                raise InterpretError(f"interpreter already closed")
            return False

        if self._parsing_exception is not None:
            if throw:
                raise self._parsing_exception
            return False
        if self._stopped_event.is_set():
            if throw:
                raise InterpretError(f"Interpretation stopped")
            return False
        self._interpretation.feed_inputs.append(delta)
        self._input_deltas_queue.put_nowait(delta)
        return True

    def commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        self._input_deltas_queue.put_nowait(None)

    def on_task_compiled(self, *callbacks: CommandTaskCallback) -> None:
        self._on_task_created_callbacks.extend(callbacks)

    def on_task_done(self, *callbacks: CommandTaskCallback) -> None:
        self._on_task_done_callbacks.extend(callbacks)

    def text_token_parser(self) -> TextTokenParser:
        """
        实现无副作用的 TokenParser 返回.
        """
        # create token parser
        return CTML2CommandTokenParser(
            callback=None,
            stream_id=self.id,
            root_tag=self._root_tag,
            tokens_replacement=self._token_replacement,
            attr_parsers=self._ctml_attr_parser,
        )

    def command_token_parser(self) -> CommandTokenParser:
        ctx = CommandTaskElementContext(
            channel_commands=self._channel_command_map,
            speech=self._speech,
            logger=self._logger,
            ignore_wrong_command=self._ignore_wrong_command,
        )
        return ctx.new_root(
            callback=None,
            stream_id=self.id,
        )

    def parsed_tokens(self) -> Iterable[CommandToken]:
        return self._interpretation.command_tokens.copy()

    def compiled_tasks(self) -> dict[str, CommandTask]:
        return self._compiled_tasks.copy()

    async def wait_stopped(self) -> Interpretation:
        if self.is_running():
            await self._stopped_event.wait()
        return self._interpretation

    def received_text(self) -> str:
        return "".join(self._interpretation.feed_inputs)

    def _text_to_command_token_parse_loop(self) -> None:
        try:
            self.parse_text_to_command_tokens(
                self._input_deltas_queue,
                self._receive_command_token,
                stopped=self._stopped_event.is_set,
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.exception("%s Interpret failed: %s", self._log_prefix, exc)
            raise
        finally:
            self._logger.info("%s text token parser loop stopped", self._log_prefix)
            self._receive_command_token(None)

    async def _command_token_to_tasks_parse_loop(self) -> None:
        task_parser = self.command_token_parser()
        try:
            await self.parse_tokens_to_command_tasks(
                tokens_queue=self._text_to_parsed_tokens_queue,
                task_callback=self._send_command_task,
                stopped=self._stopped_event.is_set,
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.exception("%s Parse command task failed", self._log_prefix)
            raise e
        finally:
            task_parser.destroy()

    async def _wait_task_done_then_stop(self) -> None:
        """
        唯一的目的, 是为了 tasks done 后设置 stopped 为 True.
        """
        wait_parse_done = asyncio.create_task(self._parsing_loop_done.wait())
        wait_stopped = asyncio.create_task(self._stopped_event.wait())
        done, pending = await asyncio.wait([wait_parse_done, wait_stopped], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if self._stopped_event.is_set():
            return
        tasks = self._managing_tasks.values()
        wait_all_task_done = asyncio.gather(*[t.wait(throw=False) for t in tasks])
        wait_stopped = asyncio.create_task(self._stopped_event.wait())
        done, pending = await asyncio.wait([wait_all_task_done, wait_stopped], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if wait_all_task_done in done:
            self._stopped_event.set()

    async def _main_parsing_loop(self) -> None:
        try:
            token_parse_loop = asyncio.create_task(asyncio.to_thread(self._text_to_command_token_parse_loop))
            task_parse_loop = asyncio.create_task(self._command_token_to_tasks_parse_loop())
            await asyncio.gather(token_parse_loop, task_parse_loop)
        except asyncio.CancelledError:
            pass
        except InterpretError as e:
            self._logger.exception("%s Parse command task failed %s", self._log_prefix, e)
            self._set_interpreter_error(e)
        except Exception as e:
            self._logger.exception("%s Interpreter main parsing loop failed: %s", self._log_prefix, e)
            self._set_interpreter_error(InterpretError(f"interpreter failed: {e}"))
        finally:
            # 主循环如果发生错误, interpreter 会终止. 这时并不会结束所有的任务.
            self._parsing_loop_done.set()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            if isinstance(exc_val, asyncio.CancelledError):
                await self.close(cancel_executing=True)
                return True
            if not isinstance(exc_val, InterpretError):
                self._logger.exception("Interpreter quit on exception %s", exc_val)
                await self.close(cancel_executing=True)
                return None
        await self.close(cancel_executing=False)
        return None

    def exception(self) -> Optional[Exception]:
        return self._parsing_exception

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("Interpreter is already started")
        self._started = True
        self._loop = asyncio.get_running_loop()
        if self._on_startup:
            await self._on_startup()
        # 启动主循环.
        task = asyncio.create_task(self._main_parsing_loop())
        self._main_parsing_loop_task = task
        self._tasks_done_then_stop_task = asyncio.create_task(self._wait_task_done_then_stop())

    async def close(self, cancel_executing: bool = True) -> Interpretation | None:
        if not self._started:
            return None
        if self._closed:
            return None
        self._closed = True
        self._interpretation.interrupted = not self._stopped_event.is_set()
        self._stopped_event.set()
        self._logger.info("%s interpreter stopping", self._log_prefix)
        try:
            if self._main_parsing_loop_task and not self._main_parsing_loop_task.done():
                self._main_parsing_loop_task.cancel()
                await self._main_parsing_loop_task
        except asyncio.CancelledError:
            pass
        try:
            if self._tasks_done_then_stop_task and not self._tasks_done_then_stop_task.done():
                self._tasks_done_then_stop_task.cancel()
                await self._tasks_done_then_stop_task
        except asyncio.CancelledError:
            pass

        if cancel_executing or self._clear_after_exit:
            for t in self._managing_tasks.values():
                if not t.done():
                    t.fail(CommandErrorCode.INTERRUPTED.error("interpreter stopped"))

        self._logger.info("interpreter %s stopped", self.id)
        # 关闭所有未执行完的任务.
        if self._interrupted and not self._parsing_exception:
            self._parsing_exception = InterpretError("Interpretation is interrupted")
        if self._parsing_exception:
            self._interpretation.exception = str(self._parsing_exception)
        self._interpretation.done = True
        r = self._interpretation
        return r

    def is_stopped(self) -> bool:
        return self._stopped_event.is_set()

    def is_closed(self) -> bool:
        return self._closed

    def is_running(self) -> bool:
        return self._started and not self._stopped_event.is_set() and not self._closed

    def is_interrupted(self) -> bool:
        return self._interpretation.interrupted

    async def wait_compiled(self, timeout: float | None = None, throw: bool = True) -> None:
        try:
            if not self._started:
                return
            self._started = True
            self.commit()
            # 等待主循环结束.
            wait_parsing_loop = asyncio.create_task(self._parsing_loop_done.wait())
            # 等待 stop 标记.
            wait_stop_event = asyncio.create_task(self._stopped_event.wait())
            tasks = [wait_parsing_loop, wait_stop_event]
            # 超时等待.
            timeout_task = None
            if timeout is not None and timeout > 0.0:
                timeout_task = asyncio.create_task(asyncio.sleep(timeout))
                tasks.append(timeout_task)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if timeout_task in done:
                raise asyncio.TimeoutError("Timed out while waiting for parser to finish")
            if throw and self._parsing_exception:
                raise self._parsing_exception
        except asyncio.CancelledError:
            self._logger.info("wait parser done is cancelled")
            pass
        except InterpretError as e:
            self._logger.exception("%s stopped due to exception: %s", self._log_prefix, e)
            self._set_interpreter_error(e)
            if throw:
                raise
        except Exception as exc:
            self._logger.exception("Wait parse done failed")
            err = InterpretError(f"Interpret failed: {exc}")
            self._set_interpreter_error(err)
            if throw:
                raise err

    async def wait_tasks(
            self,
            timeout: float | None = None,
            *,
            return_when: str = asyncio.ALL_COMPLETED,
            throw: bool = False,
            clear_undone: bool = True,
    ) -> dict[str, CommandTask]:
        # 先等待到解释器结束.
        timeleft = Timeleft(timeout or 0.0)
        # 阻塞等待解析完成.
        await self.wait_compiled(timeout, throw=throw)

        # 编译完已经超时了.
        if throw and not timeleft.alive():
            raise asyncio.TimeoutError("Timed out while waiting for parsed command tasks to finish")

        # 拿到编译完的 tasks.
        tasks = self._managing_tasks.copy()
        if len(tasks) == 0:
            return tasks

        # 按约定等待所有 task.
        waiting_tasks = []
        for t in tasks.values():
            waiting_tasks.append(asyncio.create_task(t.wait(throw=False)))

        err = None
        try:
            # 阻塞等待运行完成.
            done, pending = await asyncio.wait(
                waiting_tasks,
                timeout=timeleft.left() or None,
                return_when=return_when,
            )
            for t in pending:
                t.cancel()

            if throw:
                for task in tasks.values():
                    if exp := task.exception():
                        # 根据结果判断是否抛出异常.
                        raise exp

            # 返回所有的 tasks.
            return tasks
        except asyncio.CancelledError:
            self._logger.info("wait execution done is cancelled")
            pass
        except Exception as e:
            # 发生了预期外的异常.
            self._logger.exception("Wait execution done failed")
            err = e
            raise e
        finally:
            if clear_undone:
                for task in tasks.values():
                    if not task.done():
                        # 取消所有未完成的任务.
                        task.fail(err or CommandErrorCode.CLEARED.error("wait execution done"))
        return tasks

    def __del__(self):
        # 丢弃这个计算代码.
        CTMLInterpreter.instances_count -= 1
        if not self._destroyed:
            self.destroy()

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self._logger:
            self._logger.debug(
                "%s destroyed, CTMLInterpreter count: %d, Task count: %d",
                self._log_prefix,
                CTMLInterpreter.instances_count,
                CommandTask.instances_count,
            )
        # 确保所有的 element 被销毁了. 否则会有内存泄漏的风险.
        self._commands_map.clear()
        self._channel_metas = None
        self._channel_command_map.clear()
        self._on_task_created_callbacks.clear()
        self._managing_tasks.clear()
        self._compiled_tasks.clear()
        self._speech = None
        if self._outputted:
            self._outputted.clear()
