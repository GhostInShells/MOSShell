import asyncio
import datetime
import logging
import queue
from collections.abc import AsyncIterable, Callable, Coroutine, Iterable
from itertools import starmap
from typing import Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import Timeleft, uuid

from ghoshell_moss.core.concepts.channel import ChannelFullPath, ChannelMeta
from ghoshell_moss.core.concepts.command import Command, CommandTask, CommandTaskState, CommandToken
from ghoshell_moss.core.concepts.errors import CommandErrorCode, InterpretError
from ghoshell_moss.core.concepts.interpreter import (
    CommandTaskCallback,
    CommandTokenParserElement,
    TextTokenParser,
    Interpreter,
)
from ghoshell_moss.core.concepts.speech import Speech
from ghoshell_moss.core.ctml.elements import CommandTaskElementContext
from ghoshell_moss.core.ctml.prompt import get_moss_meta_prompt
from ghoshell_moss.core.ctml.token_parser import CTMLTokenParser, ParserStopped
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.message import Message

__all__ = [
    "DEFAULT_META_PROMPT",
    "CTMLInterpreter",
    "make_chan_prompt",
    "make_channels_prompt",
]

DEFAULT_META_PROMPT = get_moss_meta_prompt()

_Title = str
_Description = str
_Interface = str


def make_chan_prompt(channel_path: str, description: str, interface: str) -> str:
    python_interface = f"```python\n{interface}\n```\n" if interface else ""
    return f"""
## channel `{channel_path}`
{description}
{python_interface}
"""


def make_channels_prompt(channel_metas: dict[str, ChannelMeta]) -> str:
    channel_items: list[tuple[_Title, _Description, _Interface]] = []
    if len(channel_metas) == 0:
        return ""
    if "" in channel_metas:
        main = channel_metas.pop("")
        if len(main.commands) > 0:
            interface = "\n\n".join([c.interface for c in main.commands])
            channel_items.append(("", "main channel commands (do not need channel namespaces):", interface))
    for channel_path, channel_meta in channel_metas.items():
        channel_items.append(
            (
                channel_path,
                channel_meta.description,
                "\n\n".join([c.interface for c in channel_meta.commands]),
            )
        )
    if len(channel_items) == 0:
        # 返回空.
        return ""
    body = "\n\n".join(list(starmap(make_chan_prompt, channel_items)))
    return f"# MOSS Channels\n\n{body}"


class CTMLInterpreter(Interpreter):
    def __init__(
            self,
            *,
            commands: dict[ChannelFullPath, dict[str, Command]],
            speech: Speech,
            stream_id: Optional[str] = None,
            callback: Optional[CommandTaskCallback] = None,
            root_tag: str = "ctml",
            special_tokens: Optional[dict[str, str]] = None,
            logger: Optional[LoggerItf] = None,
            on_startup: Optional[Callable[[], Coroutine[None, None, None]]] = None,
            meta_system_prompt: Optional[str] = None,
            channel_metas: Optional[dict[ChannelFullPath, ChannelMeta]] = None,
    ):
        """
        :param commands: 所有 interpreter 可以使用的命令. key 是 channel path, value 是这个 channel 可以用的 commands.
        :param speech: 用来发送所有模型非 command 输出的内容.
        :param stream_id: 让 interpreter 有一个唯一的 id.
        :param callback: command task callback
        :param root_tag: 决定生成 command token 的起始和结尾标记. 通常没有功能性.
        :param special_tokens: 如果传入, 在解析时会把 输出的 key token 转换成 value token 然后解析. 用来做快速匹配.
        :param logger: 日志.
        :param on_startup: 可以定义额外的启动函数.
        :param meta_system_prompt: MOSS 解释器的基础语法规则, 如果为空则使用默认的.
        :param channel_metas: 用来定义当前所拥有的 channels 信息, 用来提供给大模型.
        """
        # 生成 stream id.
        self.id = stream_id or uuid()
        self._meta_instruction = meta_system_prompt
        self._channel_metas = channel_metas or {}
        # 准备日志.
        self._logger = logger or logging.getLogger("CTMLInterpreter")
        # 可用的 task 回调.
        self._callbacks: list[CommandTaskCallback] = []
        if callback is not None:
            self._callbacks.append(callback)
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
                unique_name = Command.make_uniquename(channel_path, command_name)
                self._commands_map[unique_name] = command

        self._root_tag = root_tag
        self._special_tokens = special_tokens or {}
        self._stopped_event = ThreadSafeEvent()
        self._parsing_exception: Optional[Exception] = None

        # output related
        self._speech = speech
        self._outputted: Optional[list[str]] = None

        # create token parser
        self._parser = CTMLTokenParser(
            callback=self._receive_command_token,
            stream_id=self.id,
            root_tag=root_tag,
            special_tokens=special_tokens,
            stop_event=self._stopped_event,
        )
        # 用线程安全队列就可以. 考虑到队列可能不是在同一个 loop 里添加
        self._input_deltas_queue: queue.Queue[str | None] = queue.Queue()
        # 内部传输 tokens 的通道.
        self._parsed_tokens_queue: queue.Queue[CommandToken | None] = queue.Queue()

        # create task element
        self._task_element_ctx = CommandTaskElementContext(
            channel_commands=self._channel_command_map,
            speech=self._speech,
            logger=self._logger,
            stop_event=self._stopped_event,
        )
        self._root_element = self._task_element_ctx.new_root(
            callback=self._send_command_task,
            stream_id=self.id,
        )

        # input buffer
        self._input_buffer: str = ""

        #  --- runtime --- #
        self._parsed_tasks: dict[str, CommandTask] = {}  # 解析生成的 tasks.
        self._parsed_tokens = []  # 解析生成的 tokens.
        self._main_parsing_task: Optional[asyncio.Task] = None  # 解析的主循环.
        self._started = False
        self._committed = False
        self._interrupted = False
        self._task_sent_done = False
        self._parsing_loop_done = asyncio.Event()  # 标记解析完成.

    def _receive_command_token(self, token: CommandToken | None) -> None:
        """将 token 记录到解析后的 tokens 中."""
        if self._stopped_event.is_set():
            return
        if token is not None:
            self._parsed_tokens.append(token)
        self._parsed_tokens_queue.put(token)

    def _send_command_task(self, task: CommandTask | None) -> None:
        try:
            if self._task_sent_done:
                return
            if self._stopped_event.is_set():
                return

            if len(self._callbacks) > 0:
                # 只发送一次 None 作为毒丸.
                if task is not None:
                    # 添加新的 task.
                    self._parsed_tasks[task.cid] = task
                    # 注册 task 的回调, 如果出了异常就干脆中断整个流程, 也别解析了.
                    task.add_done_callback(self._on_task_done)
                for callback in self._callbacks:
                    callback(task)
                self._task_sent_done = task is None
        except Exception as e:
            self._parsing_exception = InterpretError(f"Send command failed: {e}")
            self._logger.exception("Send command task failed")
            self._stopped_event.set()

    def _on_task_done(self, command_task: CommandTask) -> None:
        if self._stopped_event.is_set():
            return
        # 发现任何任务出错.
        if exception := command_task.exception():
            # 中断所有的运行.
            self._stopped_event.set()
            self._parsing_exception = exception

    def meta_system_prompt(self) -> str:
        return self._meta_instruction or DEFAULT_META_PROMPT

    def channels(self) -> dict[str, ChannelMeta]:
        return self._channel_metas

    def moss_instruction(self) -> str:
        channels_prompt = make_channels_prompt(self._channel_metas)
        if channels_prompt:
            meta_system_prompt = self.meta_system_prompt()
            return "\n\n".join([meta_system_prompt, channels_prompt])
        return ""

    def context_messages(self, *, channel_names: list[str] | None = None) -> list[Message]:
        channel_names = channel_names or self._channel_metas.keys()
        messages = []
        for channel_path_name in channel_names:
            meta = self._channel_metas.get(channel_path_name)
            if meta is not None and meta.context:
                messages.append(
                    Message.new(role="system")
                    .with_content(
                        f"<channel-context:{channel_path_name}>",
                    )
                    .as_completed(),
                )

                messages.extend(meta.context)
                messages.append(
                    Message.new(role="system")
                    .with_content(
                        f"</channel-context:{channel_path_name}>",
                    )
                    .as_completed(),
                )
        return messages

    def feed(self, delta: str) -> None:
        if not self._committed and not self._stopped_event.is_set():
            if self._parsing_exception is not None:
                raise self._parsing_exception

            self._input_buffer += delta
            self._input_deltas_queue.put_nowait(delta)

    async def parse(self, deltas: AsyncIterable[str]) -> None:
        try:
            async for delta in deltas:
                self.feed(delta)
        except Exception:
            self._logger.exception("Stream parse failed")
            self._stopped_event.set()
        finally:
            self.commit()

    def commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        self._input_deltas_queue.put_nowait(None)

    def with_task_callback(self, *callbacks: CommandTaskCallback) -> None:
        callbacks = list(callbacks)
        callbacks.extend(self._callbacks)
        self._callbacks = callbacks

    def text_token_parser(self) -> TextTokenParser:
        return self._parser

    def command_token_parser(self) -> CommandTokenParserElement:
        return self._root_element

    def parsed_tokens(self) -> Iterable[CommandToken]:
        return self._parsed_tokens.copy()

    def compiled_tasks(self) -> dict[str, CommandTask]:
        return self._parsed_tasks.copy()

    def outputted(self) -> Iterable[str]:
        if self._outputted is None:
            return self._speech.outputted()
        return self._outputted

    async def results(self) -> dict[str, str]:
        tasks = await self.wait_execution_done()
        results = {}
        for task in tasks.values():
            done_at = task.last_trace[1]
            if done_at:
                done_at_str = datetime.datetime.fromtimestamp(done_at or 0.0).strftime("%Y-%m-%d %H:%M:%S")
                done_at_str = f"[done at:{done_at_str}] "
            else:
                done_at_str = ""
            if task.success():
                result = task.result()
                if result is not None:
                    try:
                        cmd_result = str(result).strip()
                        if cmd_result:
                            results[task.tokens] = f"{cmd_result}{done_at_str}"
                    except ValueError:
                        self._logger.exception("Format command result failed")
                        pass
            else:
                error_info = CommandErrorCode.description(task.errcode, task.errmsg)
                results[task.tokens] = f"{error_info}{done_at_str}"
                break
        return results

    def executed(self) -> list[CommandTask]:
        tasks = self.compiled_tasks().copy()
        executions = []
        for task in tasks.values():
            if CommandTaskState.is_complete(task.state):
                executions.append(task)
            else:
                break
            if CommandTaskState.is_stopped(task.state):
                break
        return executions

    def inputted(self) -> str:
        return self._input_buffer

    def _token_parse_loop(self) -> None:
        try:
            with self._parser:
                while not self._stopped_event.is_set() and not self._parser.is_done():
                    try:
                        # check every 0.1 second if the loop is stopped.
                        item = self._input_deltas_queue.get(block=True, timeout=0.1)
                        if item is None:
                            self._parser.commit()
                            break
                        self._parser.feed(item)
                    except queue.Empty:
                        continue
        except asyncio.CancelledError:
            self._logger.info("interpreter %s cancelled", self.id)
        except ParserStopped as e:
            self._logger.info("interpreter %s parser stopped", self.id)
            # self._parsing_exception = InterpretError(f"Parse output stream failed: {e}")
            self._stopped_event.set()
        except Exception as exc:
            self._logger.exception("Interpret failed")
            self._parsing_exception = InterpretError(f"Interpret failed: {exc}")
            self._stopped_event.set()
            raise
        finally:
            pass

    def _task_parse_loop(self) -> None:
        try:
            while not self._stopped_event.is_set():
                try:
                    item = self._parsed_tokens_queue.get(block=True, timeout=0.1)
                    self._root_element.on_token(item)
                    if item is None or self._root_element.is_end():
                        break
                except queue.Empty:
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # todo
            self._logger.exception("Parse command task failed")
            self._parsing_exception = InterpretError(f"Parse command task failed at `{type(e)}`: {e}")
            self._stopped_event.set()
        finally:
            # todo
            pass

    async def _main_parsing_loop(self) -> None:
        try:
            token_parse_loop = asyncio.to_thread(self._token_parse_loop)
            task_parse_loop = asyncio.to_thread(self._task_parse_loop)
            await asyncio.gather(token_parse_loop, task_parse_loop)
        except asyncio.CancelledError:
            pass
        except Exception:
            self._logger.exception("Interpreter main parsing loop failed")
        finally:
            # 主循环如果发生错误, interpreter 会终止. 这时并不会结束所有的任务.
            self._parsing_loop_done.set()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._on_startup:
            await self._on_startup()
        # 启动主循环.
        task = asyncio.create_task(self._main_parsing_loop())
        self._main_parsing_task = task

    async def stop(self) -> None:
        if self._stopped_event.is_set():
            await self._parsing_loop_done.wait()
            return
        self._logger.info("interpreter %s stopping", self.id)
        self._interrupted = self._started and not self._parsing_loop_done.is_set()
        self._stopped_event.set()
        try:
            self._parser.close()
        except ParserStopped:
            pass
        try:
            if self._main_parsing_task:
                self._main_parsing_task.cancel()
                await self._main_parsing_task
        except asyncio.CancelledError:
            pass

        self._logger.info("interpreter %s stopped", self.id)
        # 关闭所有未执行完的任务.
        if self._interrupted:
            self._parsing_exception = InterpretError("Interpretation is interrupted")

    def is_stopped(self) -> bool:
        return self._stopped_event.is_set()

    def is_running(self) -> bool:
        return self._started and not self._stopped_event.is_set()

    def is_interrupted(self) -> bool:
        return self._interrupted

    async def wait_compiled(self, timeout: float | None = None, throw: bool = True) -> None:
        try:
            if not self._started:
                return
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
            if self._parsing_exception:
                raise self._parsing_exception
        except asyncio.CancelledError:
            self._logger.info("wait parser done is cancelled")
            pass
        except ParserStopped:
            self._logger.info("wait parser done: parser is stopped")
            pass
        except Exception as exc:
            self._logger.exception("Wait parse done failed")
            if throw:
                if isinstance(exc, InterpretError):
                    raise exc
                else:
                    raise InterpretError(f"Interpret failed: {exc}") from exc

    async def wait_execution_done(
            self,
            timeout: float | None = None,
            throw: bool = False,
            cancel_on_exception: bool = True,
    ) -> dict[str, CommandTask]:
        # 先等待到解释器结束.
        timeleft = Timeleft(timeout or 0.0)
        await self.wait_compiled(timeout, throw=throw)
        if throw and not timeleft.alive():
            raise asyncio.TimeoutError("Timed out while waiting for parsed command tasks to finish")

        gathering = []
        tasks = self.compiled_tasks()
        if len(tasks) == 0:
            return tasks

        waiting_tasks = []
        for t in tasks.values():
            waiting_tasks.append(asyncio.create_task(t.wait(throw=True)))

        err = None
        try:
            # ignore
            done, pending = await asyncio.wait(
                waiting_tasks,
                timeout=timeleft.left() or None,
                return_when=asyncio.ALL_COMPLETED,
            )
            for t in pending:
                t.cancel()

            # 返回所有的 tasks.
            return tasks
        except asyncio.CancelledError:
            self._logger.info("wait execution done is cancelled")
            return tasks
        except InterpretError as e:
            # interpreter error 可以抛出.
            err = e
            if throw:
                raise
        except Exception as e:
            self._logger.exception("Wait execution done failed")
            # 不抛出其它异常.
            err = InterpretError(f"Interpreter failed: {e}")
            if throw:
                raise err
        finally:
            if err is not None and cancel_on_exception:
                for task in tasks.values():
                    if not task.done():
                        # 取消所有未完成的任务.
                        task.fail(err or "wait execution failed")

        return tasks

    def __del__(self) -> None:
        self._parser.close()
        # 确保所有的 element 被销毁了. 否则会有内存泄漏的风险.
        self._commands_map.clear()
        self._channel_metas = None
        self._channel_command_map.clear()
        if self._root_element:
            self._root_element.destroy()
