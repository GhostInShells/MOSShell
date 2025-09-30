from typing import Optional, Iterable, Dict, List, AsyncIterable
from ghoshell_moss.concepts.interpreter import (
    Interpreter, CommandTaskCallback, CommandTaskElement, CommandTokenParser,
)
from ghoshell_moss.concepts.shell import Output
from ghoshell_moss.concepts.command import CommandToken, Command, CommandTask
from ghoshell_moss.concepts.errors import CommandError
from ghoshell_moss.ctml.token_parser import CTMLTokenParser
from ghoshell_moss.ctml.elements import CommandTaskElementContext
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.helpers import uuid
import logging
import asyncio
import queue


class CTMLInterpreter(Interpreter):

    def __init__(
            self,
            *,
            commands: Iterable[Command],
            output: Output,
            stream_id: Optional[str] = None,
            callback: Optional[CommandTaskCallback] = None,
            root_tag: str = "ctml",
            special_tokens: Optional[Dict[str, str]] = None,
            logger: Optional[logging.Logger] = None,
    ):
        self.id = stream_id or uuid()
        self._logger = logger or logging.getLogger("CTMLInterpreter")
        self._callbacks = []
        if callback is not None:
            self._callbacks.append(callback)
        # commands map
        self._commands_map = {c.name(): c for c in commands}
        self._root_tag = root_tag
        self._special_tokens = special_tokens or {}
        self._stopped_event = ThreadSafeEvent()
        self._fatal_exception: Optional[Exception] = None

        # output related
        self._output = output
        self._outputted: Optional[List[str]] = None

        # create token parser
        self._parser = CTMLTokenParser(
            callback=self._receive_command_token,
            stream_id=self.id,
            root_tag=root_tag,
            special_tokens=special_tokens,
            stop_event=self._stopped_event,
        )
        self._parsed_tokens = []
        # 用线程安全队列就可以. 考虑到队列可能不是在同一个 loop 里添加
        self._input_deltas_queue: queue.Queue[str | None] = queue.Queue()

        # create task element
        self._parsed_tokens_queue: queue.Queue[CommandToken | None] = queue.Queue()
        self._task_element_ctx = CommandTaskElementContext(
            commands=self._commands_map,
            output=self._output,
            logger=self._logger,
            stop_event=self._stopped_event,
        )
        self._root_element = self._task_element_ctx.new_root(
            callback=self._send_command_task,
            stream_id=self.id,
        )
        self._parsed_tasks: Dict[str, CommandTask] = {}

        # input buffer
        self._input_buffer: str = ""

        # runtime
        self._main_task: Optional[asyncio.Task] = None
        self._started = False
        self._committed = False
        self._interrupted = False
        self._task_sent = False
        self._main_loop_done = asyncio.Event()

    def _receive_command_token(self, token: CommandToken | None) -> None:
        if token is not None:
            self._parsed_tokens.append(token)
        self._parsed_tokens_queue.put(token)

    def _send_command_task(self, task: CommandTask | None) -> None:
        try:
            if len(self._callbacks) > 0:
                # 只发送一次 None 作为毒丸.
                if task is not None or not self._task_sent:
                    for callback in self._callbacks:
                        callback(task)
                    self._task_sent = task is None
            if task is not None:
                self._parsed_tasks[task.cid] = task
        except Exception as e:
            self._fatal_exception = e
            self._logger.exception(e)
            self._stopped_event.set()

    def meta_instruction(self) -> str:
        raise NotImplementedError

    def feed(self, delta: str) -> None:
        if not self._committed and not self._stopped_event.is_set():
            if self._fatal_exception is not None:
                raise self._fatal_exception

            self._input_buffer += delta
            self._input_deltas_queue.put_nowait(delta)

    async def parse(self, deltas: AsyncIterable[str]) -> None:
        try:
            async for delta in deltas:
                self.feed(delta)
        except Exception as e:
            self._logger.exception(e)
            self._stopped_event.set()
        finally:
            self.commit()

    def commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        self._input_deltas_queue.put_nowait(None)

    def with_callback(self, callback: CommandTaskCallback) -> None:
        self._callbacks.append(callback)

    def parser(self) -> CommandTokenParser:
        return self._parser

    def root_task_element(self) -> CommandTaskElement:
        return self._root_element

    def parsed_tokens(self) -> Iterable[CommandToken]:
        return self._parsed_tokens.copy()

    def parsed_tasks(self) -> Dict[str, CommandTask]:
        return self._parsed_tasks.copy()

    def outputted(self) -> Iterable[str]:
        if self._outputted is None:
            return self._output.outputted()
        return self._outputted

    def results(self) -> Dict[str, str]:
        raise NotImplementedError("todo")

    def executed(self) -> str:
        raise NotImplementedError("todo")

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
        except Exception as exc:
            # todo
            self._logger.exception(exc)
            self._fatal_exception = exc
            self._stopped_event.set()
        finally:
            # todo
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
        except Exception as e:
            # todo
            self._logger.exception(e)
            self._fatal_exception = e
            self._stopped_event.set()
        finally:
            # todo
            pass

    async def _main(self) -> None:
        try:
            token_parse_loop = asyncio.to_thread(self._token_parse_loop)
            task_parse_loop = asyncio.to_thread(self._task_parse_loop)
            await asyncio.gather(token_parse_loop, task_parse_loop)
        except Exception as exc:
            self._logger.exception(exc)
        finally:
            for task in self._parsed_tasks.values():
                if not task.done():
                    task.cancel()
            self._main_loop_done.set()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        task = asyncio.create_task(self._main())
        self._main_task = task

    async def stop(self) -> None:
        if self._stopped_event.is_set():
            return
        self._interrupted = self._started and not self._main_loop_done.is_set()
        self._stopped_event.set()
        self._parser.close()
        if self._main_task is not None:
            await self._main_task
        if self._interrupted:
            for task in self._parsed_tasks.values():
                if not task.done():
                    task.cancel("interrupted")

    def is_stopped(self) -> bool:
        return self._stopped_event.is_set()

    def is_running(self) -> bool:
        return self._started and not self._stopped_event.is_set()

    def is_interrupted(self) -> bool:
        return self._interrupted

    async def wait_parse_done(self) -> None:
        if not self._started:
            return
        self.commit()
        # 等待主循环结束.
        await self._main_loop_done.wait()
        if self._fatal_exception:
            raise self._fatal_exception

    async def wait_execution_done(self) -> None:
        await self.wait_parse_done()
        waits = []
        tasks = list(self._parsed_tasks.values())
        for task in tasks:
            waits.append(task.wait())
        try:
            if len(waits) > 0:
                await asyncio.gather(*waits, return_exceptions=False)
        except asyncio.CancelledError:
            pass
        except CommandError:
            pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel("execution done")

    def __del__(self) -> None:
        self._parser.close()
        if self._root_element:
            self._root_element.destroy()
