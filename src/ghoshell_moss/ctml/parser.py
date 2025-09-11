import queue
import threading
from xml import sax

import logging
import xml.sax
from xml.sax import saxutils
from typing import List, Iterable, Optional, Dict, Any, Callable
from xml import sax
from ghoshell_moss.concepts.interpreter import CommandParser
from ghoshell_moss.concepts.command import CommandToken, CommandTask
from ghoshell_moss.concepts.errors import CommandError, InterpretError
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_common.helpers import uuid
from queue import Queue


class CMTLElement:

    def __init__(self, *, idx: int, stream_id: str, chan: str, name: str, qname: str, attrs):
        self.idx = idx
        self.qname = qname
        self.ns = chan
        self.name = name
        self.deltas = ""
        self.part_idx = 0
        self._has_delta = False
        self.attrs = dict(attrs)
        self.stream_id = stream_id

    def start_token(self) -> CommandToken:
        attr_expression = []
        for k, v in self.attrs.items():
            quoted_value = saxutils.quoteattr(v)
            attr_expression.append(f"{k}={quoted_value}")
        content = f"<{self.qname} " + " ".join(attr_expression) + ">"
        part_idx = self.part_idx
        self.part_idx += 1
        return CommandToken(
            name=self.name,
            chan=self.ns,
            idx=self.idx,
            part_idx=part_idx,
            stream_id=self.stream_id,
            type="start",
            kwargs=self.attrs,
            content=content,
        )

    def on_child_command(self):
        if self._has_delta:
            self._has_delta = False
            self.deltas = ""
            self.part_idx += 1

    def add_delta(self, delta: str, gen_token: bool = True) -> Optional[CommandToken]:
        self.deltas += delta
        if not self._has_delta:
            self._has_delta = len(delta.strip()) > 0
            if self._has_delta:
                # fist none empty delta
                delta = self.deltas

        if gen_token and self._has_delta:
            return CommandToken(
                name=self.name,
                chan=self.ns,
                idx=self.idx,
                part_idx=self.part_idx,
                stream_id=self.stream_id,
                type="delta",
                kwargs=None,
                content=delta,
            )
        return None

    def end_token(self) -> CommandToken:
        if self._has_delta:
            self.part_idx += 1
        return CommandToken(
            name=self.name,
            chan=self.ns,
            idx=self.idx,
            part_idx=self.part_idx,
            stream_id=self.stream_id,
            type="end",
            kwargs=None,
            content=f"</{self.qname}>",
        )


class ParsingStoppedError(Exception):
    pass


class CTMLTokenSaxHandler(xml.sax.ContentHandler, xml.sax.ErrorHandler):

    def __init__(
            self,
            root_tag: str,
            stream_id: str,
            callback: Callable[[CommandToken | InterpretError | None], None],
            default_chan: str = "",
            logger: Optional[logging.Logger] = None,
    ):
        self.done_event = threading.Event()
        self._root = root_tag
        self._stream_id = stream_id
        self._default_chan = default_chan
        self._idx = 0
        self._callback = callback
        self._logger = logger or logging.getLogger("CTMLTokenSaxHandler")
        self._parsing_elements: List[CMTLElement] = []
        self._ended_command_tokens = []

    def send_token(self, token: CommandToken) -> None:
        if self._callback is not None and not self.done_event.is_set():
            self._callback(token)

    def startElementNS(self, name: tuple[str, str], qname: str, attrs: Dict) -> None:
        if self.done_event.is_set():
            raise ParsingStoppedError("Done event already set")

        cmd_chan, cmd_name = name
        element = CMTLElement(
            idx=self._idx,
            stream_id=self._stream_id,
            chan=cmd_chan or self._default_chan,
            name=cmd_name,
            qname=qname,
            attrs=attrs,
        )
        if len(self._parsing_elements) > 0:
            self._parsing_elements[-1].on_child_command()

        self._parsing_elements.append(element)
        self._idx += 1
        self.send_token(element.start_token())

    def characters(self, content: str):
        if len(self._parsing_elements) == 0:
            # todo
            raise RuntimeError(f"todo")
        element = self._parsing_elements[-1]
        token = element.add_delta(content)
        if token is not None:
            self.send_token(token)

    def endElement(self, name: str):
        if len(self._parsing_elements) == 0:
            # todo
            raise RuntimeError(f"todo")
        element = self._parsing_elements.pop(-1)
        token = element.end_token()
        self.send_token(token)

    def endDocument(self):
        if self._callback is not None:
            self._callback(None)
        # todo: logging
        self.done_event.set()

    def error(self, exception: Exception):
        self.done_event.set()
        if isinstance(exception, ParsingStoppedError):
            return
        self._logger.exception(exception)
        error = InterpretError(str(exception))
        self._callback(error)

    def fatalError(self, exception: Exception):
        self.done_event.set()
        if isinstance(exception, ParsingStoppedError):
            return
        self._logger.exception(exception)
        error = InterpretError(str(exception))
        self._callback(error)

    def warning(self, exception):
        self._logger.warning(exception)


class CMTLSaxRunner:
    """
    single time runner for Sax + CTMLTokenSaxHandler
    todo: 实现一个可复用的 Runner, 又可以随时清理掉.
    """

    def __init__(
            self,
            handler: CTMLTokenSaxHandler,
    ):
        self._handler = handler
        self._sax_parser = sax.make_parser()
        self._sax_parser.setContentHandler(handler)
        self._sax_parser.setErrorHandler(handler)
        self._parsing_queue: queue.Queue[str | None] = queue.Queue()

    def put(self, delta: str) -> None:
        self._parsing_queue.put(delta)

    def cancel(self) -> None:
        self._parsing_queue.put(None)
        self._handler.done_event.set()

    def main_loop(self) -> None:
        try:
            self._sax_parser.start()
            while not self._handler.done_event.is_set():
                try:
                    item = self._parsing_queue.get(block=True, timeout=0.1)
                    if item is None:
                        break
                    self._sax_parser.feed(item)
                except queue.Empty:
                    continue
            # sax parsing stream is done
        finally:
            self._sax_parser.stop()
            self._sax_parser.close()


class CMTLInterpreter(Interpreter):

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def wait_until_done(self) -> bool:
        pass

    def put(self, delta: str) -> None:
        pass

    def cancel(self) -> None:
        pass

    def with_command_token_callback(self, callback: Callable[[CommandToken], None]) -> None:
        pass

    def with_command_task_callback(self, callback: Callable[[CommandTask], None]) -> None:
        pass

    def parsed_tokens(self) -> Iterable[CommandToken]:
        pass

    def parsed_tasks(self) -> Iterable[CommandTask]:
        pass
