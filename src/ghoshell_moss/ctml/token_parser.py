import threading
from xml import sax

import logging
import xml.sax
from xml.sax import saxutils
from typing import List, Iterable, Optional, Callable, Dict, Set
from ghoshell_moss.concepts.command import CommandToken
from ghoshell_moss.concepts.interpreter import CommandTokenParser, CommandTokenParseError
from ghoshell_moss.helpers.token_filters import SpecialTokenMatcher
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent

CommandTokenCallback = Callable[[CommandToken | None], None]


class CMTLElement:
    """
    Utility class to generate Command Token in XMTL (Command Token Marked Language stream)
    """

    def __init__(self, *, cmd_idx: int, stream_id: str, name: str, attrs: dict):
        self.cmd_idx = cmd_idx
        self.name = name
        self.deltas = ""
        # first part idx is 0
        self.part_idx = 0
        self._has_delta = False
        self.attrs = attrs
        self.stream_id = stream_id

    @classmethod
    def make_fullname(cls, ns: str, name: str) -> str:
        return f"{ns}:{name}" if ns else name

    @staticmethod
    def make_start_mark(name: str, attrs: dict, self_close: bool) -> str:
        attr_expression = []
        for k, v in attrs.items():
            quoted_value = saxutils.quoteattr(v)
            attr_expression.append(f"{k}={quoted_value}")
        exp = " " if len(attr_expression) > 0 else ""
        self_close_mark = "/" if self_close else ""
        content = f"<{name}{exp}" + " ".join(attr_expression) + self_close_mark + ">"
        return content

    def start_token(self) -> CommandToken:
        content = self.make_start_mark(self.name, self.attrs, self_close=False)
        part_idx = self.part_idx
        self.part_idx += 1
        return CommandToken(
            name=self.name,
            cmd_idx=self.cmd_idx,
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
        if gen_token and len(delta) > 0:
            self.deltas += delta
            self._has_delta = True
            return CommandToken(
                name=self.name,
                cmd_idx=self.cmd_idx,
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
            cmd_idx=self.cmd_idx,
            part_idx=self.part_idx,
            stream_id=self.stream_id,
            type="end",
            kwargs=None,
            content=f"</{self.name}>",
        )


class ParserStopped(Exception):
    """notify the sax that parsing is stopped"""
    pass


class CTMLSaxHandler(xml.sax.ContentHandler, xml.sax.ErrorHandler):
    """初步实现 sax 解析. 实现得非常糟糕, 主要是对 sax 的回调机制有误解, 留下了大量冗余状态. 需要考虑重写一个简单版. """

    def __init__(
            self,
            root_tag: str,
            stream_id: str,
            callback: CommandTokenCallback,
            stop_event: ThreadSafeEvent,
            *,
            logger: Optional[logging.Logger] = None,
    ):
        """
        :param root_tag: do not send command token with root_tag
        :param stream_id: stream id to mark all the command token
        :param callback: callback function
        """
        self._stopped = False
        """自身的关机"""
        self._stop_event = stop_event
        """全局的关机"""

        self._root_tag = root_tag
        self._stream_id = stream_id
        # idx of the command token
        self._token_order = 0
        self._cmd_idx = 0
        # command token callback
        self._callback = callback
        # get the logger
        self._logger = logger or logging.getLogger("CTMLSaxHandler")
        # simple stack for unfinished element
        self._parsing_element_stack: List[CMTLElement] = []
        # event to notify the parsing is over.
        self.done_event = threading.Event()
        self._exception: Optional[Exception] = None

    def is_stopped(self) -> bool:
        return self._stopped or self._stop_event.is_set()

    def _send_to_callback(self, token: CommandToken | None) -> None:
        if token is None:
            # send the poison item means end
            self._callback(None)
        elif not self.done_event.is_set():
            token.order = self._token_order
            self._token_order += 1
            self._callback(token)
        else:
            # todo: log
            pass

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl) -> None:
        if self.is_stopped():
            raise ParserStopped
        dict_attrs = self.parse_attrs(attrs)

        element = CMTLElement(
            cmd_idx=self._cmd_idx,
            stream_id=self._stream_id,
            name=name,
            attrs=dict_attrs,
        )
        if len(self._parsing_element_stack) > 0:
            self._parsing_element_stack[-1].on_child_command()

        # using stack to handle elements
        self._parsing_element_stack.append(element)
        token = element.start_token()
        self._send_to_callback(token)
        self._cmd_idx += 1

    @classmethod
    def parse_attrs(cls, attrs: xml.sax.xmlreader.AttributesImpl) -> dict:
        return dict(attrs)

    def endElement(self, name: str):
        if self.is_stopped():
            raise ParserStopped
        if len(self._parsing_element_stack) == 0:
            raise ValueError("CTMLElement end element `%s` without existing one" % name)
        element = self._parsing_element_stack.pop(-1)
        token = element.end_token()
        self._send_to_callback(token)
        if len(self._parsing_element_stack) == 0:
            self.done_event.set()

    def characters(self, content: str):
        if self.is_stopped():
            raise ParserStopped

        if len(self._parsing_element_stack) == 0:
            # something goes wrong, on char without properly start or end
            raise ValueError("CTMLElement on character without properly start or end")
        element = self._parsing_element_stack[-1]
        token = element.add_delta(content)
        if token is not None:
            self._send_to_callback(token)

    def endDocument(self):
        self.done_event.set()

    def startDocument(self):
        pass

    def error(self, exception: Exception):
        self.done_event.set()
        self._logger.error(exception)
        if self._stop_event.is_set() or isinstance(exception, ParserStopped):
            # todo
            return
        self._exception = CommandTokenParseError(f"parse error: {exception}")

    def fatalError(self, exception: Exception):
        self.done_event.set()
        if self._stop_event.is_set() or isinstance(exception, ParserStopped):
            # todo
            return
        self._logger.exception(exception)
        self._exception = CommandTokenParseError(f"parse error: {exception}")

    def warning(self, exception):
        self._logger.warning(exception)

    def raise_error(self) -> None:
        if self._exception is not None:
            raise self._exception


class CTMLTokenParser(CommandTokenParser):
    """
    parsing input stream into Command Tokens
    """

    def __init__(
            self,
            callback: CommandTokenCallback | None = None,
            stream_id: str = "",
            *,
            root_tag: str = "ctml",
            stop_event: Optional[ThreadSafeEvent] = None,
            logger: Optional[logging.Logger] = None,
            special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.root_tag = root_tag
        self.logger = logger or logging.getLogger("moss")
        self.stop_event = stop_event or ThreadSafeEvent()
        self._callbacks = []
        if callback is not None:
            self._callbacks.append(callback)
        self._buffer = ""
        self._parsed: List[CommandToken] = []
        self._handler = CTMLSaxHandler(
            root_tag,
            stream_id,
            self._add_token,
            self.stop_event,
            logger=self.logger,
        )
        special_tokens = special_tokens or {}
        self._special_tokens_matcher = SpecialTokenMatcher(special_tokens)

        # lifecycle
        self._sax_parser = sax.make_parser()
        self._sax_parser.setContentHandler(self._handler)
        self._sax_parser.setErrorHandler(self._handler)

        self._stopped = False
        self._started = False
        self._committed = False
        self._sent_last_token = False

    def parsed(self) -> Iterable[CommandToken]:
        return self._parsed

    def with_callback(self, callback: CommandTokenCallback) -> None:
        self._callback = callback

    def _add_token(self, token: CommandToken | None) -> None:
        if token is not None:
            self._parsed.append(token)
        if len(self._callbacks) > 0:
            if token is None:
                if not self._sent_last_token:
                    self._sent_last_token = True
                else:
                    return
            for callback in self._callbacks:
                callback(token)

    def is_done(self) -> bool:
        return self._handler.done_event.is_set()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._sax_parser.feed(f'<{self.root_tag}>')

    def feed(self, delta: str) -> None:
        self._handler.raise_error()
        if self._stopped:
            raise ParserStopped()
        else:
            self._buffer += delta
            parsed = self._special_tokens_matcher.buffer(delta)
            self._sax_parser.feed(parsed)

    def commit(self) -> None:
        self._handler.raise_error()
        if self._committed:
            return
        self._committed = True
        last_buffer = self._special_tokens_matcher.clear()
        end_of_the_inputs = f'{last_buffer}</{self.root_tag}>'
        self._sax_parser.feed(end_of_the_inputs)

    def close(self) -> None:
        """
        stop the parser and clear the resources.
        """
        if self._stopped:
            return
        self._stopped = True
        self.commit()
        # self._handler.done_event.wait()
        self._sax_parser.close()
        # cancel
        self._add_token(None)

    def buffer(self) -> str:
        return self._buffer

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_val is not None:
            return None
        self._handler.raise_error()

    @classmethod
    def parse(
            cls,
            callback: CommandTokenCallback,
            stream: Iterable[str],
            *,
            root_tag: str = "ctml",
            stream_id: str = "",
            logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        simple example of parsing input stream into command token stream with a thread.
        but not a good practice
        """
        if isinstance(stream, str):
            stream = [stream]

        parser = cls(callback, stream_id, root_tag=root_tag, logger=logger)
        with parser:
            for element in stream:
                parser.feed(element)
            parser.commit()

    @classmethod
    def join_tokens(cls, tokens: Iterable[CommandToken]) -> str:
        # todo: 做优化能力, 比如将空的开标记合并.
        return ''.join([t.content for t in tokens])
