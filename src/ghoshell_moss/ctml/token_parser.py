import threading
from xml import sax

import logging
import xml.sax
from xml.sax import saxutils
from typing import List, Iterable, Optional, Callable, Dict, Set
from ghoshell_moss.concepts.command import CommandToken
from ghoshell_moss.concepts.interpreter import CommandTokenParser, CommandTokenParseError
from ghoshell_moss.concepts.errors import FatalError
from ghoshell_moss.helpers.token_filters import SpecialTokenMatcher
from ghoshell_moss.helpers.event import ThreadSafeEvent

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

    def __init__(
            self,
            root_tag: str,
            stream_id: str,
            callback: CommandTokenCallback,
            *,
            logger: Optional[logging.Logger] = None,
    ):
        """
        :param root_tag: do not send command token with root_tag
        :param stream_id: stream id to mark all the command token
        :param callback: callback function
        """
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

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesNSImpl) -> None:
        if self.done_event.is_set():
            return None
        dict_attrs = {}
        if len(attrs) > 0:
            for qname in attrs.getQNames():
                dict_attrs[qname] = attrs.getValueByQName(qname)

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
    def parse_attrs(cls, attrs: xml.sax.xmlreader.AttributesNSImpl) -> dict:
        dict_attrs = {}
        if len(attrs) > 0:
            for qname in attrs.getQNames():
                dict_attrs[qname] = attrs.getValueByQName(qname)
        return dict_attrs

    def endElement(self, name: str):
        if len(self._parsing_element_stack) == 0:
            raise FatalError("CTMLElement end element without existing one")
        element = self._parsing_element_stack.pop(-1)
        token = element.end_token()
        self._send_to_callback(token)

    def characters(self, content: str):
        if len(self._parsing_element_stack) == 0:
            # something goes wrong, on char without properly start or end
            raise FatalError("CTMLElement on character without properly start or end")
        element = self._parsing_element_stack[-1]
        token = element.add_delta(content)
        if token is not None:
            self._send_to_callback(token)

    def endDocument(self):
        # todo: log
        self._send_to_callback(None)
        self.done_event.set()

    def error(self, exception: Exception):
        if isinstance(exception, ParserStopped):
            return
        self._logger.exception(exception)

    def fatalError(self, exception: Exception):
        if isinstance(exception, ParserStopped):
            return
        self._logger.exception(exception)
        if self.done_event.is_set():
            return None
        self.done_event.set()
        # todo: wrap the exception
        raise CommandTokenParseError(f"CTML Parse Exception from sax: {exception}")

    def warning(self, exception):
        self._logger.warning(exception)


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
        self.logger = logger or logging.getLogger("CTMLParser")
        self._stop_event = stop_event or ThreadSafeEvent()
        self._callback = callback
        self._buffer = ""
        self._parsed: List[CommandToken] = []
        self._handler = CTMLSaxHandler(
            root_tag,
            stream_id,
            self._add_token,
            logger=logger,
        )
        self._sax_parser = xml.sax.make_parser()
        self._sax_parser.setContentHandler(self._handler)
        self._sax_parser.setErrorHandler(self._handler)
        self._stopped = False
        self._started = False
        self._ended = False
        special_tokens = special_tokens or {}
        self._special_tokens_matcher = SpecialTokenMatcher(special_tokens)

    def is_running(self) -> bool:
        return self._started and not self._stopped and not self._ended and not self._stop_event.is_set()

    def parsed(self) -> Iterable[CommandToken]:
        return self._parsed

    def with_callback(self, callback: CommandTokenCallback) -> None:
        self._callback = callback

    def _add_token(self, token: CommandToken | None) -> None:
        if token is not None:
            self._parsed.append(token)
        if not self._stopped and self._callback is not None:
            self._callback(token)

    def is_done(self) -> bool:
        return self._handler.done_event.is_set() or self._stopped

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._sax_parser.feed(f'<{self.root_tag}>')

    def feed(self, delta: str) -> None:
        if self.is_running():
            self._buffer += delta
            parsed = self._special_tokens_matcher.buffer(delta)
            self._sax_parser.feed(parsed)
        else:
            raise ParserStopped()

    def end(self) -> None:
        if not self.is_running():
            return
        if self._ended:
            return
        self._ended = True
        last_buffer = self._special_tokens_matcher.clear()
        end_of_the_inputs = f'{last_buffer}</{self.root_tag}>'
        self._sax_parser.feed(end_of_the_inputs)

    def stop(self) -> None:
        """
        stop the parser and clear the resources.
        """
        if self._stopped:
            return
        self._stopped = True
        # cancel
        self._add_token(None)
        self._handler.done_event.set()
        self._sax_parser.close()

    def buffer(self) -> str:
        return self._buffer

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            self.end()
        self.stop()

    @classmethod
    def parse(
            cls,
            callback: CommandTokenCallback,
            stream: Iterable[str],
            *,
            root_tag: str = "ctml",
            stream_id: str = "",
            default_chan: str = "",
            logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        simple example of parsing input stream into command token stream with a thread.
        but not a good practice
        """
        if isinstance(stream, str):
            stream = [stream]

        parser = cls(callback, stream_id, root_tag=root_tag, default_chan=default_chan, logger=logger)
        with parser:
            for element in stream:
                parser.feed(element)
            parser.end()

    @classmethod
    def join_tokens(cls, tokens: Iterable[CommandToken]) -> str:
        # todo: 做优化能力, 比如将空的开标记合并.
        return ''.join([t.content for t in tokens])
