import logging
import threading
import xml.sax
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, Iterable, Protocol
from xml import sax
from xml.sax import saxutils

from ghoshell_moss.core.concepts.command import CommandToken
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.concepts.interpreter import CommandTokenParser
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.core.helpers.token_filters import SpecialTokenMatcher
from ast import literal_eval

CommandTokenCallback = Callable[[CommandToken | None], None]

__all__ = [
    "CMTLSaxElement",
    "CTMLSaxHandler",
    "ParserStopped",
    "AttrParser",
    "AttrPrefixParser",
    "CTMLTokenParser",
    "literal_parser",
]


class CMTLSaxElement:
    """
    Utility class to generate Command Token in XMTL (Command Token Marked Language stream)
    """

    def __init__(
            self,
            *,
            cmd_idx: int,
            stream_id: str,
            chan: str,
            name: str,
            attrs: dict[str, str],
            parsed: dict[str, Any] | None = None,
            call_id: int | None = None,
    ):
        self.cmd_idx = cmd_idx
        self.call_id = call_id
        self.name = name
        self.chan = chan or ""
        self.deltas = ""
        # first part idx is 0
        self.part_idx = 0
        self._has_delta = False
        self.attrs = attrs
        self.parsed = parsed
        self.stream_id = stream_id

    @classmethod
    def make_fullname(cls, chan: Optional[str], name: str, call_id: Optional[int] = None) -> str:
        parts = []
        if chan:
            parts.append(chan)
        parts.append(name)
        if call_id is not None:
            parts.append(str(call_id))
        return ":".join(parts)

    @classmethod
    def make_start_mark(cls, chan: str, name: str, attrs: dict, self_close: bool, call_id: Optional[int] = None) -> str:
        attr_expression = []
        for k, v in attrs.items():
            quoted_value = saxutils.quoteattr(str(v))
            attr_expression.append(f"{k}={quoted_value}")
        exp = " " if len(attr_expression) > 0 else ""
        self_close_mark = "/" if self_close else ""
        fullname = cls.make_fullname(chan, name, call_id)
        content = f"<{fullname}{exp}" + " ".join(attr_expression) + self_close_mark + ">"
        return content

    @classmethod
    def make_end_mark(cls, chan: Optional[str], name: str, call_id: Optional[int] = None) -> str:
        return f"</{cls.make_fullname(chan, name, call_id)}>"

    def start_token(self) -> CommandToken:
        content = self.make_start_mark(self.chan, self.name, self.attrs, self_close=False, call_id=self.call_id)
        part_idx = self.part_idx
        self.part_idx += 1
        return CommandToken(
            name=self.name,
            chan=self.chan,
            cmd_idx=self.cmd_idx,
            part_idx=part_idx,
            stream_id=self.stream_id,
            call_id=self.call_id,
            seq="start",
            kwargs=self.parsed or self.attrs,
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
                chan=self.chan,
                cmd_idx=self.cmd_idx,
                part_idx=self.part_idx,
                stream_id=self.stream_id,
                call_id=self.call_id,
                seq="delta",
                kwargs=None,
                content=delta,
            )
        return None

    def end_token(self) -> CommandToken:
        if self._has_delta:
            self.part_idx += 1
        return CommandToken(
            name=self.name,
            chan=self.chan,
            cmd_idx=self.cmd_idx,
            call_id=self.call_id,
            part_idx=self.part_idx,
            stream_id=self.stream_id,
            seq="end",
            kwargs=None,
            content=CMTLSaxElement.make_end_mark(self.chan, self.name, call_id=self.call_id),
        )


class ParserStopped(Exception):
    """notify the sax that parsing is stopped"""

    pass


SpecialAttrParser = Callable[[str, str], Optional[tuple[str, Any]]]


class AttrParser(Protocol):
    description: str

    @abstractmethod
    def parse(self, name: str, value: str) -> Optional[tuple[str, Any]]:
        pass


class AttrPrefixParser(AttrParser):

    def __init__(
            self,
            desc: str,
            prefix: str,
            parser: Callable[[str], Any],
    ):
        self.description = desc
        self._prefix = prefix
        self._parser = parser

    def parse(self, name: str, value: str) -> Optional[tuple[str, Any]]:
        if not name.startswith(self._prefix):
            return None
        attr_name = name[len(self._prefix):]
        try:
            parsed = self._parser(value)
            return attr_name, parsed
        except (ValueError, SyntaxError):
            return None


literal_parser = AttrPrefixParser(
    desc="凡是用 literal- 开头的参数, 都会执行 ast.literal_eval 解析值.",
    prefix="literal-",
    parser=lambda v: literal_eval(v),
)


class CTMLSaxHandler(xml.sax.ContentHandler, xml.sax.ErrorHandler):
    """初步实现 sax 解析. 实现得非常糟糕, 主要是对 sax 的回调机制有误解, 留下了大量冗余状态. 需要考虑重写一个简单版."""

    def __init__(
            self,
            root_tag: str,
            stream_id: str,
            callback: CommandTokenCallback,
            stop_event: ThreadSafeEvent,
            *,
            attr_parsers: list[AttrParser] | None = None,
            logger: Optional[logging.Logger] = None,
            ensure_call_id: bool = False,
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
        self._attr_parsers = attr_parsers or []
        self._ensure_call_id = ensure_call_id

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
        self._parsing_element_stack: list[CMTLSaxElement] = []
        self._attr_parsers = attr_parsers or []
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

    def startElementNS(self, name: tuple[str, str], qname: str, attrs: xml.sax.xmlreader.AttributesNSImpl):
        if self.is_stopped():
            raise ParserStopped
        chan, command_name = name
        dict_attrs = {}
        for attr_qname in attrs.getQNames():
            _, name = attrs.getNameByQName(attr_qname)
            attr_value = attrs.getValueByQName(attr_qname)
            dict_attrs[name] = attr_value
        self._start_command_token_element(chan, command_name, dict_attrs)

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl | dict) -> None:
        if self.is_stopped():
            raise ParserStopped
        parts = name.split(":", 2)
        call_id = None
        if len(parts) == 1:
            chan = ""
            command_name = parts[0]
        elif len(parts) == 2:
            chan, command_name = parts
        elif len(parts) == 3:
            chan, command_name, call_id = parts
        else:
            chan = ""
            command_name = parts[0]
        try:
            if call_id is not None:
                call_id = int(call_id)
        except ValueError:
            call_id = None
        dict_attrs, parsed = self.parse_attrs(attrs)
        self._start_command_token_element(chan, command_name, dict_attrs, parsed_attrs=parsed, call_id=call_id)

    def _start_command_token_element(
            self,
            chan: str,
            name: str,
            attrs: dict,
            *,
            parsed_attrs: dict | None = None,
            call_id: Optional[int] = None,
    ) -> None:
        if call_id is None and self._ensure_call_id:
            call_id = self._cmd_idx
        element = CMTLSaxElement(
            cmd_idx=self._cmd_idx,
            stream_id=self._stream_id,
            name=name,
            chan=chan,
            attrs=attrs,
            parsed=parsed_attrs,
            call_id=call_id,
        )
        if len(self._parsing_element_stack) > 0:
            self._parsing_element_stack[-1].on_child_command()

        # using stack to handle elements
        self._parsing_element_stack.append(element)
        token = element.start_token()
        self._send_to_callback(token)
        self._cmd_idx += 1

    def parse_attrs(
            self,
            attrs: xml.sax.xmlreader.AttributesImpl | dict,
    ) -> tuple[dict[str, str], dict[str, Any] | None]:
        values = dict(attrs)
        if len(self._attr_parsers) == 0:
            return values, None
        result = {}
        for name, value in values.items():
            key, val = self._parse_attr(name, value)
            result[key] = val
        return values, result

    def _parse_attr(self, name: str, value: str) -> tuple[str, Any]:
        for parser in self._attr_parsers:
            got = parser.parse(name, value)
            if got is not None:
                new_name, new_value = got
                return new_name, new_value
        return name, value

    def endElement(self, name: str):
        if self.is_stopped():
            raise ParserStopped
        if len(self._parsing_element_stack) == 0:
            raise ValueError(f"CTMLElement end element `{name}` without existing one")
        element = self._parsing_element_stack.pop(-1)
        token = element.end_token()
        self._send_to_callback(token)
        if len(self._parsing_element_stack) == 0:
            self.done_event.set()

    def endElementNS(self, name: tuple[str, str], qname: str):
        self.endElement(qname)

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
        self._exception = InterpretError(f"parse error: {exception}")

    def fatalError(self, exception: Exception):
        self.done_event.set()
        if self._stop_event.is_set() or isinstance(exception, ParserStopped):
            # todo
            return
        self._logger.exception(exception)
        self._exception = InterpretError(f"parse error: {exception}")

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
            special_tokens: Optional[dict[str, str]] = None,
            attr_parsers: list[AttrParser] | None = None,
            with_call_id: bool = False,
    ):
        self.root_tag = root_tag
        self.logger = logger or logging.getLogger("moss")
        self.stop_event = stop_event or ThreadSafeEvent()
        self._callbacks = []
        if callback is not None:
            self._callbacks.append(callback)
        self._buffer = ""
        self._parsed: list[CommandToken] = []
        self._handler = CTMLSaxHandler(
            root_tag,
            stream_id,
            self._add_token,
            self.stop_event,
            logger=self.logger,
            attr_parsers=attr_parsers,
            ensure_call_id=with_call_id,
        )
        special_tokens = special_tokens or {}
        self._special_tokens_matcher = SpecialTokenMatcher(special_tokens)

        # lifecycle
        self._sax_parser = sax.make_parser()
        self._sax_parser.setFeature(sax.handler.feature_namespaces, False)
        self._sax_parser.setFeature(sax.handler.feature_namespace_prefixes, False)

        self._sax_parser.setContentHandler(self._handler)
        self._sax_parser.setErrorHandler(self._handler)

        self._stopped = False
        self._started = False
        self._committed = False
        self._sent_last_token = False

    def parsed(self) -> Iterable[CommandToken]:
        return self._parsed

    def with_callback(self, *callbacks: CommandTokenCallback) -> None:
        callbacks = list(callbacks)
        callbacks.extend(self._callbacks)
        self._callbacks = callbacks

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
        self._sax_parser.feed(f"<{self.root_tag}>")

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
        end_of_the_inputs = f"{last_buffer}</{self.root_tag}>"
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
        try:
            self._sax_parser.close()
        except xml.parsers.expat.ExpatError:
            pass
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
            attr_parsers: Optional[list[AttrParser]] = None,
            with_call_id: bool = False,
    ) -> None:
        """
        simple example of parsing input stream into command token stream with a thread.
        but not a good practice
        """
        if isinstance(stream, str):
            stream = [stream]

        parser = cls(
            callback,
            stream_id,
            root_tag=root_tag,
            logger=logger,
            attr_parsers=attr_parsers,
            with_call_id=with_call_id,
        )
        with parser:
            for element in stream:
                parser.feed(element)
            parser.commit()

    @classmethod
    def join_tokens(cls, tokens: Iterable[CommandToken]) -> str:
        # todo: 做优化能力, 比如将空的开标记合并.
        return "".join([t.content for t in tokens])
