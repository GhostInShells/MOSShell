import threading
from xml import sax

import logging
import xml.sax
from xml.sax import saxutils
from typing import List, Iterable, Optional, Callable
from ghoshell_moss.concepts.command import CommandToken
from ghoshell_moss.concepts.interpreter import CommandTokenParser, CommandTokenParseError
from ghoshell_moss.concepts.errors import FatalError

CommandTokenCallback = Callable[[CommandToken], None]


class CMTLElement:
    """
    Utility class to generate Command Token in XMTL (Command Token Marked Language stream)
    """

    def __init__(self, *, cmd_idx: int, stream_id: str, chan: str, name: str, attrs: dict):
        self.cmd_idx = cmd_idx
        self.ns = chan
        self.name = name
        self.deltas = ""
        # first part idx is 1
        self.part_idx = 1
        self._has_delta = False
        self.fullname = self.make_fullname(chan, name)
        self.attrs = attrs
        self.stream_id = stream_id

    @classmethod
    def make_fullname(cls, ns: str, name: str) -> str:
        return f"{ns}:{name}" if ns else name

    def start_token(self) -> CommandToken:
        attr_expression = []
        for k, v in self.attrs.items():
            quoted_value = saxutils.quoteattr(v)
            attr_expression.append(f"{k}={quoted_value}")
        exp = " " if len(attr_expression) > 0 else ""
        content = f"<{self.fullname}{exp}" + " ".join(attr_expression) + ">"
        part_idx = self.part_idx
        self.part_idx += 1
        return CommandToken(
            name=self.name,
            chan=self.ns,
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
            chan=self.ns,
            cmd_idx=self.cmd_idx,
            part_idx=self.part_idx,
            stream_id=self.stream_id,
            type="end",
            kwargs=None,
            content=f"</{self.fullname}>",
        )


class ParserStopped(Exception):
    pass


class CTMLSaxHandler(xml.sax.ContentHandler, xml.sax.ErrorHandler):

    def __init__(
            self,
            root_tag: str,
            stream_id: str,
            callback: CommandTokenCallback,
            *,
            default_chan: str = "",
            logger: Optional[logging.Logger] = None,
    ):
        """
        :param root_tag: do not send command token with root_tag
        :param stream_id: stream id to mark all the command token
        :param callback: callback function
        :param default_chan: default channel name

        """
        self._root_tag = root_tag
        self._stream_id = stream_id
        self._default_chan = default_chan
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

    def _send_to_callback(self, token: CommandToken) -> None:
        if not self.done_event.is_set():
            self._token_order += 1
            token.order = self._token_order
            self._callback(token)
        else:
            # todo: log
            pass

    def startElementNS(self, name: tuple[str, str], qname: str, attrs: xml.sax.xmlreader.AttributesNSImpl) -> None:
        if self.done_event.is_set():
            return None
        cmd_chan, cmd_name = name
        dict_attrs = {}
        if len(attrs) > 0:
            for qname in attrs.getQNames():
                dict_attrs[qname] = attrs.getValueByQName(qname)

        element = CMTLElement(
            cmd_idx=self._cmd_idx,
            stream_id=self._stream_id,
            chan=cmd_chan or self._default_chan,
            name=cmd_name,
            attrs=dict_attrs,
        )
        if len(self._parsing_element_stack) > 0:
            self._parsing_element_stack[-1].on_child_command()

        # using stack to handle elements
        self._parsing_element_stack.append(element)
        if element.fullname != self._root_tag:
            token = element.start_token()
            self._send_to_callback(token)
        self._cmd_idx += 1

    def endElementNS(self, name: tuple[str, str], qname):
        if len(self._parsing_element_stack) == 0:
            raise FatalError("CTMLElement end element without existing one")
        element = self._parsing_element_stack.pop(-1)
        if element.fullname != self._root_tag:
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
        self.done_event.set()

    def error(self, exception: Exception):
        self._logger.exception(exception)

    def fatalError(self, exception: Exception):
        self._logger.exception(exception)
        if self.done_event.is_set():
            return None
        self.done_event.set()
        # todo: wrap the exception
        raise CommandTokenParseError(f"CTML Parse Exception from sax: {exception}")

    def warning(self, exception):
        self._logger.warning(exception)


class CTMLParser(CommandTokenParser):
    """
    parsing input stream into Command Tokens
    """

    def __init__(
            self,
            callback: CommandTokenCallback | None = None,
            stream_id: str = "",
            *,
            root_tag: str = "cmtl",
            default_chan: str = "",
            logger: Optional[logging.Logger] = None,
    ):
        self.root_tag = root_tag
        self.logger = logger or logging.getLogger("CTMLParser")
        if callback is None:
            value = []
            callback = value.append
        self._callback = callback
        self._buffer = ""
        self._parsed: List[CommandToken] = []
        self._handler = CTMLSaxHandler(
            root_tag,
            stream_id,
            callback,
            default_chan=default_chan,
            logger=logger,
        )
        self._sax_parser = xml.sax.make_parser()
        self._sax_parser.setFeature(xml.sax.handler.feature_namespaces, 1)
        self._sax_parser.setContentHandler(self._handler)
        self._sax_parser.setErrorHandler(self._handler)
        self._stopped = False
        self._started = False
        self._ended = False

    def is_running(self) -> bool:
        return self._started and not self._stopped and not self._ended

    def with_callback(self, callback: CommandTokenCallback) -> None:
        self._callback = callback

    def is_done(self) -> bool:
        return self._handler.done_event.is_set()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._sax_parser.feed(f'<{self.root_tag}>')

    def feed(self, delta: str) -> None:
        if self._stopped:
            raise ParserStopped()
        elif self.is_running():
            self._buffer += delta
            self._sax_parser.feed(delta)
        else:
            return

    def end(self) -> None:
        if not self.is_running():
            return
        if self._ended:
            return
        self._ended = True
        self._sax_parser.feed(f'</{self.root_tag}>')

    def stop(self) -> None:
        """
        stop the parser and clear the resources.
        """
        if self._stopped:
            return
        self._stopped = True
        # cancel
        self._handler.done_event.set()
        self._sax_parser.close()

    def buffer(self) -> str:
        return self._buffer

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
