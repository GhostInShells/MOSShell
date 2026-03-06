import logging
import threading
import xml.sax
from abc import abstractmethod
from typing import Optional, Any, Callable, Iterable, Protocol
from xml import sax
from xml.sax import saxutils

from ghoshell_moss.core.concepts.command import CommandToken
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.concepts.interpreter import TextTokenParser
from ghoshell_moss.core.helpers.token_filters import TokensReplacementMatcher
from ghoshell_common.helpers import Timeleft
from ast import literal_eval

CommandTokenCallback = Callable[[CommandToken | None], None]

__all__ = [
    "CMTLSaxElement",
    "CTMLSaxHandler",
    "ParserStopped",
    "AttrParser",
    "AttrPrefixParser",
    "AttrWithTypeSuffixParser",
    "CTML2CommandTokenParser",
    "ctml_default_parsers",
]

_POSITION_ARGS_KEY = "_args"


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
        parsed_args: list[str] | None = None,
        parsed_kwargs: dict[str, Any] | None = None,
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
        self.parsed_args = parsed_args
        self.parsed_kwargs = parsed_kwargs
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
            args=self.parsed_args or [],
            kwargs=self.parsed_kwargs if self.parsed_kwargs is not None else self.attrs,
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


class AttrWithTypeSuffixParser(AttrParser):
    def __init__(
        self,
        description: str = "允许属性跟随后缀, 形如 a:str",
        parser_map: dict[str, Callable[[str], Any]] | None = None,
    ):
        self.description = description
        self._parser_map = parser_map or {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": lambda v: list(literal_eval(v)),
            "dict": lambda v: dict(literal_eval(v)),
            "None": lambda v: None,
            "none": lambda v: None,
            "literal": literal_eval,
            "lambda": lambda v: eval(f"lambda: {v}")(),
        }

    def parse(self, name: str, value: str) -> Optional[tuple[str, Any]]:
        parts = name.split(":", 1)
        if len(parts) == 1:
            return None
        key = parts[0]
        type_name = parts[1]
        if type_name not in self._parser_map:
            return None
        parser = self._parser_map[type_name]
        try:
            return key, parser(value)
        except (TypeError, ValueError):
            # 无法解析的情况.
            return None


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
        attr_name = name[len(self._prefix) :]
        try:
            parsed = self._parser(value)
            return attr_name, parsed
        except (ValueError, SyntaxError):
            return None


ctml_default_parsers = [
    AttrWithTypeSuffixParser(
        description="允许属性跟随后缀, 形如 a:str",
    ),
]


def get_error_context(xml_string, exception, window=20):
    """
    xml_string: 原始 XML 字符串
    exception: 捕获到的 SAXParseException
    window: 错误位置前后截取的字符长度
    """
    lines = xml_string.splitlines()
    line_no = exception.getLineNumber() - 1  # 索引从 0 开始
    col_no = exception.getColumnNumber() - 1

    if line_no < len(lines):
        error_line = lines[line_no]
        # 截取错误位置附近的内容，方便肉眼定位
        start = max(0, col_no - window)
        end = min(len(error_line), col_no + window)
        context = error_line[start:end]
        marker = " " * (col_no - start) + "^"
        return f"Line {line_no + 1}, Col {col_no + 1}:\n{context}\n{marker}"
    return "Unknown location"


class CTMLSaxHandler(xml.sax.ContentHandler, xml.sax.ErrorHandler):
    """初步实现 sax 解析. 实现得非常糟糕, 主要是对 sax 的回调机制有误解, 留下了大量冗余状态. 需要考虑重写一个简单版."""

    def __init__(
        self,
        root_tag: str,
        stream_id: str,
        callback: CommandTokenCallback,
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
        self._attr_parsers = attr_parsers or ctml_default_parsers
        self._ensure_call_id = ensure_call_id

        self._root_tag = root_tag
        self._stream_id = stream_id
        # idx of the command token
        self._token_order = 0
        self._cmd_idx = 0
        # command token callback
        self._callback = callback
        # get the logger
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = f"[{self.__class__.__name__}][{self._root_tag}]"
        # simple stack for unfinished element
        self._parsing_element_stack: list[CMTLSaxElement] = []
        self._attr_parsers = attr_parsers or []
        # event to notify the parsing is over.
        self.done_event = threading.Event()
        self._exception: Optional[Exception] = None
        self._parsing_text = ""

    def buffer_input(self, text: str):
        """
        方便发生异常时可以定位错误在哪里.
        """
        self._parsing_text += text

    def is_stopped(self) -> bool:
        return self._stopped

    def _send_to_callback(self, token: CommandToken | None) -> None:
        if token is None:
            # send the poison item means end
            self._callback(None)
        else:
            token.order = self._token_order
            self._token_order += 1
            self._callback(token)

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
        args, dict_attrs, parsed_kwargs = self.parse_attrs(attrs)
        self._start_command_token_element(
            chan,
            command_name,
            dict_attrs,
            parsed_args=args,
            parsed_kwargs=parsed_kwargs,
            call_id=call_id,
        )

    def _start_command_token_element(
        self,
        chan: str,
        name: str,
        attrs: dict,
        *,
        parsed_args: list | None = None,
        parsed_kwargs: dict | None = None,
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
            parsed_args=parsed_args,
            parsed_kwargs=parsed_kwargs,
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
    ) -> tuple[list[Any], dict[str, str], dict[str, Any]]:
        origin_attrs = dict(attrs)
        dict_attrs = origin_attrs.copy()
        if _POSITION_ARGS_KEY in dict_attrs:
            value = dict_attrs.pop(_POSITION_ARGS_KEY)

            try:
                args = literal_eval(value)
            except ValueError as e:
                self._logger.error(
                    "%s receive position args value error: %s, %s",
                    self._log_prefix,
                    e,
                    origin_attrs,
                )
                raise InterpretError(
                    f"Invalid position args: {value}. {_POSITION_ARGS_KEY} must be python literal list",
                )
            if isinstance(args, tuple) or isinstance(args, set):
                args = list(args)
        else:
            args = []
        if not isinstance(args, list):
            self._logger.error(
                "%s receive position args can not parsed to list: %s",
                self._log_prefix,
                origin_attrs,
            )
            raise InterpretError(
                f"Invalid position args: {args}. {_POSITION_ARGS_KEY} must be python literal list",
            )

        if len(self._attr_parsers) == 0:
            return args, origin_attrs, dict_attrs
        result = {}
        for name, value in dict_attrs.items():
            if name == _POSITION_ARGS_KEY:
                continue
            key, val = self._parse_attr(name, value)
            result[key] = val
        return args, origin_attrs, result

    def _parse_attr(self, name: str, value: str) -> tuple[str, Any]:
        for parser in self._attr_parsers:
            got = parser.parse(name, value)
            if got is not None:
                new_name, new_value = got
                return new_name, new_value

        try:
            _value = literal_eval(value)
            value = _value
        except (SyntaxError, TypeError, ValueError):
            pass

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
        if self.done_event.is_set():
            return
        self.done_event.set()
        self._logger.error(exception)
        if isinstance(exception, xml.sax.SAXParseException):
            exp_str = get_error_context(self._parsing_text, exception)
        else:
            exp_str = str(exception)
        self._exception = InterpretError(f"CTML parse fatal error: {exp_str}. Check CDATA and open-close tag rules")

    def fatalError(self, exception: Exception):
        if self.done_event.is_set():
            return
        self.done_event.set()
        self._logger.exception(exception)
        if isinstance(exception, xml.sax.SAXParseException):
            exp_str = get_error_context(self._parsing_text, exception)
        else:
            exp_str = str(exception)
        self._exception = InterpretError(f"CTML parse fatal error: {exp_str}. Check CDATA and open-close tag rules")

    def warning(self, exception):
        self._logger.warning(exception)

    def raise_error(self) -> None:
        if self._exception is not None:
            raise self._exception


class CTML2CommandTokenParser(TextTokenParser):
    """
    parsing input stream into Command Tokens
    实现这个设计时, 正在从 Python 多线程思维向 Async 思维转向, 两种风格在打架.
    这一版未来需要彻底重做. 但基本的 feature 不变.

    目前的用法过于复杂:
    >>> def run_parser(parser: CTML2CommandTokenParser, tokens: Iterable[str], callback: CommandTokenCallback) -> None:
    >>>     with parser.with_callback(callback):
    >>>          for token in tokens:
    >>>             parser.feed(token)
    >>>          parser.commit()
    >>>          parser.wait_done()

    在一个线程里完成回调.
    目前主要的问题是, 这个 Parser 从上游拿到退出通知, 导致全生命周期耦合. 还是 golang 的 ctx 思路.
    它既然被管控, 应该完全被上层控制, 不要理解上层.
    python 应该通过 with statement 正确的解决一切生命周期问题. 而不是通过特别复杂的链路讯号.
    """

    def __init__(
        self,
        callback: CommandTokenCallback | None = None,
        stream_id: str = "",
        *,
        root_tag: str = "ctml",
        logger: Optional[logging.Logger] = None,
        tokens_replacement: Optional[dict[str, str]] = None,
        attr_parsers: list[AttrParser] | None = None,
        with_call_id: bool = False,
    ):
        self.root_tag = root_tag
        self.logger = logger or logging.getLogger("moss")
        self._log_prefix = f"[{self.__class__.__name__}][{self.root_tag}]"
        self._callbacks = []
        if callback is not None:
            self._callbacks.append(callback)
        self._buffer = ""
        self._parsed: list[CommandToken] = []
        self._handler = CTMLSaxHandler(
            root_tag,
            stream_id,
            self._deliver_token,
            logger=self.logger,
            attr_parsers=attr_parsers or [],
            ensure_call_id=with_call_id,
        )
        tokens_replacement = tokens_replacement or {}
        self._tokens_replacement_matcher = TokensReplacementMatcher(tokens_replacement)

        # lifecycle
        self._sax_parser = None
        self._stopped = False
        self._closed = False
        self._started = False
        self._committed = False
        self._last_token_delivered = False

    def parsed(self) -> Iterable[CommandToken]:
        return self._parsed

    def stop(self) -> None:
        self.logger.error(f"%s stop by outside", self._log_prefix)
        self._stopped = True

    def with_callback(self, *callbacks: CommandTokenCallback) -> None:
        callbacks = list(callbacks)
        callbacks.extend(self._callbacks)
        self._callbacks = callbacks

    def wait_done(self) -> None:
        if self.is_running():
            self._handler.done_event.wait()

    def _deliver_token(self, token: CommandToken | None) -> None:
        if token is not None:
            if self._last_token_delivered:
                self.logger.error(f"%s Delivered token %s is already delivered", self._log_prefix, token)
                return
            self._parsed.append(token)
        if self._stopped:
            # 不发送任何信息
            return
        if len(self._callbacks) > 0:
            if token is None:
                if not self._last_token_delivered:
                    self._last_token_delivered = True
                else:
                    return
            for callback in self._callbacks:
                try:
                    callback(token)
                except Exception as e:
                    self.logger.exception("%s deliver token failed %s", self._log_prefix, e)

    def is_done(self) -> bool:
        return self._sax_parser is not None and self._handler.done_event.is_set()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError(f"CTML2CommandTokenParser is already stopped ")
        if self._started:
            return
        self._started = True
        self._sax_parser = sax.make_parser()
        self._sax_parser.setFeature(sax.handler.feature_namespaces, False)
        self._sax_parser.setFeature(sax.handler.feature_namespace_prefixes, False)
        self._sax_parser.setContentHandler(self._handler)
        self._sax_parser.setErrorHandler(self._handler)
        self._sax_parser.feed(f"<{self.root_tag}>")

    def is_running(self) -> bool:
        return self._started and not self._closed and self._sax_parser is not None

    def _check_running(self):
        if not self._started:
            raise RuntimeError(f"CTML2CommandTokenParser is not started yet")
        if not self.is_running():
            raise ParserStopped()
        if self._handler:
            self._handler.raise_error()

    def feed(self, delta: str) -> None:
        self._check_running()
        self._buffer += delta
        parsed = self._tokens_replacement_matcher.buffer(delta)
        self._handler.buffer_input(delta)
        self._sax_parser.feed(parsed)

    def commit(self) -> None:
        if self._committed:
            # 只执行一次.
            return
        self._committed = True
        # 正常退出时, 需要发送消息.
        if not self._handler.done_event.is_set():
            # 获取未完成的粘包.
            last_buffer = self._tokens_replacement_matcher.clear()
            self._buffer += last_buffer
            # 发送尾包.
            end_of_the_inputs = f"{last_buffer}</{self.root_tag}>"
            self._sax_parser.feed(end_of_the_inputs)

    def close(self) -> None:
        """
        stop the parser and clear the resources.
        """
        if self._closed:
            # 可重入.
            return
        if not self._started:
            return
        self._closed = True
        # 通知下游结束.
        self.commit()
        # 退出后也设置自身结束.
        self._handler.done_event.set()
        try:
            # 关闭 parser.
            self._sax_parser.close()
        except xml.parsers.expat.ExpatError as e:
            self.logger.exception("close sax parser failed: %s", e)
            pass
        self._deliver_token(None)

    def buffered(self) -> str:
        return self._buffer

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 确保退出.
        self.close()
        if exc_val is not None:
            if isinstance(exc_val, ParserStopped):
                # ParserStopped 中断自身循环. 不用对外抛出.
                return True
            self.logger.exception("%s exception during context manager: %s", self._log_prefix, exc_val)
            return None
        if not self._stopped:
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
