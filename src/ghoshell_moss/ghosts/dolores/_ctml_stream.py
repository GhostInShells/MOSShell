"""Streaming extractor for the streaming CTML tools' (``moss_interpret`` / ``moss_react``) single string argument.

The tool's arguments are the JSON object ``{"ctml": "<the CTML>"}``. dsh streams the argument text
token by token (``tool-call-delta.argumentsDelta``); this object turns that raw JSON fragment stream
back into the CTML character stream, so the CTML reaches the articulator while the model is still
generating — the multi-channel streaming syntax (``chunks__``/``text__``) depends on it.

Two halves, both following the JSON grammar:

- a **prefix scanner** (JSON-token granularity, whitespace-tolerant) walking ``{`` → ``"ctml"`` →
  ``:`` → the opening ``"`` of the value. The prefix is predictable, so it can be matched as it
  arrives; a mismatch flips :attr:`matched` and the caller falls back to the full ``tool/call`` event.
- a **string-body unescaper** reversing RFC 8259 escapes (``\\"`` … ``\\t``, ``\\uXXXX``, surrogate
  pairs). The closing quote needs no prediction — an unescaped ``"`` simply ends the body, and the
  trailing ``}`` is ignored. A partial token (``\\``, ``\\uD8``, a lone surrogate) may split across
  any chunk boundary, because the whole state persists between :meth:`add` calls. A malformed escape
  flips :attr:`failed` — the argument stream is corrupt, so the caller aborts instead of falling back.
"""

from __future__ import annotations

__all__ = ["CtmlArgumentStream"]

# RFC 8259 §7 — the eight single-character escapes.
_ESCAPES = {
    '"': '"',
    '\\': '\\',
    '/': '/',
    'b': '\b',
    'f': '\f',
    'n': '\n',
    'r': '\r',
    't': '\t',
}
_HEX = frozenset('0123456789abcdefABCDEF')

# stages
_PREFIX = 'prefix'   # matching {"ctml": up to the value's opening quote
_BODY = 'body'       # inside the value string
_ESCAPE = 'escape'   # saw a backslash, awaiting the escape char
_UNICODE = 'unicode'  # saw \u, collecting 4 hex digits
_DONE = 'done'       # the value's closing quote was read
_FAILED = 'failed'   # the arguments are not {"ctml": "…"} — stop consuming


class CtmlArgumentStream:
    """Feed ``argumentsDelta`` fragments in, get the CTML characters out.

    Usage::

        stream = CtmlArgumentStream()
        for delta in arguments_deltas:
            ctml_delta = stream.add(delta)   # '' until the value's opening quote arrives
            ...                              # forward ctml_delta to the articulator
        stream.value                         # the whole CTML, for reconciliation

    Not thread-safe: it is driven by one consumer (the run's event loop).
    """

    def __init__(self, key: str = 'ctml') -> None:
        # prefix tokens of {"ctml": "…"} — whitespace is allowed between tokens, not inside them.
        self._tokens: tuple[str, ...] = ('{', f'"{key}"', ':', '"')
        self._token = 0
        self._pos = 0
        self._stage = _PREFIX
        # unescape state, all of it surviving chunk boundaries.
        self._hex = ''
        self._high: int | None = None  # pending high surrogate code unit
        self._value = ''
        self._matched = True
        self._failed = False

    # ── public surface ───────────────────────────────────────

    @property
    def value(self) -> str:
        """The CTML decoded so far (for reconciling against the full ``tool/call`` arguments)."""
        return self._value

    @property
    def complete(self) -> bool:
        """The value's closing quote has been read — the whole CTML is in."""
        return self._stage == _DONE

    @property
    def matched(self) -> bool:
        """False once the arguments stopped looking like ``{"ctml": "…"}`` — fall back to ``tool/call``."""
        return self._matched

    @property
    def failed(self) -> bool:
        """True once the argument stream was corrupt (a malformed escape) — abort, don't fall back."""
        return self._failed

    def add(self, delta: str | None) -> str:
        """Feed one raw ``argumentsDelta``; return the CTML characters it yielded (``''`` if none)."""
        if not delta:
            return ''
        text = ''.join(filter(None, (self._feed(char) for char in delta)))
        if text:
            self._value += text
        return text

    # ── the state machine ────────────────────────────────────

    def _feed(self, char: str) -> str:
        """Consume one character; return what it emitted (``''`` for structural input)."""
        stage = self._stage
        if stage == _BODY:
            return self._feed_body(char)
        if stage == _ESCAPE:
            return self._feed_escape(char)
        if stage == _UNICODE:
            return self._feed_unicode(char)
        if stage == _PREFIX:
            self._feed_prefix(char)
        return ''

    def _feed_prefix(self, char: str) -> None:
        """Match the ``{"ctml": "`` prefix at JSON-token granularity; enter _BODY at the opening quote."""
        token = self._tokens[self._token]
        if self._pos == 0 and char.isspace():
            return  # whitespace between tokens is insignificant in JSON
        if self._pos < len(token) and char == token[self._pos]:
            self._pos += 1
            if self._pos == len(token):
                self._token += 1
                self._pos = 0
                if self._token == len(self._tokens):
                    self._stage = _BODY
            return
        # a different shape (another key, a non-string value, …): give up on streaming.
        self._matched = False
        self._stage = _FAILED

    def _feed_body(self, char: str) -> str:
        if char == '\\':
            self._stage = _ESCAPE
            return ''
        if char == '"':  # an unescaped quote can only be the value's closing quote
            pending = self._take_high()
            self._stage = _DONE
            return pending
        return self._take_high() + char

    def _feed_escape(self, char: str) -> str:
        if char == 'u':
            self._hex = ''
            self._stage = _UNICODE
            return ''
        decoded = _ESCAPES.get(char)  # a pending high surrogate here can never pair up
        if decoded is None:  # invalid escape — the argument stream is corrupt
            self._failed = True
            self._stage = _FAILED
            return ''
        self._stage = _BODY
        return self._take_high() + decoded

    def _feed_unicode(self, char: str) -> str:
        if char not in _HEX:  # malformed \u — the argument stream is corrupt
            self._failed = True
            self._stage = _FAILED
            return ''
        self._hex += char
        if len(self._hex) < 4:
            return ''
        unit = int(self._hex, 16)
        self._hex = ''
        self._stage = _BODY
        high = self._high
        if high is None:
            if 0xD800 <= unit <= 0xDBFF:  # a high surrogate must wait for its low half
                self._high = unit
                return ''
            return chr(unit)
        self._high = None
        if 0xDC00 <= unit <= 0xDFFF:  # the pair completes into one astral code point
            return chr(0x10000 + ((high - 0xD800) << 10) + (unit - 0xDC00))
        # not a pair: emit the lone high surrogate, then this unit on its own terms.
        return chr(high) + self._feed_unicode_unit(unit)

    def _feed_unicode_unit(self, unit: int) -> str:
        if 0xD800 <= unit <= 0xDBFF:
            self._high = unit
            return ''
        return chr(unit)

    def _take_high(self) -> str:
        """Flush a pending lone high surrogate (json.loads keeps it as-is)."""
        high, self._high = self._high, None
        return '' if high is None else chr(high)
