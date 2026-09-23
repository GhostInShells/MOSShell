"""ReactStore + react tools — react is a small string-template function over CTML.

The store is char → template (in-memory). moss_define_reacts bulk-defines (merge/overwrite);
moss_react expands one. Test the substitution edge cases and the tool parse directly.
"""

import json

import pytest

from ghoshell_moss.deepseek_harness.types.session_events import ToolCallEvent

from ._react import React, ReactStore
from ._tools import DefineReactsToolCall, ReactToolCall


def _store(*reacts: tuple[str, str]) -> ReactStore:
    store = ReactStore()
    store.define([React(char=c, template=t) for c, t in reacts])
    return store


def test_define_and_render():
    store = _store(("s", "<say>%s</say>"))
    assert store.render("s", ["hello"]) == "<say>hello</say>"


def test_render_multi_arg():
    store = _store(("m", "<move x=%s y=%s/>"))
    assert store.render("m", ["1", "2"]) == "<move x=1 y=2/>"


def test_render_without_args_no_placeholder():
    store = _store(("a", "<nod/>"))
    assert store.render("a") == "<nod/>"


def test_render_missing_arg_raises():
    store = _store(("s", "<say>%s</say>"))
    with pytest.raises(ValueError):
        store.render("s")


def test_render_extra_arg_raises():
    store = _store(("s", "<say>%s</say>"))
    with pytest.raises(ValueError):
        store.render("s", ["a", "b"])


def test_render_unknown_char_raises_keyerror():
    with pytest.raises(KeyError):
        ReactStore().render("x", [])


def test_render_literal_percent():
    store = _store(("p", "<say>%s%%</say>"))
    assert store.render("p", ["100"]) == "<say>100%</say>"


def test_define_returns_defined_chars():
    store = ReactStore()
    defined = store.define([React(char="s", template="<say>%s</say>"), React(char="m", template="<move/>")])
    assert defined == ["s", "m"]


def test_define_overwrites():
    store = _store(("s", "<a>%s</a>"))
    store.define([React(char="s", template="<b>%s</b>")])
    assert store.render("s", ["x"]) == "<b>x</b>"


def test_define_rejects_multichar():
    with pytest.raises(ValueError):
        ReactStore().define([React(char="ab", template="")])


def test_react_tool_call_parses():
    event = ToolCallEvent(
        name="moss_react",
        arguments=json.dumps({"char": "s", "args": ["hello"], "wait_next_moment": False}),
        callId="c1",
    )
    call = ReactToolCall.from_tool_call(event)
    assert call.char == "s"
    assert call.args == ["hello"]
    assert call.wait_next_moment is False


def test_react_tool_call_defaults():
    event = ToolCallEvent(name="moss_react", arguments=json.dumps({"char": "s"}), callId="c2")
    call = ReactToolCall.from_tool_call(event)
    assert call.args is None
    assert call.wait_next_moment is True


def test_define_reacts_tool_call_parses():
    event = ToolCallEvent(
        name="moss_define_reacts",
        arguments=json.dumps({"reacts": [{"char": "s", "template": "<say>%s</say>"}]}),
        callId="c3",
    )
    call = DefineReactsToolCall.from_tool_call(event)
    assert call.reacts == [React(char="s", template="<say>%s</say>")]


def test_react_tool_call_name_mismatch():
    event = ToolCallEvent(name="moss_ctml_append", arguments=json.dumps({"ctml": ""}), callId="c4")
    assert ReactToolCall.from_tool_call(event) is None
