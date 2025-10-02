from typing import List
from ghoshell_moss.concepts.command import CommandToken, CommandTokenType
from ghoshell_moss.concepts.errors import InterpretError
from ghoshell_moss.ctml.token_parser import CTMLTokenParser
from collections import deque


def test_token_parser_baseline():
    q = deque[CommandToken]()
    parser = CTMLTokenParser(callback=q.append, stream_id="stream")
    content = "<foo><bar/>h</foo>"
    with parser:
        for c in content:
            parser.feed(c)
        parser.commit()
    assert parser.is_done()
    assert parser.buffer() == content
    # receive the poison item
    assert q.pop() is None
    assert len(q) == 7

    # output tokens in order
    order = 0
    for token in q:
        # start from 0
        assert token.order == order
        assert token.stream_id == "stream"
        order += 1

    # command start make idx ++
    for token in q:
        if token.name == "foo":
            assert token.cmd_idx == 1
        elif token.name == "bar":
            assert token.cmd_idx == 2

    part_idx = 0
    for token in q:
        if token.name == "foo":
            # the cmd idx is the same since only one foo exists
            assert token.cmd_idx == 1
            # the part idx increase since only 'h' as delta
            assert token.part_idx == part_idx
            part_idx += 1


def test_token_parser_with_args():
    content = '<foo a="1" b="[2, 3]"/>'
    q = deque[CommandToken | None]()
    CTMLTokenParser.parse(q.append, iter(content))
    assert q.pop() is None
    assert q[1].name == "foo"
    assert q[1].kwargs == {"a": "1", "b": "[2, 3]"}


def test_delta_token_baseline():
    content = "<foo>hello<bar/>world</foo>"
    q = deque[CommandToken | None]()
    CTMLTokenParser.parse(q.append, iter(content))
    # received the poison item
    assert q.pop() is None

    text = ""
    for token in q:
        if token.name == "foo":
            text += token.content
    assert text == "<foo>helloworld</foo>"

    for token in q:
        if token.name != "foo":
            continue
        elif token.type == "start":
            assert token.part_idx == 0
        elif token.type == "delta":
            assert token.part_idx in (1, 2)
        elif token.type == "end":
            assert token.part_idx == 3

    delta_part_1 = ""
    delta_part_1_count = 0
    for token in q:
        if token.name == "foo" and token.part_idx == 1:
            delta_part_1 += token.content
            delta_part_1_count += 1
    assert delta_part_1 == "hello"

    delta_part_2 = ""
    delta_part_2_count = 0
    for token in q:
        if token.name == "foo" and token.part_idx == 2:
            delta_part_2 += token.content
            delta_part_2_count += 1
    assert delta_part_2 == "world"

    # [<foo>, 1], [he-l-l-o, 5], [<bar>,1], [</bar>, 1], [wo-r-l-d, 5], [</foo>, 1]
    assert (len(q) - 2) == (1 + delta_part_1_count + 2 + delta_part_2_count + 1)


def test_token_with_attrs():
    content = "hello<foo bar='123'/>world"
    q: List[CommandToken] = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    # received the poison item
    assert q.pop() is None
    assert q[0].name == "speak"
    assert q[-1].name == "speak"

    # skip the head and the tail
    q = q[1:-1]

    foo_token_count = 0
    for token in q:
        if token.name == "foo":
            assert token.cmd_idx == 1
            foo_token_count += 1
            if token.type == "start":
                # is string value
                assert token.kwargs == dict(bar="123")
    assert foo_token_count == 2

    first_token = q[0]
    last_token = q[-1]
    # belongs to the root, cmd_idx is 0
    # root tag parts: <speak> , hello, world, </speak>
    assert first_token.name == "speak"
    assert first_token.cmd_idx == 0
    assert first_token.part_idx == 1
    assert first_token.type == CommandTokenType.DELTA.value

    assert last_token.name == "speak"
    assert last_token.cmd_idx == 0
    assert last_token.type == CommandTokenType.DELTA.value
    assert last_token.part_idx == 2


def test_token_with_cdata():
    content = 'hello<foo><![CDATA[{"a": 123, "b":"234"}]]></foo>world'
    q = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None

    # expect hte cdata are escaped
    expect = '{"a": 123, "b":"234"}'
    foo_deltas = ""
    for token in q[1:-1]:
        if token.name == "foo" and token.type == "delta":
            foo_deltas += token.content
    assert expect == foo_deltas


def test_token_with_prefix():
    content = "<speaker__say>hello</speaker__say>"
    q = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="ctml")
    assert q.pop() is None
    for token in q[1:-1]:
        assert token.name == "speaker__say"


def test_token_with_recursive_cdata():
    content = '<foo><![CDATA[hello<![CDATA[foo]]>world]]></foo>'
    q = deque[CommandToken]()
    e = None
    try:
        CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    except Exception as ex:
        e = ex
    assert isinstance(e, InterpretError)


def test_space_only_delta():
    content = '<foo> </foo>'
    q = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None

    q = q[1:-1]
    assert "".join(t.content for t in q) == content


def test_namespace_tag():
    content = '<foo:bar a="123" />'
    q: List[CommandToken] = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None
    q = q[1:-1]
    assert len(q) == 2

    start_token = q[0]
    assert start_token.name == "bar"
    assert start_token.chan == "foo"
    assert start_token.kwargs == dict(a="123")
