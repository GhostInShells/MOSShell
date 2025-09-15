from ghoshell_moss.concepts.command import CommandToken
from ghoshell_moss.concepts.interpreter import CommandTokenParseError
from ghoshell_moss.ctml.token_parser import CTMLParser
from collections import deque


def test_token_parser_baseline():
    q = deque[CommandToken]()
    parser = CTMLParser(callback=q.append, stream_id="stream")
    content = "<foo><bar/>h</foo>"
    with parser:
        assert parser.is_running()
        for c in content:
            parser.feed(c)
        parser.end()
    assert not parser.is_running()
    assert parser.is_done()
    assert parser.buffer() == content
    assert len(q) == 5

    # output tokens in order
    order = 0
    for token in q:
        # start from 1
        order += 1
        assert token.order == order
        assert token.stream_id == "stream"

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
            part_idx += 1
            assert token.part_idx == part_idx


def test_delta_token_baseline():
    content = "<foo>hello<bar/>world</foo>"
    q = deque[CommandToken]()
    CTMLParser.parse(q.append, iter(content))

    text = ""
    for token in q:
        if token.name == "foo":
            text += token.content
    assert text == "<foo>helloworld</foo>"

    for token in q:
        if token.name != "foo":
            continue
        elif token.type == "start":
            assert token.part_idx == 1
        elif token.type == "delta":
            assert token.part_idx in (2, 3)
        elif token.type == "end":
            assert token.part_idx == 4

    delta_part_1 = ""
    delta_part_1_count = 0
    for token in q:
        if token.name == "foo" and token.part_idx == 2:
            delta_part_1 += token.content
            delta_part_1_count += 1
    assert delta_part_1 == "hello"

    delta_part_2 = ""
    delta_part_2_count = 0
    for token in q:
        if token.name == "foo" and token.part_idx == 3:
            delta_part_2 += token.content
            delta_part_2_count += 1
    assert delta_part_2 == "world"

    # [<foo>, 1], [he-l-l-o, 5], [<bar>,1], [</bar>, 1], [wo-r-l-d, 5], [</foo>, 1]
    assert len(q) == (1 + delta_part_1_count + 2 + delta_part_2_count + 1)


def test_token_with_attrs():
    content = "hello<foo bar='123'/>world"
    q = deque[CommandToken]()
    CTMLParser.parse(q.append, iter(content), root_tag="speak")

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
    assert first_token.cmd_idx == 0
    assert last_token.cmd_idx == 0
    # first part of the root tag
    assert first_token.part_idx == 1
    # second part of the root tag
    assert last_token.part_idx == 2

    assert first_token.name == "speak"
    assert last_token.name == "speak"


def test_token_with_cdata():
    content = 'hello<foo><![CDATA[{"a": 123, "b":"234"}]]></foo>world'
    q = deque[CommandToken]()
    CTMLParser.parse(q.append, iter(content), root_tag="speak")

    # expect hte cdata are escaped
    expect = '{"a": 123, "b":"234"}'
    foo_deltas = ""
    for token in q:
        if token.name == "foo" and token.type == "delta":
            foo_deltas += token.content
    assert expect == foo_deltas


def test_token_with_recursive_cdata():
    content = '<foo><![CDATA[hello<![CDATA[foo]]>world]]></foo>'
    q = deque[CommandToken]()
    e = None
    try:
        CTMLParser.parse(q.append, iter(content), root_tag="speak")
    except Exception as ex:
        e = ex
    assert isinstance(e, CommandTokenParseError)
