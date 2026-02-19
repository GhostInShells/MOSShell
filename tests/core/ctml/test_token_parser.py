from collections import deque

from ghoshell_moss.core.concepts.command import CommandToken, CommandTokenType
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.ctml.token_parser import CTMLTokenParser, literal_parser


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
        elif token.seq == "start":
            assert token.part_idx == 0
        elif token.seq == "delta":
            assert token.part_idx in (1, 2)
        elif token.seq == "end":
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
    q: list[CommandToken] = []
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
            if token.seq == "start":
                # is string value
                assert token.kwargs == {"bar": "123"}
    assert foo_token_count == 2

    first_token = q[0]
    last_token = q[-1]
    # belongs to the root, cmd_idx is 0
    # root tag parts: <speak> , hello, world, </speak>
    assert first_token.name == "speak"
    assert first_token.cmd_idx == 0
    assert first_token.part_idx == 1
    assert first_token.seq == CommandTokenType.DELTA.value

    assert last_token.name == "speak"
    assert last_token.cmd_idx == 0
    assert last_token.seq == CommandTokenType.DELTA.value
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
        if token.name == "foo" and token.seq == "delta":
            foo_deltas += token.content
    assert expect == foo_deltas


def test_token_with_cdata_content():
    content = """
<mac_jxa:run_jxa><![CDATA[
(function() {
    const Calendar = Application('Calendar');
    Calendar.includeStandardAdditions = true;
    Calendar.activate();
    return "已为你打开日历应用";
})();
]]></mac_jxa:run_jxa>
"""
    q = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="ctml")
    assert q.pop() is None
    assert len(q) > 1


def test_token_with_prefix():
    content = "<speaker__say>hello</speaker__say>"
    q = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="ctml")
    assert q.pop() is None
    for token in q[1:-1]:
        assert token.name == "speaker__say"


def test_token_with_recursive_cdata():
    content = "<foo><![CDATA[hello<![CDATA[foo]]>world]]></foo>"
    q = deque[CommandToken]()
    e = None
    try:
        CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    except Exception as ex:
        e = ex
    assert isinstance(e, InterpretError)


def test_space_only_delta():
    content = "<foo> </foo>"
    q = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None

    q = q[1:-1]
    assert "".join(t.content for t in q) == content


def test_namespace_tag():
    content = '<foo:bar a="123" />'
    q: list[CommandToken] = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None
    q = q[1:-1]
    assert len(q) == 2

    start_token = q[0]
    assert start_token.name == "bar"
    assert start_token.chan == "foo"
    assert start_token.kwargs == {"a": "123"}


def test_parser_with_chinese():
    content = "<foo.bar:baz>你好啊</foo.bar:baz>"
    q: list[CommandToken] = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None
    q = q[1:-1]

    assert "".join([t.content for t in q]) == content


def test_token_parser_with_json():
    content = """
<jetarm:run_trajectory>
    {"joint_names": ["gripper", "wrist_roll", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
    "points": [{"positions": [2.16, 11.16, -60.0, -135.0, 60.0, -0.36], "time_from_start": 0.0},
    {"positions": [5.0, 15.0, -55.0, -130.0, 55.0, 2.0], "time_from_start": 1.0},
    {"positions": [2.16, 11.16, -60.0, -135.0, 60.0, -0.36], "time_from_start": 2.0}]}
</jetarm:run_trajectory>
"""
    q: list[CommandToken] = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak", )
    assert q.pop() is None
    q = q[1:-1]

    assert "".join([t.content for t in q]) == content


def test_token_parser_with_idx():
    content = "<a:foo:3 literal-a='[1, 2]'></a:foo:3><bar/>"
    q: list[CommandToken] = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak", attr_parsers=[literal_parser])
    q = q[1:-1]
    token = q.pop(0)
    assert token.seq == "start"
    assert token.call_id == 3
    assert token.order == 1
    assert token.kwargs['a'] == [1, 2]
    next_token = None
    for token in q:
        if token.name == "bar":
            next_token = token
            break
    assert next_token is not None
    assert next_token.seq == "start"
    assert next_token.cmd_idx == 2
    assert next_token.call_id is None

    content = "<a:foo:1 literal-a='[1, 2]'></a:foo:1><bar/>"
    q: list[CommandToken] = []
    CTMLTokenParser.parse(q.append, iter(content), root_tag="speak", attr_parsers=[literal_parser], with_call_id=True)
    got_content = "".join([t.content for t in q[1:-2]])
    assert got_content == '<a:foo:1 literal-a="[1, 2]"></a:foo:1><bar:2></bar:2>'
