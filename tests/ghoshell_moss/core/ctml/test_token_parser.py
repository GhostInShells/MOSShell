import threading
import time
from collections import deque

import pytest

from ghoshell_moss.core.concepts.command import CommandToken, CommandTokenSeq
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.ctml.token_parser import CTML2CommandTokenParser, ctml_default_parsers, AttrPrefixParser
from ast import literal_eval


def test_token_parser_baseline():
    q = deque[CommandToken]()
    parser = CTML2CommandTokenParser(callback=q.append, stream_id="stream")
    content = "<foo><bar/>h</foo>"
    with parser:
        for c in content:
            parser.feed(c)
        parser.commit()
    assert parser.is_done()
    assert parser.buffered() == content
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
    has_delta = False
    for token in q:
        if token.seq == "delta":
            has_delta = True
        if token.name == "foo":
            # the cmd idx is the same since only one foo exists
            assert token.cmd_idx == 1
            # the part idx increase since only 'h' as delta
            assert token.part_idx == part_idx
            part_idx += 1
    assert has_delta


def test_token_parser_with_args():
    content = '<foo a="1" b="[2, 3]"/>'
    q = deque[CommandToken | None]()
    CTML2CommandTokenParser.parse(q.append, iter(content))
    assert q.pop() is None
    assert q[1].name == "foo"
    assert q[1].kwargs == {"a": "1", "b": "[2, 3]"}


def test_delta_token_baseline():
    content = "<foo>hello<bar/>world</foo>"
    q = deque[CommandToken | None]()
    CTML2CommandTokenParser.parse(q.append, iter(content))
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
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak")
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
    assert first_token.seq == CommandTokenSeq.DELTA.value

    assert last_token.name == "speak"
    assert last_token.cmd_idx == 0
    assert last_token.seq == CommandTokenSeq.DELTA.value
    assert last_token.part_idx == 2


def test_token_with_cdata():
    content = 'hello<foo><![CDATA[{"a": 123, "b":"234"}]]></foo>world'
    q = []
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak")
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
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="ctml")
    assert q.pop() is None
    assert len(q) > 1


def test_token_with_prefix():
    content = "<speaker__say>hello</speaker__say>"
    q = []
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="ctml")
    assert q.pop() is None
    for token in q[1:-1]:
        assert token.name == "speaker__say"


def test_token_with_recursive_cdata():
    content = "<foo><![CDATA[hello<![CDATA[foo]]>world]]></foo>"
    q = deque[CommandToken]()
    e = None
    try:
        CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak")
    except Exception as ex:
        e = ex
    assert isinstance(e, InterpretError)


def test_space_only_delta():
    content = "<foo> </foo>"
    q = []
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None

    q = q[1:-1]
    assert "".join(t.content for t in q) == content


def test_namespace_tag():
    content = '<foo:bar a="123" />'
    q: list[CommandToken] = []
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak")
    assert q.pop() is None
    q = q[1:-1]
    assert len(q) == 2

    start_token = q[0]
    assert start_token.name == "bar"
    assert start_token.chan == "foo"
    assert start_token.kwargs == {"a": "123"}


def test_arg_with_parsers():
    content = '<foo:bar a="123" b:str="123"/>'
    q: list[CommandToken] = []
    CTML2CommandTokenParser.parse(
        q.append,
        iter(content),
        root_tag="speak",
        attr_parsers=ctml_default_parsers,
    )
    assert q.pop() is None
    q = q[1:-1]
    assert len(q) == 2

    start_token = q[0]
    assert start_token.name == "bar"
    assert start_token.chan == "foo"
    assert start_token.kwargs == {"a": 123, "b": "123"}


def test_parser_with_chinese():
    content = "<foo.bar:baz>你好啊</foo.bar:baz>"
    q: list[CommandToken] = []
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak")
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
    CTML2CommandTokenParser.parse(
        q.append,
        iter(content),
        root_tag="speak",
    )
    assert q.pop() is None
    q = q[1:-1]

    assert "".join([t.content for t in q]) == content


def test_token_parser_with_attr_suffix():
    # CTML 1.0.0 隐藏使用三元命名法, chan:command:call_id.
    content = "<a:foo:3 a:list='[1, 2]' b:lambda='2*3' c:dict='{\"foo\": 123}' />"
    q: list[CommandToken] = []
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak", attr_parsers=ctml_default_parsers)
    q = q[1:-1]
    for token in q:
        if token.seq == "start":
            assert token.call_id == '3'
            assert token.kwargs == {"a": [1, 2], "b": 6, "c": {"foo": 123}}


def test_ctml_with_suffix_idx():
    content = "<a:foo:3 literal-a='[1, 2]'></a:foo:3><bar/>"
    q: list[CommandToken] = []
    parsers = ctml_default_parsers.copy()
    parsers.append(AttrPrefixParser(desc="", prefix="literal-", parser=lambda v: literal_eval(v)))
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak", attr_parsers=parsers)
    q = q[1:-1]
    token = q.pop(0)
    assert token.seq == "start"
    assert token.call_id == '3'
    assert token.order == 1
    assert token.kwargs["a"] == [1, 2]
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
    literal_parser = AttrPrefixParser(desc="", prefix="literal-", parser=lambda v: literal_eval(v))
    CTML2CommandTokenParser.parse(
        q.append, iter(content), root_tag="speak", attr_parsers=[literal_parser], with_call_id=True
    )
    got_content = "".join([t.content for t in q[1:-2]])
    assert got_content == '<a:foo:1 literal-a="[1, 2]"></a:foo:1><bar></bar>'


def test_ctml_attr_with_args():
    content = "<a:foo _args='[1, 2]'></a:foo>"
    q: list[CommandToken] = []
    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak", attr_parsers=ctml_default_parsers)
    q = q[1:-1]
    token = q.pop(0)
    assert token.seq == "start"
    assert token.args == [1, 2]


def test_token_parser_in_threads():
    got = []

    _content = "<a:foo _args='[1, 2]'></a:foo>"

    def iter_content():
        for c in _content:
            time.sleep(0.01)
            yield c

    def in_thread_parse():
        q: list[CommandToken] = []
        CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
        got.append(list(q))

    threads = []
    for i in range(10):
        t = threading.Thread(target=in_thread_parse)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    assert len(got) == 10
    expect = ""
    for tokens in got:
        content = ""
        for token in tokens:
            if token is not None:
                content += token.content

        if not expect:
            expect = content
            continue
        assert content == expect


def test_token_parser_receive_empty():
    q: list[CommandToken] = []

    def iter_content():
        yield from []

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    # 拿到了 CTML 开头, 和 None 结尾.
    assert len(q) == 3
    assert q.pop() is None
    assert len(q) == 2


def test_token_parser_raise_on_invalid_args():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<foo:bar _args='{1: 2}'/>":
            yield c

    with pytest.raises(InterpretError):
        CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)


def test_token_with_scope():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<foo:bar><baz /></foo:bar>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    for token in q:
        if token and token.name == "baz":
            # 被赋予了命名空间.
            assert token.chan == "foo"


def test_token_with_scope_func():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<_ channel='foo'><baz /><_ channel='foo.bar'>hello<zoo /></_><_><coo />world</_></_>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    count = 0
    for token in q:
        if token and token.name == "baz":
            # 被赋予了命名空间.
            count += 1
            assert token.chan == "foo"
        if token and token.name == "zoo":
            count += 1
            assert token.chan == "foo.bar"
        if token and token.name == "coo":
            count += 1
            assert token.chan == "foo"
        if token and token.seq == 'delta':
            assert token.chan in ['foo.bar', 'foo']
    assert count > 1


def test_token_with_call_id():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<_ channel='foo'><bar _cid='123' /></_>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    has_baz = False
    for token in q:
        if token and token.name == "bar" and token.seq == 'start':
            assert token.chan == "foo"
            assert token.call_id == '123'
            has_baz = True
    assert has_baz


def test_token_content_within_scope():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<_>hello<bar _cid='123' /> world</_>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    content = ""
    for token in q:
        if token and token.seq == 'delta':
            assert token.chan == ""
            content += token.content
    assert content == "hello world"


def test_token_delta_inherit_channel_within_scope():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<_ name='foo'>hello<bar _cid='123' /> world</_>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    content = ""
    for token in q:
        if token and token.seq == 'delta':
            assert token.chan == ""
            content += token.content
    assert content == "hello world"


def test_sub_token_has_it_own_scope():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<_ channel='foo'>hello<foo.bar:bar _cid='123' /> world</_>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    has_bar = False
    for token in q:
        if token and token.seq == 'start' and token.name == 'bar':
            assert token.chan == "foo.bar"
            assert token.call_id == '123'
            has_bar = True
    assert has_bar


def test_sub_scope_inherit_channel():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<_ channel='foo'><_>hello<bar _cid='123' /></_> world</_>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    has_bar = False
    content = ""
    for token in q:
        if token and token.seq == 'start' and token.name == 'bar':
            assert token.chan == "foo"
            has_bar = True
        if token and token.seq == "delta":
            if token.chan == "foo":
                content += token.content
    assert has_bar
    assert content == "hello world"


def test_sub_scope_not_allow_defer_parent():
    q: list[CommandToken] = []

    def iter_content():
        # args shall be an array
        for c in "<_ channel='foo'><_ channel='bar'>hello<bar _cid='123' /></_> world</_>":
            yield c

    # bar scope 越界了 foo.
    with pytest.raises(InterpretError):
        CTML2CommandTokenParser.parse(
            q.append,
            iter_content(),
            root_tag="speak",
            attr_parsers=ctml_default_parsers,
        )


def test_sub_scope_with_inherit_scope():
    q: list[CommandToken] = []

    # 隐藏继承逻辑, 不轻易开放. 当子节点 channel 用 . 开头时, 实际上会继承 scope.
    def iter_content():
        # args shall be an array
        for c in "<_ channel='foo'><_ channel='.bar'>hello<bar _cid='123' /></_> world</_>":
            yield c

    CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)
    has_bar = False
    for token in q:
        if token and token.seq == 'start' and token.name == 'bar':
            assert token.chan == "foo.bar"
            has_bar = True
    assert has_bar


def test_scope_with_until_flow_and_timeout():
    """测试 CTML 1.0.0 新增的 until="flow" 和 timeout 属性解析"""
    content = '<_ channel="robot.arm" until="flow" timeout="5.0"><bar /></_>'
    q: list[CommandToken] = []

    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak", attr_parsers=ctml_default_parsers)
    assert q.pop() is None
    q = q[1:-1]

    scope_start_token = q[0]
    assert scope_start_token.seq == "start"
    assert scope_start_token.name == "_"
    # 验证 kwargs 是否正确承载了这些属性，且 timeout 被正确转为 float (如果 default parser 支持的话)
    # 注意：如果你们的 literal_eval 默认不处理纯字符串属性，这里的值可能是 string，视你的 parser 基础逻辑而定
    assert scope_start_token.kwargs.get("until") == "flow"
    assert str(scope_start_token.kwargs.get("timeout")) == "5.0"


def test_token_parser_comprehensive_type_suffixes():
    """测试完整的类型消歧义 (bool, float, none)"""
    content = '<foo:bar a:bool="True" b:bool="False" c:float="3.14" d:none="None" e:str="123"/>'
    q: list[CommandToken] = []

    CTML2CommandTokenParser.parse(q.append, iter(content), root_tag="speak", attr_parsers=ctml_default_parsers)
    assert q.pop() is None
    q = q[1:-1]

    token = q[0]
    assert token.seq == "start"
    assert token.kwargs["a"] is True
    assert token.kwargs["b"] is False
    assert token.kwargs["c"] == 3.14
    assert token.kwargs["d"] is None
    assert token.kwargs["e"] == "123"


def test_token_parser_raise_on_missing_quotes():
    """强制红线测试：严禁省略属性引号"""
    q: list[CommandToken] = []

    def iter_content():
        for c in "<foo:bar arg=123 />":
            yield c

    # 缺乏引号应该在 XML 解析阶段直接引发 InterpretError (快速失败)
    with pytest.raises(InterpretError):
        CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)


def test_token_parser_raise_on_mismatched_tags():
    """健壮性测试：标签开闭不匹配时的快速失败"""
    q: list[CommandToken] = []

    def iter_content():
        for c in "<foo><bar></foo></bar>":
            yield c

    with pytest.raises(InterpretError):
        CTML2CommandTokenParser.parse(q.append, iter_content(), root_tag="speak", attr_parsers=ctml_default_parsers)

