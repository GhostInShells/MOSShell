"""
极简的 Message 协议测试
验证核心数据类型的基本功能
"""

from datetime import datetime

from ghoshell_moss.message import (
    Message,
    MessageMeta,
    Addition,
    WithAdditional,
    Text,
)
import json


def test_message_meta_basic():
    """测试 MessageMeta 基本功能"""
    meta = MessageMeta(
        role="user",
        name="test_user",
    )

    assert meta.role == "user"
    assert meta.name == "test_user"
    assert isinstance(meta.id, str) and len(meta.id) > 0
    assert isinstance(meta.created, datetime)

    # 测试 XML 转换
    xml = meta.to_xml()
    assert 'role="user"' in xml
    assert 'name="test_user"' in xml
    assert xml.startswith("<message") and xml.endswith("/>")


def test_message_creation():
    """测试 Message 创建和基本属性"""
    # 使用 new() 方法创建
    msg = Message.new(name="test")
    assert msg.name == "test"
    assert msg.id == msg.meta.id

    # 测试 with_content 方法
    msg.with_content("Hello world")
    assert msg.contents is not None
    assert len(msg.contents) == 1
    assert Text.from_content(msg.contents[0]).text == "Hello world"

    # 测试 is_empty
    empty_msg = Message.new()
    assert empty_msg.is_empty() == True
    assert msg.is_empty() == False


def test_message_serialization():
    """测试 Message 序列化/反序列化"""
    # 创建带内容的 Message
    msg = Message.new(name="ai", tag="message")
    msg.with_content("Hello", "World")

    # 测试 dump
    data = msg.dump()
    assert "meta" in data
    assert "contents" in data
    assert len(data["contents"]) == 2

    # 测试 JSON 序列化
    json_str = msg.to_json()
    assert isinstance(json_str, str)

    # 测试从 JSON 反序列化
    parsed = Message.model_validate_json(json_str)
    assert parsed.name == "ai"
    assert parsed.contents is not None
    assert len(parsed.contents) == 2

    # 测试 to_contents() 方法
    contents = list(msg.as_contents())
    assert len(contents) == 4  # 开始标签 + meta + 2个内容 + 结束标签
    assert Text.from_content(contents[0]).text.strip().startswith("<message")
    assert Text.from_content(contents[1]).text == "Hello"
    assert Text.from_content(contents[2]).text == "World"
    assert Text.from_content(contents[3]).text.strip() == "</message>"


def test_addition_system():
    """测试 Addition 扩展系统"""

    class TestAddition(Addition):
        """测试用的 Addition"""
        field1: str = "default"
        field2: int = 0

        @classmethod
        def keyword(cls) -> str:
            return "test.addition"

    # 创建目标对象
    class TestTarget(WithAdditional):
        additional = None

    target = TestTarget()

    # 测试 set 和 read
    addition = TestAddition(field1="value", field2=42)
    addition.set(target)

    assert target.additional is not None
    assert "test.addition" in target.additional

    # 测试 read
    recovered = TestAddition.read(target)
    assert recovered is not None
    assert recovered.field1 == "value"
    assert recovered.field2 == 42

    # 测试 get_or_create
    existing = addition.get_or_create(target)
    assert existing.field1 == addition.field1 and existing.field2 == addition.field2  # 值相等


def test_message_serializable():
    message = Message.new(name="ai", timestamp=True)
    js = message.model_dump_json()
    data = json.loads(js)
    new_message = Message(**data)
    assert new_message == message


def test_message_with_addition():
    message = Message.new(name="ai", timestamp=True)

    class TestAddition(Addition):
        foo: str = 'foo'

        @classmethod
        def keyword(cls) -> str:
            return "test.addition"

    message.with_additions(TestAddition())
    assert TestAddition.read(message) is not None

    copied = message.model_copy()
    assert TestAddition.read(copied).foo == "foo"
