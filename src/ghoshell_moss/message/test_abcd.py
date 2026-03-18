"""
极简的 Message 协议测试
验证核心数据类型的基本功能
"""

import json
from datetime import datetime, timezone

import pytest

from ghoshell_moss.message.abcd import (
    Message,
    MessageMeta,
    Role,
    Content,
    Addition,
    WithAdditional,
)
from ghoshell_moss.message.contents import (
    Text,
    Base64Image,
    ImageUrl,
)


def test_message_meta_basic():
    """测试 MessageMeta 基本功能"""
    meta = MessageMeta(
        role="user",
        name="test_user",
        stage="thinking",
        issuer="terminal",
        issuer_id="term_001",
    )

    assert meta.role == "user"
    assert meta.name == "test_user"
    assert meta.stage == "thinking"
    assert meta.issuer == "terminal"
    assert meta.issuer_id == "term_001"
    assert isinstance(meta.id, str) and len(meta.id) > 0
    assert isinstance(meta.created_at, datetime)

    # 测试 XML 转换
    xml = meta.to_xml()
    assert "role='user'" in xml
    assert "name='test_user'" in xml
    assert "stage='thinking'" in xml
    assert xml.startswith("<meta") and xml.endswith("/>")


def test_message_creation():
    """测试 Message 创建和基本属性"""
    # 使用 new() 方法创建
    msg = Message.new(role="user", name="test")
    assert msg.role == "user"
    assert msg.name == "test"
    assert msg.id == msg.meta.id

    # 测试 with_content 方法
    msg.with_content("Hello world")
    assert msg.contents is not None
    assert len(msg.contents) == 1
    assert msg.contents[0]["type"] == "text"
    assert msg.contents[0]["data"] == "Hello world"

    # 测试 is_empty
    empty_msg = Message.new()
    assert empty_msg.is_empty() == True
    assert msg.is_empty() == False


def test_content_model_text():
    """测试 Text ContentModel 转换"""
    text_obj = Text(text="Hello world")

    # 测试 marshal
    marshaled = text_obj.marshal()
    assert marshaled == "Hello world"

    # 测试 to_content
    content = text_obj.to_content()
    assert content["type"] == "text"
    assert content["data"] == "Hello world"

    # 测试 from_content
    recovered = Text.from_content(content)
    assert recovered is not None
    assert recovered.text == "Hello world"

    # 测试 unmarshal
    data = Text.unmarshal("Test text")
    assert data == {"text": "Test text"}


def test_content_model_image_url():
    """测试 ImageUrl ContentModel 转换"""
    url = "https://example.com/image.jpg"
    img_obj = ImageUrl(url=url)

    # 测试 marshal
    marshaled = img_obj.marshal()
    assert marshaled == url

    # 测试 to_content
    content = img_obj.to_content()
    assert content["type"] == "image_url"
    assert content["data"] == url

    # 测试 from_content
    recovered = ImageUrl.from_content(content)
    assert recovered is not None
    assert recovered.url == url

    # 测试 unmarshal
    data = ImageUrl.unmarshal(url)
    assert data == {"url": url}


def test_message_serialization():
    """测试 Message 序列化/反序列化"""
    # 创建带内容的 Message
    msg = Message.new(role="assistant", name="ai")
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
    assert parsed.role == "assistant"
    assert parsed.name == "ai"
    assert parsed.contents is not None
    assert len(parsed.contents) == 2

    # 测试 XML 转换
    xml = msg.to_xml()
    assert xml.startswith("<message>") or xml.startswith("<message:")
    assert "Hello" in xml
    assert "World" in xml


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


def test_role_enum():
    """测试 Role 枚举功能"""
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.SYSTEM.value == "system"
    assert Role.DEVELOPER.value == "developer"

    # 测试 all() 方法
    all_roles = Role.all()
    assert "user" in all_roles
    assert "assistant" in all_roles
    assert "system" in all_roles
    assert "developer" in all_roles

    # 测试 new_meta 方法
    meta = Role.USER.new_meta(name="test_user", stage="thinking")
    assert meta.role == "user"
    assert meta.name == "test_user"
    assert meta.stage == "thinking"


def test_message_with_raw_protocol():
    """测试带原始协议的消息"""
    raw_data = {
        "role": "user",
        "content": "Hello",
        "name": "test_user"
    }

    msg = Message.from_raw(
        protocol="openai",
        raw_data=raw_data,
        type="chat.completion",
        meta=MessageMeta(role="user", name="test_user")
    )

    assert msg.protocol == "openai"
    assert msg.raw == raw_data
    assert msg.type == "chat.completion"
    assert msg.meta.role == "user"
    assert msg.meta.name == "test_user"
