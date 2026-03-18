"""
EventBus 数据类型的简单 pytest 测试.
专注于验证数据结构定义的基础问题.
"""

import datetime
import time
from typing import Any

import pytest
from pydantic import ValidationError

from .eventbus import (
    EventMeta,
    Event,
    EventModel,
    Publisher,
    Subscriber,
    SubscriberMode,
    EventBus,
)


class ExampleEventModel(EventModel):
    """测试用的 EventModel 示例"""
    value: str = "default"
    count: int = 0

    @classmethod
    def event_type(cls) -> str:
        return "test/example"


def test_event_meta_defaults():
    """测试 EventMeta 默认值"""
    meta = EventMeta()
    assert meta.id is not None
    assert meta.issuer == ""
    assert meta.event_type == ""
    assert meta.priority == 0
    assert meta.overdue == 0
    assert isinstance(meta.created_at, datetime.datetime)


def test_event_meta_custom():
    """测试 EventMeta 自定义值"""
    meta = EventMeta(
        issuer="Shell/test",
        event_type="test/event",
        priority=5,
        overdue=10.0,
    )
    assert meta.issuer == "Shell/test"
    assert meta.event_type == "test/event"
    assert meta.priority == 5
    assert meta.overdue == 10.0


def test_event_default():
    """测试 Event 默认创建"""
    event = Event()
    assert isinstance(event.meta, EventMeta)
    assert event.data == {}


def test_event_is_overdue():
    """测试 Event 的过期判断逻辑"""
    # overdue <= 0 应该永不过期
    event1 = Event(meta=EventMeta(overdue=0))
    assert not event1.is_overdue()  # 应该返回 False

    event2 = Event(meta=EventMeta(overdue=-1))
    assert not event2.is_overdue()  # 应该返回 False

    # 新创建的事件，overdue=10秒，应该未过期
    event3 = Event(meta=EventMeta(overdue=10.0))
    assert not event3.is_overdue()

    # 创建已过期的事件（通过修改 created_at）
    old_time = datetime.datetime.now() - datetime.timedelta(seconds=15)
    meta = EventMeta(overdue=5.0)
    meta.created_at = old_time
    event4 = Event(meta=meta)
    assert event4.is_overdue()  # 应该返回 True


def test_event_model_basics():
    """测试 EventModel 基础功能"""
    model = ExampleEventModel(value="test", count=42)
    assert model.value == "test"
    assert model.count == 42
    assert model.event_type() == "test/example"


def test_event_model_from_event():
    """测试从 Event 创建 EventModel"""
    # 有效事件
    event = Event(
        meta=EventMeta(event_type="test/example"),
        data={"value": "from_event", "count": 100}
    )
    model = ExampleEventModel.from_event(event)
    assert model is not None
    assert model.value == "from_event"
    assert model.count == 100

    # 事件类型不匹配
    wrong_event = Event(meta=EventMeta(event_type="wrong/type"))
    model = ExampleEventModel.from_event(wrong_event)
    assert model is None


def test_event_model_to_event():
    """测试 EventModel 转换为 Event"""
    model = ExampleEventModel(value="test_value", count=77)

    event = model.to_event()
    assert event.meta.event_type == "test/example"
    assert event.data["value"] == "test_value"
    assert event.data["count"] == 77

    # 测试自定义参数
    event2 = model.to_event(overdue=30.0, priority=10)
    assert event2.meta.overdue == 30.0
    assert event2.meta.priority == 10


def test_subscriber_mode_type():
    """测试 SubscriberMode 类型"""
    # 应该可以赋值这些值
    mode1: SubscriberMode = 'queue'
    mode2: SubscriberMode = 'priority'

    assert mode1 == 'queue'
    assert mode2 == 'priority'


def test_abstract_classes_cannot_be_instantiated():
    """测试抽象类不能直接实例化"""
    with pytest.raises(TypeError):
        Publisher()

    with pytest.raises(TypeError):
        Subscriber()

    with pytest.raises(TypeError):
        EventBus()


if __name__ == "__main__":
    """直接运行测试以快速验证"""
    # 快速运行主要测试
    test_event_meta_defaults()
    test_event_meta_custom()
    test_event_default()
    test_event_is_overdue()
    test_event_model_basics()
    test_event_model_from_event()
    test_event_model_to_event()

    print("所有基础测试通过!")

    # 特别验证 is_overdue 逻辑
    print("\nis_overdue() 逻辑验证:")
    event = Event(meta=EventMeta(overdue=0))
    result = event.is_overdue()
    print(f"  overdue=0 时 is_overdue() = {result} (预期: False)")
    assert result is False, f"预期 False, 得到 {result}"

    event2 = Event(meta=EventMeta(overdue=-1))
    result2 = event2.is_overdue()
    print(f"  overdue=-1 时 is_overdue() = {result2} (预期: False)")
    assert result2 is False, f"预期 False, 得到 {result2}"