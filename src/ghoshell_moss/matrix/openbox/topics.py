# Openbox topic manifest — canonical default topic declarations.
#
# Shipped baseline: TopicModel subclasses declaring typed events.  Matrix scans
# via issubclass(obj, TopicModel) and converts to TopicSchema via topic_schema().
#
# Declare a topic here only when something actually publishes it — a registered
# schema is a promise that the name resolves cross-process.
#
# Project extends by:  from ghoshell_moss.matrix.openbox.topics import *
#
# --
# Openbox Topic 清单 — 开箱默认 topic 声明（canonical 基线）。
# TopicModel 子类声明类型化事件，Matrix 扫描自动发现。
# 只声明真有生产者的 topic —— 注册即承诺该名字可跨进程解析。

from ghoshell_moss.types.topics import (
    AudioSampleTopic,
    ClauseTopic,
    FaceTopic,
)

__all__ = [
    'AudioSampleTopic',
    'ClauseTopic',
    'FaceTopic',
]
