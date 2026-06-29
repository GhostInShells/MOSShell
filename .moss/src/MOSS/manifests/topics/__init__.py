# Topic manifest — event topic protocol declarations.
#
# Define TopicModel subclasses to declare typed topics.  Matrix scans via
# issubclass(obj, TopicModel), converts each to TopicSchema via topic_schema().
# Message transport can use Topic (weakly-typed) while declaration uses TopicModel.
#
# Mode extends by: from MOSS.manifests.topics import *
#
# --
# Topic 清单 — 事件协议声明。
# 用 TopicModel 子类声明类型化事件，Matrix 扫描自动发现。

from ghoshell_moss.topics import (
    AudioRuntimeTopic,
    SpeechTopic,
    ErrorTopic,
)
