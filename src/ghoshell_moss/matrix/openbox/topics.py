# Openbox topic manifest — canonical default topic declarations.
#
# Shipped baseline: TopicModel subclasses declaring typed events.  Matrix scans
# via issubclass(obj, TopicModel) and converts to TopicSchema via topic_schema().
#
# Project extends by:  from ghoshell_moss.matrix.openbox.topics import *
#
# --
# Openbox Topic 清单 — 开箱默认 topic 声明（canonical 基线）。
# TopicModel 子类声明类型化事件，Matrix 扫描自动发现。

from ghoshell_moss.types.audio import (
    ConversationTopic,
    AudioPlaybackTopic,
)
from ghoshell_moss.core.concepts.topic import (
    ErrorTopic,
)

__all__ = [
    'ConversationTopic',
    'AudioPlaybackTopic',
    'ErrorTopic',
]
