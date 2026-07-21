# 事件协议声明 — 约束可通讯的 topic 类型。
# 在此文件中定义 TopicModel 或 TopicSchema 子类，Matrix 启动时自动发现。
#
# 显式继承全局 topics，然后追加 mode 专属的：
from MOSS.manifests.topics import *  # noqa: F403
