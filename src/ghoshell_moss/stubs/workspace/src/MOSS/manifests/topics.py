# MOSS Topic manifest.
#
# 事件协议声明：用代码直接做协议声明，约定环境中可通讯的 topic 类型。
# TopicModel 子类本身就是协议声明 — 定义了 topic 的 schema、类型和默认名称。
# 类一旦出现在模块命名空间（import 或直接定义），scan_package 就能通过
# issubclass(obj, TopicModel) / issubclass(obj, TopicSchema) 发现，以 topic_name 为键聚合。
# 消息协议传输可以用 Topic (弱类型) 而非 TopicModel (强类型) 传递.
#
# 发现路径：MOSS.manifests.topics  (pkg or module)
# 定义项目专属的 Topic, 需要确认 Topic 可以通过环境依赖被读取.
# 相关 Topic 可以声明在 MOSS.manifests.topics.my_topic 等路径下, 使得不使用相同 python 解释器的项目仍然可以发现源码.

from ghoshell_moss.topics import (
    AudioRuntimeTopic, SpeechTopic,
    ErrorTopic,
)
