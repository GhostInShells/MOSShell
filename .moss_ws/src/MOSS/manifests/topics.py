# MOSS Topic manifest.
#
# 事件协议声明：用代码直接做协议声明，约定环境中可通讯的 topic 类型。
# TopicModel 子类本身就是协议声明 — 定义了 topic 的 schema、类型和默认名称。
# 类一旦出现在模块命名空间（import 或直接定义），scan_package 就能通过
# issubclass(obj, TopicModel) 发现，以 topic_name 为键聚合。
#
# 注意：Signal（如 InputSignal）不属于 topic 声明。Signal 是 Mindflow 的
# 输入信号类型，在 nuclei.py 中被 NucleusMeta.signals() 引用，不应在此声明。
#
# 发现路径：MOSS.manifests.topics
# 深入：moss codex get-interface ghoshell_moss.core.concepts.topic:TopicModel
from ghoshell_moss.topics.audio import SubtitleTopic
