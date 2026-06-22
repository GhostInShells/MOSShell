"""字幕 Topic 发布配置。

Ghost 进程 SpeechServiceProvider 读取此配置，决定是否启用 Zenoh Topic
总线字幕发布（替代旧 HTTP 旁路）。

过渡策略：enable_topic 默认 False（保持向后兼容，走旧 HTTP 旁路）。
设置 True 后走 Topic 总线路径。
"""

from pydantic import Field
from ghoshell_moss.contracts.configs import ConfigType

__all__ = ["SubtitleTopicConfig"]


class SubtitleTopicConfig(ConfigType):
    """字幕 Topic 发布配置。

    Ghost 进程 SpeechServiceProvider.factory() 读取此配置：
    - enable_topic=False（默认）：不注入字幕回调，字幕功能禁用
    - enable_topic=True：从容器获取 TopicService，创建 pub 闭包注入给 BaseTTSSpeech
    """

    enable_topic: bool = Field(
        default=False,
        description="是否启用 Topic 总线字幕发布。True 时 SpeechServiceProvider 创建 Topic 发布闭包",
    )
    topic_path: str = Field(
        default="moshi/subtitle",
        description="字幕发布的 Topic 路径。消费端（Reflex）需订阅同一路径",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "subtitle_topic"
