# 配置模型声明 — 显式继承全局 configs。
# 全局的类注册（Type[ConfigType]）通过 import * 继承，保持文件持久化语义。
# 如需运行时覆盖（仅内存），在此处实例化带值的 ConfigType：
#   tts_override = TTSManagerConfig(default_speaker="大壹")  # is_override=True
from MOSS.manifests.configs import *  # noqa: F403
from ghoshell_moss.channels.web_bookmark import WebConfig

# ── 字幕 Topic 总线：替代旧 HTTP 旁路 ──
# 启用后 Ghost 进程 SpeechServiceProvider 通过 Zenoh 发布 SubtitleTopic，
# Reflex 进程通过 TopicWindow 订阅后渲染为 SSE 字幕。
subtitle_topic_config = SubtitleTopicConfig(enable_topic=True, topic_path="moshi/subtitle")
