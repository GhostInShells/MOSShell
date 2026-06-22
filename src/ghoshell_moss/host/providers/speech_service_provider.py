from ghoshell_moss.contracts.speech import Speech, TTS, StreamAudioPlayer
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.speech import BaseTTSSpeech
from ghoshell_moss.core.speech.subtitle_config import SubtitleTopicConfig
from ghoshell_moss.core.concepts.topic import TopicName
from ghoshell_moss.topics.audio import SubtitleTopic
from ghoshell_container import IoCContainer, Provider, INSTANCE

__all__ = ['TTSSpeechServiceProvider']


class TTSSpeechServiceProvider(Provider[Speech]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        logger = con.force_fetch(LoggerItf)
        player = con.force_fetch(StreamAudioPlayer)
        tts = con.force_fetch(TTS)

        # ── 字幕 Topic 总线回调（可选，由 SubtitleTopicConfig 控制）──
        subtitle_callback = None
        try:
            from ghoshell_moss.contracts.configs import ConfigStore
            from ghoshell_moss.core.concepts.topic import TopicService
            config_store = con.force_fetch(ConfigStore)
            subtitle_config = config_store.get(SubtitleTopicConfig)
            logger.info("[speech] SubtitleTopicConfig from ConfigStore: enable_topic=%s, topic_path=%s",
                        subtitle_config.enable_topic, subtitle_config.topic_path)
        except Exception as e:
            logger.warning("[speech] ConfigStore.get(SubtitleTopicConfig) 异常: %s (%s)", e, type(e).__name__)
            subtitle_config = None

        # ConfigStore 未命中 mode 覆盖时，直接从 mode config 模块回退
        if subtitle_config is None or not subtitle_config.enable_topic:
            import importlib
            from ghoshell_moss.core.blueprint.environment import Environment
            mode_name = Environment.discover().moss_mode_name
            if mode_name:
                try:
                    mode_configs = importlib.import_module(f"MOSS.modes.{mode_name}.configs")
                    _stc = getattr(mode_configs, "subtitle_topic_config", None)
                    if isinstance(_stc, SubtitleTopicConfig):
                        subtitle_config = _stc
                        logger.info("[speech] SubtitleTopicConfig 从 mode %s 回退导入: enable_topic=%s",
                                    mode_name, _stc.enable_topic)
                except ImportError:
                    pass

        if subtitle_config is not None and subtitle_config.enable_topic:
            topic_service = con.force_fetch(TopicService)
            topic_name = TopicName(subtitle_config.topic_path)

            def _publish_subtitle(text: str, is_final: bool, batch_id: str = "") -> None:
                """句级字幕发布闭包：组装 SubtitleTopic → pub 到 Zenoh 总线。

                由 TTSSpeechStream._play_loop() 在音频播放线程中逐句调用。
                TopicService.pub() 内部通过 run_in_executor 安全投递到 Zenoh。
                """
                try:
                    topic_service.pub(
                        SubtitleTopic(text=text, is_final=is_final, batch_id=batch_id),
                        name=str(topic_name),
                    )
                except Exception:
                    pass  # 字幕丢失非关键故障，与旧 HTTP 旁路行为一致

            subtitle_callback = _publish_subtitle

        return BaseTTSSpeech(
            logger=logger,
            player=player,
            tts=tts,
            subtitle_callback=subtitle_callback,
        )
