from ghoshell_moss.contracts.speech import Speech, TTS, StreamAudioPlayer
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.speech import BaseTTSSpeech
<<<<<<< Updated upstream
=======
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.host.speech.speech_event_publisher import SpeechEventPublisher
from ghoshell_moss.topics.audio import SpeechStreamingTopic
>>>>>>> Stashed changes
from ghoshell_container import IoCContainer, Provider, INSTANCE

__all__ = ['TTSSpeechServiceProvider']


class TTSSpeechServiceProvider(Provider[Speech]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        logger = con.force_fetch(LoggerItf)
        player = con.force_fetch(StreamAudioPlayer)
        tts = con.force_fetch(TTS)
<<<<<<< Updated upstream
=======
        topic_service = con.force_fetch(TopicService)

        # ── 通用 Speech Streaming Topic 发布 ──
        # 对标 MiniAudioStreamPlayer → AudioTransport.pub_topic(AudioRuntimeTopic)，
        # TTS 引擎始终发布 SpeechStreamingTopic，不询问配置。消费者自行决定订阅。
        publisher = SpeechEventPublisher(
            topic_service=topic_service,
            topic_name=SpeechStreamingTopic.default_topic_name(),
            logger=logger,
        )

        def _publish_streaming(topic: SpeechStreamingTopic) -> None:
            """句级流式发布闭包：由 TTSSpeechStream._play_loop() 逐句调用。"""
            publisher.pub(topic)

>>>>>>> Stashed changes
        return BaseTTSSpeech(
            logger=logger,
            player=player,
            tts=tts,
<<<<<<< Updated upstream
=======
            streaming_callback=_publish_streaming,
>>>>>>> Stashed changes
        )
