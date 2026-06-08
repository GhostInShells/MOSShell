from ghoshell_moss.contracts.speech import Speech, TTS, StreamAudioPlayer
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.speech import BaseTTSSpeech
from ghoshell_container import IoCContainer, Provider, INSTANCE

__all__ = ['TTSSpeechServiceProvider']


class TTSSpeechServiceProvider(Provider[Speech]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        logger = con.force_fetch(LoggerItf)
        player = con.force_fetch(StreamAudioPlayer)
        tts = con.force_fetch(TTS)
        topic_service = con.force_fetch(TopicService)
        matrix = con.force_fetch(Matrix)
        ghost_name = matrix.ghost_name
        speaker_name = ghost_name if ghost_name and ghost_name != 'None' else "Ghost"
        return BaseTTSSpeech(
            logger=logger,
            player=player,
            tts=tts,
            topic_service=topic_service,
            speaker_name=speaker_name,
        )
