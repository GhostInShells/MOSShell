from typing import Iterable, Type

from ghoshell_moss.contracts.speech import Speech, TTS, StreamAudioPlayer, TTSSpeech
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.speech import BaseTTSSpeech
from ghoshell_container import IoCContainer, Provider

__all__ = ['TTSSpeechServiceProvider']


class TTSSpeechServiceProvider(Provider[Speech]):

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type]:
        yield TTSSpeech

    def factory(self, con: IoCContainer) -> TTSSpeech:
        logger = con.force_fetch(LoggerItf)
        player = con.force_fetch(StreamAudioPlayer)
        tts = con.force_fetch(TTS)
        return BaseTTSSpeech(
            logger=logger,
            player=player,
            tts=tts,
        )
