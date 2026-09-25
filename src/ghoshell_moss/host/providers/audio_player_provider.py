from ghoshell_moss.contracts.speech import StreamAudioPlayer
from ghoshell_container import IoCContainer, Provider

__all__ = ["MiniAudioPlayerProvider"]


class MiniAudioPlayerProvider(Provider[StreamAudioPlayer]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> StreamAudioPlayer:
        from ghoshell_moss.host.audios.miniaudio_impl.contract import AbstractMiniAudioFactory
        factory = con.force_fetch(AbstractMiniAudioFactory)
        return factory.player()
