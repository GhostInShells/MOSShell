from typing import Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.audio import AudioCaptureSource

__all__ = ["MiniAudioCaptureProvider"]


class MiniAudioCaptureProvider(Provider[AudioCaptureSource]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[AudioCaptureSource]:
        return AudioCaptureSource

    def factory(self, con: IoCContainer) -> AudioCaptureSource:
        from ghoshell_moss.host.audios.miniaudio_impl.contract import AbstractMiniAudioFactory
        factory = con.force_fetch(AbstractMiniAudioFactory)
        return factory.capture()
