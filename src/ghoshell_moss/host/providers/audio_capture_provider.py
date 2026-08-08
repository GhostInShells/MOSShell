from ghoshell_moss.contracts.audio import AudioCaptureSource, AudioCaptureConfig
from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_container import IoCContainer, Provider
from typing import Type

__all__ = ["AudioCaptureProvider"]


class AudioCaptureProvider(Provider[AudioCaptureSource]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[AudioCaptureSource]:
        return AudioCaptureSource

    def factory(self, con: IoCContainer) -> AudioCaptureSource:
        store = con.force_fetch(ConfigStore)
        conf = store.get_or_create(AudioCaptureConfig())
        matrix = con.force_fetch(Matrix)

        from ghoshell_moss.host.listener.capture.matrix_audio_transport import MatrixAudioTransport
        from ghoshell_moss.host.listener.capture.miniaudio_capture import MiniAudioCaptureSource

        transport = MatrixAudioTransport(matrix=matrix)
        return MiniAudioCaptureSource(transport=transport, config=conf)
