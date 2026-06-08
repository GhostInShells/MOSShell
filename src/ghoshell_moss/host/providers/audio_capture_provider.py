from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.host.speech.capture.miniaudio_capture import MiniAudioCaptureSource
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport

__all__ = ["AudioCaptureProvider"]


class AudioCaptureProvider(Provider[AudioCaptureSource]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> AudioCaptureSource:
        from ghoshell_moss.contracts.audio import AudioCaptureConfig
        from ghoshell_moss.core.blueprint.matrix import Matrix

        matrix = con.force_fetch(Matrix)
        store = con.force_fetch(ConfigStore)
        config = store.get_or_create(AudioCaptureConfig())
        transport = MatrixAudioTransport(matrix=matrix)
        return MiniAudioCaptureSource(transport=transport, config=config)
