from typing import Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.audio import AudioCaptureSource, AudioCaptureConfig
from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.workspace import Workspace

__all__ = ["AudioCaptureProvider"]


class AudioCaptureProvider(Provider[AudioCaptureSource]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[AudioCaptureSource]:
        return AudioCaptureSource

    def factory(self, con: IoCContainer) -> AudioCaptureSource:
        store = con.force_fetch(ConfigStore)
        conf = store.get_or_create(AudioCaptureConfig())
        workspace = con.force_fetch(Workspace)
        logger = con.force_fetch(LoggerItf)

        from ghoshell_moss.host.listener.capture.miniaudio_capture import MiniAudioCaptureSource

        return MiniAudioCaptureSource(config=conf, workspace=workspace, logger=logger)
