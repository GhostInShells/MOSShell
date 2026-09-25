from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.host.audios.miniaudio_impl.contract import AbstractMiniAudioFactory

__all__ = ['MiniAudioFactoryProvider']


class MiniAudioFactoryProvider(Provider):

    def singleton(self) -> bool:
        return True

    def contract(self):
        return AbstractMiniAudioFactory

    def factory(self, con: IoCContainer):
        from ghoshell_moss.host.audios.miniaudio_impl.configs import MiniAudioFactoryConfig
        from ghoshell_moss.host.audios.miniaudio_impl.factory import MiniAudioFactory
        store = con.force_fetch(ConfigStore)
        conf = store.get_or_create(MiniAudioFactoryConfig())
        workspace = con.force_fetch(Workspace)
        logger = con.force_fetch(LoggerItf)
        factory = MiniAudioFactory(config=conf, workspace=workspace, logger=logger)
        con.add_shutdown(factory.shutdown)
        return factory
