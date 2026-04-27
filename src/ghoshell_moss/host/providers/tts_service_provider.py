from typing import Literal
from ghoshell_moss.contracts.speech import TTS
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.configs import ConfigType, ConfigStore
from ghoshell_moss.core.speech.volcengine_tts import VolcengineTTSConf, VolcengineTTS
from ghoshell_container import IoCContainer, Provider, INSTANCE
from pydantic import Field

__all__ = ['TTSServiceProvider']


class TTSManagerConfig(ConfigType):
    """
    tts manager config
    """
    use: Literal['volcengine_stream_tts_model'] = Field(
        default='volcengine_stream_tts_model',
        description='which driver to use',
    )

    volcengine_stream_tts_model_config: VolcengineTTSConf = Field(
        default_factory=VolcengineTTSConf,
        description="volc engine tts config"
    )

    @classmethod
    def conf_name(cls) -> str:
        return 'tts_factory'


class TTSServiceProvider(Provider[TTS]):
    """tts service provider"""

    def singleton(self) -> bool:
        return False

    def factory(self, con: IoCContainer) -> INSTANCE:
        store = con.force_fetch(ConfigStore)
        manager_conf = store.get_or_create(TTSManagerConfig())

        if manager_conf.use == 'volcengine_stream_tts_model':
            return self._factory_volcengine_stream_tts_model(
                con,
                manager_conf.volcengine_stream_tts_model_config,
            )
        else:
            raise NotImplementedError(f"{manager_conf.use} not implemented")

    def _factory_volcengine_stream_tts_model(
            self,
            con: IoCContainer,
            conf: VolcengineTTSConf,
    ) -> TTS:
        logger = con.force_fetch(LoggerItf)
        return VolcengineTTS(
            conf=conf,
            logger=logger,
        )
