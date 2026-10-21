from typing import Literal
from ghoshell_moss.contracts.speech import TTS
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.configs import ConfigType, ConfigStore
from ghoshell_moss.host.speech.volcengine_tts.config import VolcengineTTSConf
from ghoshell_moss.host.speech.mimo_tts.config import MiMoTTSConf
from ghoshell_container import IoCContainer, Provider, INSTANCE
from pydantic import Field

__all__ = ['TTSServiceProvider', 'TTSManagerConfig']


class TTSManagerConfig(ConfigType):
    """
    tts manager config
    """
    use: Literal['volcengine_stream_tts_model', 'mimo_tts'] = Field(
        default='volcengine_stream_tts_model',
        description='which driver to use',
    )

    volcengine_stream_tts_model_config: VolcengineTTSConf = Field(
        default_factory=VolcengineTTSConf,
        description="volc engine tts config"
    )

    mimo_tts_config: MiMoTTSConf = Field(
        default_factory=MiMoTTSConf,
        description="mi mo tts config"
    )

    @classmethod
    def conf_name(cls) -> str:
        return 'tts_factory'

    def validate_configuration(self) -> None:
        """resolved 后自校验: 至少一条可用凭据 (env 缺失时 resolve 保留 $VAR 占位)."""
        if self.use != 'volcengine_stream_tts_model':
            return
        conf = self.volcengine_stream_tts_model_config

        def usable(value: str) -> bool:
            return bool(value) and not value.startswith("$")

        # 与 gen_header 选路一致: 优先新控制台 api_key; 缺则要求旧版 app_key+access_token 成对.
        if usable(conf.api_key):
            return
        legacy_missing = [
            name for name, value in (
                ("app_key", conf.app_key),
                ("access_token", conf.access_token),
            ) if not usable(value)
        ]
        if legacy_missing:
            raise ValueError(
                f"Volcengine TTS env not set: SEED_API_KEY "
                f"(legacy missing: {', '.join(legacy_missing)})"
            )


class TTSServiceProvider(Provider[TTS]):
    """tts service provider"""

    def singleton(self) -> bool:
        return False

    def factory(self, con: IoCContainer) -> INSTANCE:
        store = con.force_fetch(ConfigStore)
        manager_conf = store.get_or_create(TTSManagerConfig())
        manager_conf.validate_configuration()

        if manager_conf.use == 'volcengine_stream_tts_model':
            return self._factory_volcengine_stream_tts_model(
                con,
                manager_conf.volcengine_stream_tts_model_config,
            )
        elif manager_conf.use == 'mimo_tts':
            return self._factory_mimo_tts(
                con,
                manager_conf.mimo_tts_config,
            )
        else:
            raise NotImplementedError(f"{manager_conf.use} not implemented")

    def _factory_volcengine_stream_tts_model(
            self,
            con: IoCContainer,
            conf: VolcengineTTSConf,
    ) -> TTS:
        from ghoshell_moss.host.speech.volcengine_tts.tts import VolcengineTTS
        logger = con.force_fetch(LoggerItf)
        return VolcengineTTS(
            conf=conf,
            logger=logger,
        )

    def _factory_mimo_tts(
            self,
            con: IoCContainer,
            conf: MiMoTTSConf,
    ) -> TTS:
        from ghoshell_moss.host.speech.mimo_tts.tts import MiMoTTS
        logger = con.force_fetch(LoggerItf)
        return MiMoTTS(
            conf=conf,
            logger=logger,
        )
