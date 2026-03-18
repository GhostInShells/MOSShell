from ghoshell_ghost.contracts.configs import ConfigType
from ghoshell_moss.speech.volcengine_tts import VolcengineTTSConf


class AtomVolcengineTTSConfig(VolcengineTTSConf, ConfigType):
    """
    火山引擎流式 TTS 大模型配置项.
    """

    @classmethod
    def conf_name(cls) -> str:
        return "volcengine_tts"
