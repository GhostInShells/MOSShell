# 配置模型声明 — 显式继承全局 configs。
# 全局的类注册（Type[ConfigType]）通过 import * 继承，保持文件持久化语义。
# 如需运行时覆盖（仅内存），在此处实例化带值的 ConfigType：
#   tts_override = TTSManagerConfig(default_speaker="大壹")  # is_override=True
from MOSS.manifests.configs import *  # noqa: F403
