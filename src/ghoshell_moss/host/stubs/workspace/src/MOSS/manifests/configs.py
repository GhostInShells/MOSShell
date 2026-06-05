# MOSS Config manifest.
#
# 配置声明有两种独立机制：
#
# 1. ConfigType 类 → 文件持久化（schema + 文件路径绑定）
#    ConfigType 子类定义了配置的 schema。实例化后通过 ConfigStore.get_or_create()
#    读取 workspace/configs/{conf_name}.yml。文件存在则从文件读，不存在则用传入
#    实例做默认值写入文件。这是"文件优先"的持久化配置。
#
# 2. Config 实例 → 内存覆盖（不写文件）
#    ConfigStore.set_config(conf, override=False) 只更新内存缓存，不写磁盘。
#    mode 可以用此机制创建 mode 专属的覆盖值，而不修改全局配置文件。
#    mode 可通过显式继承 + 覆盖同键实现配置定制。
#
# 模式约定：
#   - 全局 manifests/configs.py：实例化 ConfigType，作为默认 schema + 默认值
#   - mode configs.py：只实例化需要覆盖的配置，运行时 set_config 内存覆盖
#   - 每个 ConfigType 子类必须实现 conf_name() 返回唯一标识（即文件名）

from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerConfig
from ghoshell_moss.host.providers.tts_service_provider import TTSManagerConfig

tts_config = TTSManagerConfig()

audio_player_config = AudioPlayerConfig()
