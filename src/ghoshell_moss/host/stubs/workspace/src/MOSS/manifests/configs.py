# MOSS Config manifest — 全局默认配置注册。
#
# 两种注册语义（由 scan 逻辑自动区分）：
#
# 1. Type[ConfigType] — 模块级类引用 → 系统默认，文件持久化。
#    Bootstrap 时 get_or_create()：读取 workspace/configs/{conf_name}.yml，
#    文件存在则读，不存在则以类的默认构造写入文件。
#
# 2. ConfigType 实例 — 模块级实例变量 → 运行时覆盖，仅内存。
#    Bootstrap 时 set_config(override=False)：写入内存缓存，绝不触碰 YAML 文件。
#    mode 用此语义覆盖全局默认值（如切换默认音色），不会污染 workspace/configs/。
#
# 约定：
#   - 全局 manifests/configs.py：放类（类型注册），不在此处实例化带值的覆盖
#   - mode configs.py：通过 from MOSS.manifests.configs import * 继承类，
#     然后实例化需要覆盖的配置（相同 conf_name() → is_override=True）
#   - 每个 ConfigType 子类必须实现 conf_name() 返回唯一标识（即文件名）

from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerConfig
from ghoshell_moss.host.providers.tts_service_provider import TTSManagerConfig
from ghoshell_moss.channels.mcp_hub import MCPHubConfig
from ghoshell_moss.contracts.audio import AudioCaptureConfig

tts_config = TTSManagerConfig()

audio_player_config = AudioPlayerConfig()

audio_capture_config = AudioCaptureConfig()

mcp_hub_config = MCPHubConfig()
