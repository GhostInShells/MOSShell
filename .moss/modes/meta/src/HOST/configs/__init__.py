# Mode Config manifest — mode-specific configuration overrides.
#
# Define ConfigType instances to override global defaults for this mode.
# Matrix combines global (MOSS.manifests.configs) and mode configs at runtime.
# Mode configs are written to configs/meta.{mode}.yml, with fallback to base.
#
# --
# Mode Config 清单 — mode 专属配置覆盖。
# 定义 ConfigType 实例覆盖全局默认值。

from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerConfig
from ghoshell_moss.host.providers.tts_service_provider import TTSManagerConfig
from ghoshell_moss.mcp.config import MCPHubConfig
from ghoshell_moss.contracts.audio import AudioCaptureConfig
from ghoshell_moss.contracts.llms import LLMConfig

# text-to-speech
tts_config = TTSManagerConfig()

# audio player
audio_player_config = AudioPlayerConfig()

# audio capture
audio_capture_config = AudioCaptureConfig()

# MCP hub
mcp_hub_config = MCPHubConfig()

# LLM
llm_config = LLMConfig()
