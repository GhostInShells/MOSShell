# Openbox config manifest — canonical default configuration.
#
# Shipped baseline: ConfigType instances with default values.  Matrix scans via
# isinstance(obj, ConfigType) and registers into ConfigStore via get_or_create().
# Defaults are overridable per-mode/per-workspace via YAML in configs/.
#
# Project extends by:  from ghoshell_moss.matrix.openbox.configs import *
#
# --
# Openbox Config 清单 — 开箱默认配置（canonical 基线）。
# 定义 ConfigType 实例作为默认值，Matrix 扫描自动发现并注册到 ConfigStore。
# 默认值可被 workspace configs/ 下的 YAML 覆盖。

from ghoshell_moss.contracts.llms import LLMConfig
from ghoshell_moss.host.audios.miniaudio_impl.configs import MiniAudioFactoryConfig
from ghoshell_moss.host.providers.tts_service_provider import TTSManagerConfig
from ghoshell_moss.mcp.config import MCPHubConfig

__all__ = [
    'llm_config',
    'tts_config',
    'miniaudio_factory_config',
    'mcp_hub_config',
]

# LLM 配置中心 (conf: llms)
llm_config = LLMConfig()

# TTS 语音合成 (conf: tts_factory)
tts_config = TTSManagerConfig()

# miniaudio 音频设备 (conf: miniaudio_factory) — 播放/采集是同一个 factory 的一对 stream,
# 共享设备几何 (采样率等), 不再各自声明配置.
miniaudio_factory_config = MiniAudioFactoryConfig()

# MCP hub (conf: mcp_hub)
mcp_hub_config = MCPHubConfig()
