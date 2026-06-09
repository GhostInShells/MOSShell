# MOSS IoC Provider manifest.
#
# 声明依赖注入的工厂：每个 Provider 实例声明"这个接口由这个工厂生产"。
# Matrix 启动时通过 scan_package 自动发现 → isinstance(obj, Provider) 过滤
# → 以 contract() 返回值（接口的 Python import path）为键注入 IoC 容器。
#
# 模式约定：定义一个模块级 Provider 实例，变量名自解释。
# mode 的 providers 叠加在全局之上（list.extend），不会覆盖全局的绑定。

from ghoshell_moss.host.providers import (
    HostSessionProvider,
    ZenohTopicServiceProvider,
    HostLoggerProvider,
    HostEnvZenohProvider,
    HostEnvConfigStoreProvider,
)
from ghoshell_moss.host.providers.tts_service_provider import TTSServiceProvider
from ghoshell_moss.host.providers.speech_service_provider import TTSSpeechServiceProvider
from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerProvider
from ghoshell_moss.core.resources.memory_registry import InMemoryResourceRegistryProvider
from ghoshell_moss.host.fractal.zenoh_fractal import ZenohFractalHubProvider, ZenohFractalCellContractProvider

moss_session_provider = HostSessionProvider()

config_store_provider = HostEnvConfigStoreProvider()

zenoh_session_provider = HostEnvZenohProvider()

logger_provider = HostLoggerProvider()

topic_service_provider = ZenohTopicServiceProvider()

# audio player and speech

player_service_provider = AudioPlayerProvider()

tts_service_provider = TTSServiceProvider()

speech_service_provider = TTSSpeechServiceProvider()

resources_provider = InMemoryResourceRegistryProvider()

fractal_hub_provider = ZenohFractalHubProvider()

fractal_node_provider = ZenohFractalCellContractProvider()
