# IoC Provider manifest — declare "this interface is produced by this factory".
#
# Define module-level Provider instances.  Matrix scans via isinstance(obj, Provider),
# deduplicates by id(), then registers each into the IoC container.
#
# Mode extends by: from MOSS.manifests.providers import *
#
# --
# IoC Provider 声明 — 声明依赖注入工厂。
# 定义模块级 Provider 实例，Matrix 扫描自动发现并注入 IoC 容器。
# Mode 通过 from MOSS.manifests.providers import * 继承全局。

from ghoshell_moss.matrix.providers import (
    MatrixZenohSessionProvider,
    ZenohTopicServiceProvider,
    MatrixLoggerProvider,
    ZenohQAManagerProvider,
)
from ghoshell_moss.project.providers import (
    EnvConfigStoreProvider,
    ProjectSubprocessesProvider,
    ProjectJobSupervisorProvider,
)

from ghoshell_moss.core.resources.memory_registry import InMemoryResourceRegistryProvider
from ghoshell_moss.host.providers.tts_service_provider import TTSServiceProvider
from ghoshell_moss.host.providers.speech_service_provider import TTSSpeechServiceProvider
from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerProvider
from ghoshell_moss.host.providers.audio_capture_provider import AudioCaptureProvider
from ghoshell_moss.host.providers.audio_asr_provider import AudioASRProvider

# zenoh session provider
moss_session_provider = MatrixZenohSessionProvider()

# file-based config store
config_store_provider = EnvConfigStoreProvider()

# zenoh topic system
topic_service_provider = ZenohTopicServiceProvider()

# workspace logger — returns moss root logger with TimedRotatingFileHandler
logger_provider = MatrixLoggerProvider()

# in-memory resource registry
resources_provider = InMemoryResourceRegistryProvider()

# zenoh QA exchange (cross-process ask/answer)
qa_manager_provider = ZenohQAManagerProvider()

subprocess_provider = ProjectSubprocessesProvider()

job_supervisor_provider = ProjectJobSupervisorProvider()

# -- 音频/语音基线能力 (实现留 host 作为依赖路径, 模块顶层无重 import) -- #

# text-to-speech
tts_service_provider = TTSServiceProvider()

# speech service
speech_service_provider = TTSSpeechServiceProvider()

# audio player
player_service_provider = AudioPlayerProvider()

# audio capture source
audio_capture_provider = AudioCaptureProvider()

# asr (speech recognition)
asr_provider = AudioASRProvider()
