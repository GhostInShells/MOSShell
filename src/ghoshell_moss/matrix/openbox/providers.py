# Openbox provider manifest — canonical default IoC assembly.
#
# Shipped baseline: module-level Provider instances registered out of the box.
# Matrix scans via isinstance(obj, Provider), deduplicates by id(), then
# registers each into the IoC container.
#
# Project extends by:  from ghoshell_moss.matrix.openbox.providers import *
# Mode    extends by:  from MOSS.manifests.providers import *
#
# --
# Openbox Provider 清单 — 开箱默认 IoC 装配（canonical 基线）。
# 定义模块级 Provider 实例，Matrix 扫描自动发现并注入 IoC 容器。
# Project 通过 from ghoshell_moss.matrix.openbox.providers import * 继承；
# Mode 通过 from MOSS.manifests.providers import * 继承，同名重赋值即覆盖。

from ghoshell_moss.matrix.providers import (
    MatrixZenohSessionProvider,
    ZenohTopicServiceProvider,
    MatrixLoggerProvider,
    ZenohQAManagerProvider,
    SessionWarrantProvider,
)
from ghoshell_moss.project.providers import (
    EnvConfigStoreProvider,
    ProjectSubprocessesProvider,
    ProjectJobSupervisorProvider,
    ProjectLLMFuncsProvider,
    RuntimeErrorLogProvider,
)
from ghoshell_moss.resources.memory_registry import InMemoryResourceRegistryProvider
from ghoshell_moss.host.providers.tts_service_provider import TTSServiceProvider
from ghoshell_moss.host.providers.speech_service_provider import TTSSpeechServiceProvider
from ghoshell_moss.host.audios.miniaudio_impl.factory import MiniAudioFactoryProvider
from ghoshell_moss.host.providers.audio_player_provider import MiniAudioPlayerProvider
from ghoshell_moss.host.providers.audio_capture_provider import MiniAudioCaptureProvider
from ghoshell_moss.host.providers.audio_asr_provider import AudioASRProvider
from ghoshell_moss.host.providers.listener_provider import ListenerProvider

__all__ = [
    'moss_session_provider',
    'config_store_provider',
    'topic_service_provider',
    'logger_provider',
    'runtime_error_log_provider',
    'resources_provider',
    'qa_manager_provider',
    'warrant_provider',
    'subprocess_provider',
    'job_supervisor_provider',
    'llm_funcs_provider',
    'tts_service_provider',
    'speech_service_provider',
    'miniaudio_factory_provider',
    'player_service_provider',
    'audio_capture_provider',
    'asr_provider',
    'listener_provider',
]

# zenoh session provider
moss_session_provider = MatrixZenohSessionProvider()

# file-based config store
config_store_provider = EnvConfigStoreProvider()

# zenoh topic system
topic_service_provider = ZenohTopicServiceProvider()

# workspace logger — returns moss root logger with TimedRotatingFileHandler
logger_provider = MatrixLoggerProvider()

# runtime error collector — bounded ERROR+ tail pullable by the model
runtime_error_log_provider = RuntimeErrorLogProvider()

# in-memory resource registry
resources_provider = InMemoryResourceRegistryProvider()

# zenoh QA exchange (cross-process ask/answer)
qa_manager_provider = ZenohQAManagerProvider()

# warrant authorization (storage + QA, fail-open optional capability)
warrant_provider = SessionWarrantProvider()

subprocess_provider = ProjectSubprocessesProvider()

job_supervisor_provider = ProjectJobSupervisorProvider()

# llm func engine — lazy: pydantic-ai imports only on first fetch
llm_funcs_provider = ProjectLLMFuncsProvider()

# -- 音频/语音基线能力 (实现留 host 作为依赖路径, 模块顶层无重 import) -- #

# text-to-speech
tts_service_provider = TTSServiceProvider()

# speech service
speech_service_provider = TTSSpeechServiceProvider()

# miniaudio 设备 factory — 播放 + 采集一对 stream, AEC 在此装线.
# 两个 provider 都向它取 stream; 它不在时才会降级到各自的构造路径.
miniaudio_factory_provider = MiniAudioFactoryProvider()

# audio player
player_service_provider = MiniAudioPlayerProvider()

# audio capture source
audio_capture_provider = MiniAudioCaptureProvider()

# asr (speech recognition)
asr_provider = AudioASRProvider()

# listener (耳朵 — 缝合 capture + asr)
listener_provider = ListenerProvider()
