from ghoshell_moss.host.providers import (
    WorkspaceSessionProvider,
    ZenohTopicServiceProvider,
    WorkspaceLoggerProvider,
    HostEnvZenohProvider,
    HostEnvConfigStoreProvider,
)
from ghoshell_moss.host.providers.tts_service_provider import TTSServiceProvider
from ghoshell_moss.host.providers.speech_service_provider import TTSSpeechServiceProvider
from ghoshell_moss.host.providers.audio_player_provider import PyAudioPlayerProvider

moss_session_provider = WorkspaceSessionProvider()

config_store_provider = HostEnvConfigStoreProvider()

zenoh_session_provider = HostEnvZenohProvider()

logger_provider = WorkspaceLoggerProvider()

topic_service_provider = ZenohTopicServiceProvider()

# audio player and speech

player_service_provider = PyAudioPlayerProvider()

tts_service_provider = TTSServiceProvider()

speech_service_provider = TTSSpeechServiceProvider()
