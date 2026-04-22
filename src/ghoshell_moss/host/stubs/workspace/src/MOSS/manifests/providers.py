from ghoshell_moss.host.providers import (
    WorkspaceSessionProvider,
    ZenohTopicServiceProvider,
    WorkspaceLoggerProvider,
    HostEnvZenohProvider,
    HostEnvConfigStoreProvider,
)

moss_session_provider = WorkspaceSessionProvider()

config_store_provider = HostEnvConfigStoreProvider()

zenoh_session_provider = HostEnvZenohProvider()

logger_provider = WorkspaceLoggerProvider()

topic_service_provider = ZenohTopicServiceProvider()
