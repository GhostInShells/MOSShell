from ghoshell_moss.core.duplex import DuplexChannelProvider
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_container import Container, IoCContainer, get_container
from ghoshell_common.contracts import LoggerItf
from .child_connection import ChildProcessConnection
import logging


class ChildProcessProvider(DuplexChannelProvider):

    def __init__(
            self,
            container: IoCContainer | None = None,
            receive_interval_seconds: float = 0.5,
    ):
        _container = container or get_container()
        connection = ChildProcessConnection(
            logger=_container.get(LoggerItf) or logging.getLogger('moss')
        )
        super().__init__(
            connection,
            container=_container,
            receive_interval_seconds=receive_interval_seconds,
        )


def create_child_process_provider(
        container: IoCContainer | None = None,
        *,
        receive_interval_seconds: float = 0.5,
) -> ChildProcessProvider:
    return ChildProcessProvider(container, receive_interval_seconds=receive_interval_seconds)


async def run_channel(
        channel: Channel,
        *,
        container: IoCContainer | None = None,
        receive_interval_seconds: float = 0.5,
) -> None:
    provider = create_child_process_provider(container, receive_interval_seconds=receive_interval_seconds)
    try:
        await provider.arun_until_closed(channel)
    except Exception as e:
        await provider.aclose()
        raise e


def run(
        channel: Channel,
        *,
        container: IoCContainer | None = None,
        receive_interval_seconds: float = 0.5,
) -> None:
    pass
