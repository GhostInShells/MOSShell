from typing import Callable
from ghoshell_moss.core.concepts.channel import Channel
from threading import Thread, Event
import asyncio

__all__ = ['CancelFunc', 'provide_as_thread', 'provide_as_future', 'provide_until_closed']

CancelFunc = Callable[[], None]
'''cancel the provider '''


def provide_as_thread(channel: Channel) -> tuple[Thread, CancelFunc]:
    """
    Provide the channel into the main process of MOSS.
    In this process, the channel is running in a sub thread.
    """
    pass


def provide_until_closed(channel: Channel, cancel: Event | None = None) -> None:
    """
    Provide the channel into the main process of MOSS.
    This method will block the thread, and run until the channel is closed.
    Send a threading.Event to make it cancelable outside.
    """
    pass


def provide_as_future(channel: Channel, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Future[None]:
    """
    Provide the channel into the main process of MOSS.
    Will Async run in asyncio loop.
    Return a Future that is cancelable.
    """
    pass
