from ghoshell_moss import ChannelProvider
from ghoshell_moss.core.concepts.channel import ChannelProxy
from ghoshell_moss.core.duplex import BridgeTestSuite
from ghoshell_common.helpers import uuid
import zenoh
from ._provider import ZenohChannelProvider
from ._proxy import ZenohProxyChannel
import time

__all__ = ["ZenohBridgeTestSuite"]


class ZenohBridgeTestSuite(BridgeTestSuite):

    def __init__(self):
        self._session: zenoh.Session | None = None

    def create(self, proxy_name: str = "proxy") -> tuple[ChannelProvider, ChannelProxy]:
        self._session = zenoh.open(zenoh.Config())
        node_name = "test/zenoh"
        session_id = uuid()
        provider = ZenohChannelProvider(session=self._session, node_name=node_name, session_id=session_id)
        proxy = ZenohProxyChannel(
            name=proxy_name,
            description="",
            session=self._session, node_name=node_name, session_id=session_id,
        )
        return provider, proxy

    def cleanup(self) -> None:
        if self._session is not None and not self._session.is_closed():
            self._session.close()
