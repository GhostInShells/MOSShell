from typing import ClassVar

__all__ = ["NodeChannelBridgeExpr"]


class NodeChannelBridgeExpr:
    """
    定义基于 Node 概念的 Channel 通道.
    假设 Channel 的通讯是基于 Node 的.
    """

    NODE_BRIDGE_PREFIX_TEMPLATE: ClassVar[str] = "MOSS/{session_scope}/node/{address}/channel_bridge"

    PROVIDER_LIVENESS_KEY: ClassVar[str] = "provider_liveness"
    PROXY_LIVENESS_KEY: ClassVar[str] = "proxy_liveness"
    PROVIDER_RECEIVER: ClassVar[str] = "provider"
    PROXY_RECEIVER: ClassVar[str] = "proxy"

    def __init__(self, address: str, session_scope: str):
        self.address = address
        self.session_scope = session_scope
        self.bridge_prefix = self.NODE_BRIDGE_PREFIX_TEMPLATE.format(
            session_scope=self.session_scope,
            address=self.address,
        )
        self.provider_liveness_key: str = "/".join([self.bridge_prefix, self.PROVIDER_LIVENESS_KEY])
        self.proxy_liveness_key: str = "/".join([self.bridge_prefix, self.PROXY_LIVENESS_KEY])

        self.provider_receiver_key: str = "/".join([self.bridge_prefix, self.PROVIDER_RECEIVER])
        '''proxy send to provider'''

        self.proxy_receiver_key: str = "/".join([self.bridge_prefix, self.PROXY_RECEIVER])
        '''provider send to proxy'''
