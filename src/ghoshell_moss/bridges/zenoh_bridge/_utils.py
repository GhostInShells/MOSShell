from typing import ClassVar, Protocol, runtime_checkable

__all__ = ["BridgeExpr", "ChannelBridgeNodeExpr", "HubKeyExpr"]


@runtime_checkable
class BridgeExpr(Protocol):
    """Channel bridge key expression 的接口契约。

    Provider/Proxy/Hub 通过此 Protocol 感知 key 体系，
    不依赖具体实现的类继承。
    """

    address: str
    bridge_prefix: str
    provider_liveness_key: str
    proxy_liveness_key: str
    provider_receiver_key: str
    proxy_receiver_key: str
    proxy_liveness_prefix: str
    provider_liveness_prefix: str


class HubKeyExpr:

    def __init__(self, namespace: str):
        self.prefix = namespace

    def new_expr(self, address: str) -> BridgeExpr:
        return ChannelBridgeNodeExpr(prefix=self.prefix, address=address)


class ChannelBridgeNodeExpr(BridgeExpr):
    """
    基线 BridgeExpr 实现。约定优先于配置。
    """

    PROVIDER_LIVENESS_KEY: ClassVar[str] = "provider_liveness"
    PROXY_LIVENESS_KEY: ClassVar[str] = "proxy_liveness"
    PROVIDER_RECEIVER: ClassVar[str] = "provider"
    PROXY_RECEIVER: ClassVar[str] = "proxy"

    def __init__(
            self,
            prefix: str,
            address: str,
    ):
        self.bridge_prefix = prefix
        self.address = address
        self.proxy_liveness_prefix: str = "/".join([self.bridge_prefix, self.PROXY_LIVENESS_KEY])
        self.provider_liveness_prefix: str = "/".join([self.bridge_prefix, self.PROVIDER_LIVENESS_KEY])
        self.provider_liveness_key: str = "/".join([self.provider_liveness_prefix, address])
        self.proxy_liveness_key: str = "/".join([self.proxy_liveness_prefix, address])

        self.provider_receiver_key: str = "/".join([self.bridge_prefix, self.PROVIDER_RECEIVER, address])
        '''proxy send to provider'''
        self.proxy_receiver_key: str = "/".join([self.bridge_prefix, self.PROXY_RECEIVER, address])
        '''provider send to proxy'''
