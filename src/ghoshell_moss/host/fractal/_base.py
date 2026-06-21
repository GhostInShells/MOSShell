__all__ = [
    'FractalKeyExpr',
    'FRACTAL_DEFAULT_NAMESPACE', 'FRACTAL_SESSION_SCOPE',
]

FRACTAL_SESSION_SCOPE = "fractal_session"
FRACTAL_DEFAULT_NAMESPACE = "MOSS/fractal"


class FractalKeyExpr:
    """
    约定一个声明的标准实现.
    主要是为 zenoh 体系服务, 但实际上也适合 redis 或其它广播协议.
    :param hub_name: 提供服务的 Fractal Hub 的名称. 
    :param namespace: 命名空间.  
    """

    def __init__(
            self,
            hub_name: str,
            *,
            namespace: str | None = None,
    ):
        self.hub_name = hub_name.strip('/')
        self._namespace = namespace.strip('/') if namespace else FRACTAL_DEFAULT_NAMESPACE

    def liveness_key(self, node_name: str) -> str:
        prefix = self.liveness_namespace()
        return f"{prefix}/{node_name}"

    def liveness_namespace(self) -> str:
        return f"{self._namespace}/{self.hub_name}/liveness"

    def liveness_wildcard(self) -> str:
        return f"{self.liveness_namespace()}/**"

    def provider_namespace(self):
        # 监听不同的 Fractal cell 的提供者.
        return f"{self._namespace}/{self.hub_name}/providers"

    def provider_wildcard(self) -> str:
        return f"{self.provider_namespace()}/**"

    def provider_key(self, cell_name: str) -> str:
        cell_name = cell_name.strip('/')
        prefix = self.provider_namespace()
        return "/".join([prefix, cell_name])
