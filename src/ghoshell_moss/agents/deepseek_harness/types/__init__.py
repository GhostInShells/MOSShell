"""
apiproxy 接口面 + SDK 协议的强类型数据模型.

数据面 (请求载荷 / 响应值 / 错误码枚举 / 流帧), 不含 transport/carrier 机制
(RpcId minting / AbortSignal / 四象限 wire-form). 品牌类型为 str, 判别联合用
str | Literal 支持扩展.

模块按依赖序拆分, 避免循环 import:
  rpc (信封+错误) → nouns (WorkspaceView/JobView) → events (帧) → sessions → domains → sdk.
"""

from . import domains, events, nouns, rpc, sdk, sessions
from .rpc import *
from .nouns import *
from .events import *
from .sessions import *
from .domains import *
from .sdk import *

__all__ = (
    rpc.__all__
    + nouns.__all__
    + events.__all__
    + sessions.__all__
    + domains.__all__
    + sdk.__all__
)
