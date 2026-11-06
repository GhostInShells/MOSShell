"""
apiproxy 接口面 + SDK 协议的强类型数据模型.

数据面 (请求载荷 / 响应值 / 错误码枚举 / 流帧), 不含 transport/carrier 机制
(RpcId minting / AbortSignal / 四象限 wire-form). 品牌类型为 str, 判别联合用
str | Literal 支持扩展.

模块按依赖序拆分, 避免循环 import:
  rpc (信封+错误) → nouns (WorkspaceView/JobView) → events (帧) → sessions → sdk.
"""

from . import attachments, events, nouns, refs, rpc, sdk, session_state, sessions
from .attachments import *
from .events import *
from .nouns import *
from .refs import *
from .rpc import *
from .sdk import *
from .session_state import *
from .sessions import *

__all__ = (
    attachments.__all__
    + session_state.__all__
    + rpc.__all__
    + nouns.__all__
    + events.__all__
    + sessions.__all__
    + sdk.__all__
    + refs.__all__
)
