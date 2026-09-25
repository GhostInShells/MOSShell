"""
apiproxy RPC 信封与错误码.

镜像 dsh `packages/host/apiproxy/src/api/rpc.ts` 的抽象契约. 四象限消息
(ClientRequest/ServerResponse/ServerRequest/ClientResponse) 是物理 carrier 形状,
不属于本数据面; 这里只建模共享的结果包装 (RpcResult) 与错误码枚举 (RpcErrorCode).

信封是热路径薄载体 (纯透传 rpcId/result, 无校验项), 用 TypedDict (零运行时成本,
忠实于 TS 的 compile-time interface), 不做 pydantic 校验; 载荷在消费方按需用具体
pydantic 模型 validate. 品牌类型为 str, 判别联合用 str | Literal 支持扩展.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

__all__ = [
    "RpcErrorCode",
    "RpcError",
    "RpcResult",
]


# 错误码闭枚举 (源: RpcErrorCode = keyof RpcErrorDetailsMap, "Closed error-code union").
RpcErrorCode = Literal[
    "bad-request",
    "cancelled",
    "session-not-found",
    "model-unavailable",
    "session-conflict",
    "invalid-time-zone",
    "workspace-attach-failed",
    "workspace-not-found",
    "workspace-invalid-path",
    "workspace-name-conflict",
    "workspace-move-invalid",
    "directory-unreadable",
    "directory-exists",
    "directory-create-failed",
    "directory-picker-unavailable",
    "agent-preset-read-only",
    "agent-preset-locked",
    "agent-preset-conflict",
    "agent-preset-not-found",
    "agent-preset-invalid",
    "agent-busy",
    "attachment-error",
    "queue-item-not-found",
    "steer-unavailable",
    "command-error",
    "unknown-command",
    "settings-rejected",
    "settings-not-exposed",
    "settings-conflict",
    "credential-rejected",
    "model-discovery-failed",
    "title-invalid",
    "fork-unavailable",
    "subagent-parent-unavailable",
    "subagent-not-found",
    "subagent-catalog-diagnostic",
    "subagent-not-resumable",
    "subagent-unauthorized",
    "subagent-delivery-unavailable",
    "internal",
]


class RpcError(TypedDict):
    """业务失败分支. `details` 是 code 对应的类型化诊断载荷 (40 种 shape), 按不透明 dict 保留."""

    code: RpcErrorCode
    message: str
    details: dict[str, Any]


class RpcResult(TypedDict):
    """业务成功/失败结果槽 (TS 是判别联合 {ok:true,value} | {ok:false,error}).

    `value` 是成功分支载荷, 每动词的具体类型在各自域模块建模; 消费方按动词用
    具体 pydantic 模型 validate `value`. `value`/`error` 二选一.
    """

    ok: bool
    value: NotRequired[Any]
    error: NotRequired[RpcError]
