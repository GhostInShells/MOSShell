"""
DshClient — apiproxy 全局「管理面」动词的强类型 facade.

职责单一: 把 apiproxy 的全局管理面动词, 以强类型方法暴露成 facade.
每个动词一个命名方法, 方法名与动词一一绑定 (session_list → session.list),
method 字符串只出现一次, 写死在方法体内; 不搞 METHOD 常量表, 不搞 RPC 泛型自描述.

依赖最小: 只组合 base_url + logger + timeout, 自建 http client. 不持有 DshLauncher,
不感知进程生命周期 / ws / subprocess.

只挂管理面动词 (session/workspace/host/agent-preset/settings/credentials/llm/skill/goal
的只读与 CRUD); 不挂驱动 session 推理循环的动词 (session.prompt/cancel/update-queue/
select-model/attachment/...), 那些留给未来的 session 层.

数据面: 请求载荷与响应值沿用 types/ 里已有的 pydantic 模型 (纯数据, 不背协议名).
错误分支统一为 DshRpcException (携带 RpcError 的 code/message/details).

类型未建全、暂不挂的动词 (待补 params/value 后再装):
  agent-preset.remove             — 缺 value (仅 AgentPresetRemoveParams)
  settings.update/replace/mutate  — 缺 value (仅 params)
  credential.set/unset            — 缺 value (仅 params)
  goal.edit                       — 缺 value (仅 GoalEditParams)
  goal.pause/resume/complete      — 缺 params (仅共享 GoalRefValue)
  goal.clear                      — 缺 params (仅 GoalClearValue)
"""

from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel

from ghoshell_moss.deepseek_harness.types import domains, sessions
from ghoshell_moss.deepseek_harness.types.rpc import RpcError
from ghoshell_moss.contracts.logger import LoggerItf

__all__ = [
    "DshClient",
    "DshRpcException",
]

_ValueT = TypeVar("_ValueT", bound=BaseModel)


class DshRpcException(Exception):
    """dsh rpc 业务失败分支: 携带 RpcError(code/message/details)."""

    def __init__(self, method: str, error: RpcError) -> None:
        self.method = method
        self.error = error
        super().__init__(f"dsh rpc {method} failed [{error['code']}]: {error['message']}")


class DshClient:
    """强类型全局管理面 facade: 每个方法 = 一个 apiproxy 动词, 不背业务逻辑."""

    def __init__(self, base_url: str, logger: LoggerItf, *, timeout: float = 10.0) -> None:
        self._base_url = base_url
        self._logger = logger
        self._http_client = httpx.AsyncClient(timeout=timeout)
        self._rpc_counter = 0

    async def call(self, method: str, params: BaseModel | None, value_cls: type[_ValueT]) -> _ValueT:
        """引擎: params 序列化 → 信封 POST → ok 则 value_cls.model_validate, 否则 raise."""
        payload = params.model_dump(exclude_none=True, by_alias=True) if params is not None else {}
        self._rpc_counter += 1
        envelope = {
            "type": "client-request",
            "rpcId": f"rpc-{self._rpc_counter}",
            "method": method,
            "payload": payload,
        }
        self._logger.debug("dsh rpc %s (rpcId=%s)", method, envelope["rpcId"])
        resp = await self._http_client.post(
            f"{self._base_url}/api/{method}",
            json=envelope,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        if not result["ok"]:
            error = result.get("error")
            if error is None:
                error = {"code": "internal", "message": "no error", "details": {}}
            raise DshRpcException(method, error)
        return value_cls.model_validate(result.get("value"))

    async def plugin_call(self, path: str, payload: dict | None = None) -> dict:
        """plugin webServer 面 (非 apiproxy): 裸 POST `{base}{path}`, 返回响应 JSON dict.

        与 call() 分工: call() 走 apiproxy 的 client-request 信封 (POST /api/{method});
        plugin 注册的 HTTP 路由走这里 — raw JSON body, 无信封, 响应为 plugin 自定义 JSON
        (非 result 包络). 与 launcher.call 同语义.
        """
        resp = await self._http_client.post(f"{self._base_url}{path}", json=payload or {})
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._http_client.aclose()

    # ---- session (管理面) ---- #

    async def session_list(self, params: sessions.SessionListParams) -> sessions.SessionListValue:
        return await self.call("session.list", params, sessions.SessionListValue)

    async def session_search(self, params: sessions.SessionSearchParams) -> sessions.SessionSearchValue:
        return await self.call("session.search", params, sessions.SessionSearchValue)

    async def session_create(self, params: sessions.SessionCreateParams) -> sessions.SessionCreateValue:
        return await self.call("session.create", params, sessions.SessionCreateValue)

    async def session_history(self, params: sessions.SessionHistoryParams) -> sessions.SessionHistoryValue:
        return await self.call("session.history", params, sessions.SessionHistoryValue)

    async def session_rename(self, params: sessions.SessionRenameParams) -> sessions.SessionRenameValue:
        return await self.call("session.rename", params, sessions.SessionRenameValue)

    async def session_fork(self, params: sessions.SessionForkParams) -> sessions.SessionForkValue:
        return await self.call("session.fork", params, sessions.SessionForkValue)

    # ---- workspace ---- #

    async def workspace_list(self) -> domains.WorkspaceListValue:
        return await self.call("workspace.list", None, domains.WorkspaceListValue)

    async def workspace_create(self, params: domains.WorkspaceCreateParams) -> domains.WorkspaceCreateValue:
        return await self.call("workspace.create", params, domains.WorkspaceCreateValue)

    async def workspace_rename(self, params: domains.WorkspaceRenameParams) -> domains.WorkspaceRenameValue:
        return await self.call("workspace.rename", params, domains.WorkspaceRenameValue)

    async def workspace_delete(self, params: domains.WorkspaceDeleteParams) -> domains.WorkspaceDeleteValue:
        return await self.call("workspace.delete", params, domains.WorkspaceDeleteValue)

    async def workspace_insert_before(self, params: domains.WorkspaceInsertBeforeParams) -> domains.WorkspaceInsertBeforeValue:
        return await self.call("workspace.insert-before", params, domains.WorkspaceInsertBeforeValue)

    async def workspace_insert_session_before(
            self, params: domains.WorkspaceInsertSessionBeforeParams,
    ) -> domains.WorkspaceInsertSessionBeforeValue:
        return await self.call("workspace.insert-session-before", params, domains.WorkspaceInsertSessionBeforeValue)

    async def workspace_archive_session(
            self, params: domains.WorkspaceArchiveSessionParams,
    ) -> domains.WorkspaceArchiveSessionValue:
        return await self.call("workspace.archive-session", params, domains.WorkspaceArchiveSessionValue)

    # ---- host ---- #

    async def host_describe(self) -> domains.HostDescribeValue:
        return await self.call("host.describe", None, domains.HostDescribeValue)

    async def host_pick_directory(self) -> domains.HostPickDirectoryValue:
        return await self.call("host.pick-directory", None, domains.HostPickDirectoryValue)

    async def host_list_directory(self, params: domains.HostListDirectoryParams) -> domains.DirectoryListing:
        return await self.call("host.list-directory", params, domains.DirectoryListing)

    async def host_create_directory(self, params: domains.HostCreateDirectoryParams) -> domains.HostCreateDirectoryValue:
        return await self.call("host.create-directory", params, domains.HostCreateDirectoryValue)

    async def host_open_path(self, params: domains.HostOpenPathParams) -> domains.HostOpenPathValue:
        return await self.call("host.open-path", params, domains.HostOpenPathValue)

    # ---- agent-preset ---- #

    async def agent_preset_list(self) -> domains.AgentPresetListValue:
        return await self.call("agent-preset.list", None, domains.AgentPresetListValue)

    async def agent_preset_select(self, params: domains.AgentPresetSelectParams) -> domains.AgentPresetSelectValue:
        return await self.call("agent-preset.select", params, domains.AgentPresetSelectValue)

    async def agent_preset_read(self, params: domains.AgentPresetReadParams) -> domains.AgentPresetReadValue:
        return await self.call("agent-preset.read", params, domains.AgentPresetReadValue)

    async def agent_preset_copy(self, params: domains.AgentPresetCopyParams) -> domains.AgentPresetCopyValue:
        return await self.call("agent-preset.copy", params, domains.AgentPresetCopyValue)

    async def agent_preset_open_document(
            self, params: domains.AgentPresetOpenDocumentParams,
    ) -> domains.AgentPresetOpenDocumentValue:
        return await self.call("agent-preset.open-document", params, domains.AgentPresetOpenDocumentValue)

    # ---- settings ---- #

    async def settings_describe(self) -> domains.SettingsDescribeValue:
        return await self.call("settings.describe", None, domains.SettingsDescribeValue)

    async def settings_open_document(self) -> domains.SettingsOpenDocumentValue:
        return await self.call("settings.open-document", None, domains.SettingsOpenDocumentValue)

    # ---- credentials ---- #

    async def credential_describe(self, params: domains.CredentialDescribeParams) -> domains.CredentialDescribeValue:
        return await self.call("credential.describe", params, domains.CredentialDescribeValue)

    # ---- llm ---- #

    async def llm_providers(self) -> domains.LlmProvidersValue:
        return await self.call("llm.providers", None, domains.LlmProvidersValue)

    async def llm_models(self) -> domains.LlmModelsValue:
        return await self.call("llm.models", None, domains.LlmModelsValue)

    async def llm_discover_models(self, params: domains.LlmDiscoverModelsParams) -> domains.LlmDiscoverModelsValue:
        return await self.call("llm.discover-models", params, domains.LlmDiscoverModelsValue)

    # ---- skill ---- #

    async def skill_list(self, params: domains.SkillListParams) -> domains.SkillListValue:
        return await self.call("skill.list", params, domains.SkillListValue)

    # ---- goal ---- #

    async def goal_create(self, params: domains.GoalCreateParams) -> domains.GoalCreateValue:
        return await self.call("goal.create", params, domains.GoalCreateValue)
