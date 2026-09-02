"""Dolores 的强类型 tool 调用模型 — 对齐 dsh_plugin/moss-dolores-ghost-plugin.ts 的 defineTool.

镜像 deepseek_harness/types/session_events.py 的 SessionEventModel 判别范式, 但**无泛型基座**:
每个 tool 类自带 ``from_tool_call(event)`` — 先按函数名判别 (与 plugin defineTool 的
name 对齐), 名字不符返回 None 不碰 arguments; 匹配才 ``json.loads(arguments)`` 解析成
强类型字段. 分派处用字面量 ``if x := Tool.from_tool_call(event):`` 链, 不再引入注册表/抽象基座.

字段命名: ``from_tool_call`` 把 ToolCallEvent 的 ``callId`` (dsh camelCase) 搬运到
本类 ``call_id`` (snake_case), 与 project 常量风格一致.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import Self

from ghoshell_moss.deepseek_harness.types.session_events import ToolCallEvent
from ghoshell_moss.core.blueprint.moment import Moment

__all__ = ["FetchNextMomentToolCall", "WaitNextMomentToolCall", "AppendCtmlToolCall"]

_ResultType = dict | list | str | None


class ToolCallResult(BaseModel):
    """
    tool 通过 rpc 回调 plugin 的原始数据结构.
    需要将当前数据结构映射成 plugin 对应接口的入参.
    plugin 侧的 yield tool ( dsh 侧让出会话等待 moss 下一帧 moment) 不通过 tool call result 协议返回.
    """
    call: ToolCallEvent
    result: _ResultType = Field(
        description="tool call 给模型的返回值. "
    )
    error: str | None = Field(
        default=None,
        description="参数请求失败. ",
    )
    moment: Optional[Moment] = Field(
        default=None,
        description="携带的 moment, 在 plugin 的 rpc 接口返回 call id 对应的 result 后解锁 tool 返回, moment 应该注入."
                    "moment 注入只走 <moment> (context) 槽位; inputs 只在 thinking/enter 走 steer."
    )


class ToolCallParameter(BaseModel, ABC):
    tool_call_event: ToolCallEvent | None = Field(
        default=None,
        description="原始的 tool call, 实例化后必不为空"
    )

    @classmethod
    @abstractmethod
    def tool_name(cls) -> str:
        ...

    @classmethod
    def from_tool_call(cls, event: ToolCallEvent) -> Self | None:
        """
        从 tool call event 中构建.
        :raise ValidationError: 入参构建失败.
        """
        if event.name != cls.tool_name():
            return None
        parameter = cls.model_validate_json(event.arguments or "{}")
        parameter.tool_call_event = event
        return parameter

    @classmethod
    async def run_tool(
            cls,
            event: ToolCallEvent,
            handler: Callable[[Self], Awaitable[_ResultType | ToolCallResult]],
    ) -> ToolCallResult | None:
        try:
            call = cls.from_tool_call(event)
            if call is None:
                return None
        except ValidationError:
            return ToolCallResult(
                call=event,
                result=None,
                error="invalid tool parameter",
            )

        try:
            result = await handler(call)
            if isinstance(result, ToolCallResult):
                return result
            return call.new_tool_call_result(result)

        except Exception as e:
            return ToolCallResult(
                call=event,
                result=None,
                error=str(e),
            )

    def new_tool_call_result(self, result: _ResultType) -> ToolCallResult:
        return ToolCallResult(
            call=self.tool_call_event,
            result=result,
            # moment 在外部决定是否拼装.
            moment=None,
        )


class FetchNextMomentToolCall(ToolCallParameter):
    """moss_fetch_next_moment tool — 主动 fetch 下一帧: 产 moment, 返回 {moment_ref} 并注入 context.

    对齐 plugin.ts ``defineTool({ name: 'moss_fetch_next_moment', parameters: {} })``.
    """

    @classmethod
    def tool_name(cls) -> str:
        return "moss_fetch_next_moment"


class WaitNextMomentToolCall(ToolCallParameter):
    """moss_wait_next_moment (yield) tool — 被动让出, 阻塞等下一帧 moment.

    对齐 plugin.ts ``defineTool({ name: 'moss_wait_next_moment', parameters: {} })``.
    控制信号, 不产 ToolCallResult (不走 tool-result RPC) — 见 ToolCallResult docstring.
    """

    @classmethod
    def tool_name(cls) -> str:
        return "moss_wait_next_moment"


class AppendCtmlToolCall(ToolCallParameter):
    """moss_append_ctml tool — 追加一段 ctml 到执行, 思维超前于行为 (interleaved).

    对齐 plugin.ts ``defineTool({ name: 'moss_append_ctml', parameters: { ctml, refresh_meta, wait_done } })``.
    不产 moment — 产 moment 是 fetch_next_moment 的职责, 两个函数语义不混.
    """

    ctml: str = Field(default="", description="要执行的 ctml 命令.")
    refresh_meta: bool = Field(default=False, description="执行前刷新 shell meta (deferred — 当前 no-op).")
    wait_done: bool = Field(default=False, description="true 等 action 执行完 (wait_action_done), false 只等编译 (wait_compiled).")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_append_ctml"
