from ghoshell_moss.core.blueprint.mindflow import Impulse, Reaction, Attention
from ghoshell_moss.core.mindflow.base_mindflow import AbsMindflow
from ghoshell_moss.core.mindflow.priority_attention import PriorityProtectionAttention
from ghoshell_moss.core.mindflow.input_signal_nucleus import InputSignalNucleus
from ghoshell_moss.contracts import LoggerItf

__all__ = ['PriorityMindflow', 'new_default_mindflow']


class PriorityMindflow(AbsMindflow):
    """
    Mindflow with priority-based arbitration (PriorityProtectionAttention).
    """

    def __init__(
            self,
            *,
            protection_seconds: float = 2.5,
            logger: LoggerItf | None = None, **kwargs):
        super().__init__(logger=logger, **kwargs)
        self._protection_seconds = protection_seconds

    def _build_attention(self, impulse: Impulse, inherit_outcome: Reaction) -> Attention:
        return PriorityProtectionAttention(
            impulse=impulse,
            previous=inherit_outcome,
            logger=self._logger,
            protection_seconds=self._protection_seconds,
        )


def new_default_mindflow(
        *,
        protection_seconds: float = 2.5,
        input_buffer_size: int = 20,
        input_suppress_seconds: float = 0.5,
        logger: LoggerItf | None = None,
) -> PriorityMindflow:
    """
    开箱即用的 Mindflow 工厂.

    预配置:
      - InputSignalNucleus — IM 红点式 input signal 聚合
      - PriorityProtectionAttention — 纯优先级 + 固定保护期仲裁

    最小用法:

        mindflow = new_default_mindflow()
        async with mindflow:
            mindflow.add_signal(Signal.new("input", ...))
            async for attention in mindflow.loop():
                ...

    InputSignalNucleus 不绑定任何 Mindflow 实现 — 工厂函数显式注册,
    你可以随意替换自己的 nucleus 或 attention.
    """
    mindflow = PriorityMindflow(
        protection_seconds=protection_seconds,
        logger=logger,
    )
    # 输入信号聚合
    mindflow.with_nucleus(InputSignalNucleus(
        buffer_size=input_buffer_size,
        suppress_seconds=input_suppress_seconds,
        logger=logger,
    ))
    return mindflow
