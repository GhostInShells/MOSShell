"""
Mindflow 调度中枢实现.

蓝图:  ghoshell_moss.core.blueprint.mindflow

模块索引:

  base_attention       — AbsAttention (抽象生命周期) + BaseAttention (强度衰减仲裁)
  base_mindflow        — AbsMindflow (抽象调度) + BaseMindflow (强度衰减实现)
  buffer_nucleus       — BufferNucleus, 极简信号闸门 (Gemini 3 原版)
  input_signal_nucleus — InputSignalNucleus, IM 红点式信号聚合
  priority_attention   — PriorityProtectionAttention, 纯优先级+固定保护期仲裁
  priority_mindflow    — PriorityMindflow + new_default_mindflow() 工厂函数
"""

from ghoshell_moss.core.mindflow.priority_mindflow import new_default_mindflow, PriorityMindflow

__all__ = ['new_default_mindflow', 'PriorityMindflow']
