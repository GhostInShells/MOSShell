"""Listener host core — 火山 ASR 实现 + audio capture 实现.

子模块直接 import (``host.listener.volcengine_asr`` / ``host.listener.capture``),
本包 ``__init__`` 不再聚合导出 (旧 VoiceController/VoiceStateMachine 状态机已删除).
"""
