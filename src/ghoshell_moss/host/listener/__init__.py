"""Listener host core — 火山 ASR 实现 + listener 治理.

子模块直接 import (``host.listener.volcengine_sauc``), 本包 ``__init__`` 不再聚合
导出 (旧 VoiceController/VoiceStateMachine 状态机已删除). audio capture 实现已迁至
``host.audios``.
"""
