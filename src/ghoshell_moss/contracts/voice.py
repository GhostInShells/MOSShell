"""Voice contract — 语音总装表面 (听说交错的可插拔总装点).

Voice 收口了以下装线逻辑, 它们在 stage 1 散落在 ``host/moss_runtime.py``:
- 说侧 clause → ClauseTopic 桥 (广播 ghost 自己的话)
- 说侧 player observe → AudioSampleTopic 桥 (广播频谱)
- 听侧 ListenerController 组装 + 生命周期 enter
- 听说联动: ``feed_ghost_clause`` (ghost clause → ASR corpus tail)
- 人类锁级联: ``pause`` → listener controller

runtime 通过三条接口消费:
- ``speech()``: shell 装 speech 的引用来源 (可 None).
- ``listener_channel()``: shell.main_channel 挂载听侧命令 channel (可 None).
- ``run(speech, listen)``: 声明本次启用哪一侧, 返回 ``VoiceLifecycle`` 交给
  ``matrix.add_lifecycle_object`` 托管. **无副作用** — 装线全部在 Lifecycle
  的 ``__aenter__``, ``run`` 只做参数绑定.

生命周期契约: Voice lifecycle **在 shell __aenter__ 之前** enter (voice 前起),
shell 起来后 speech 才真正开始产 clause — 桥的回调注册是空转的、正确对齐.

IoC 里存在 → 有语音能力; 缺席 → runtime 拿 None, 无语音 (shell.set_speech(None),
无 listener channel 挂载, 无 lifecycle 托管).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from typing_extensions import Self

from ghoshell_moss.contracts.speech import Speech
from ghoshell_moss.core.blueprint.matrix import MatrixLifecycleObject
from ghoshell_moss.core.concepts.channel import Channel

__all__ = ["Voice", "VoiceLifecycle"]


class VoiceLifecycle(MatrixLifecycleObject):
    """Voice 的运行时生命周期表面 — matrix 托管.

    额外持有 ``pause`` 表面, runtime.pause 级联到此 (再级联到 listener controller).
    """

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...

    @abstractmethod
    def pause(self, toggle: bool = True) -> None:
        """人类锁级联 — pause(True) 抑制听侧; pause(False) 释放."""
        ...


class Voice(ABC):
    """交错语音总装表面 — IoC 里注册即启用, 缺席即无语音.

    实现方 (``host/voice/interleaved.py``) 在 IoC factory 里组装:
    从 container 拿 Speech / ASRListener / Matrix, 构造 controller (需要
    ``matrix.session.topics`` / ``matrix.session.add_signal`` / ``matrix.this.name``,
    故 factory 触发时刻必须在 matrix bootstrap 之后 — runtime 在 __aenter__
    的 bootstrap 步之后再 ``container.get(Voice)``).

    ``speech()`` / ``listener_channel()`` 在构造完即可读, 供 shell 装线在
    lifecycle enter 之前完成 (shell.set_speech / main_channel.import_channels).
    """

    @abstractmethod
    def speech(self) -> Optional[Speech]:
        """当前持有的 Speech 实例, runtime 装到 shell. None → shell 不挂 say/mute."""
        ...

    @abstractmethod
    def listener_channel(self) -> Optional[Channel]:
        """听侧命令 channel (activate/stop/get_transcript/...), runtime 挂到
        shell.main_channel. None → 无 listener, 不挂 channel.
        """
        ...

    @abstractmethod
    def run(self, *, speech: bool, listen: bool) -> VoiceLifecycle:
        """声明本次启用哪一侧, 返回 lifecycle.

        - speech=True, listen=True → 完整交错 (说侧桥 + 听侧 controller + 联动).
        - speech=True, listen=False → 说侧独立 (clause 桥 / audio sample 桥装线,
          无 controller, 无 feed_ghost_clause).
        - speech=False, listen=True → 只听 (无说侧桥, 无 feed_ghost_clause).
        - speech=False, listen=False → 空 lifecycle (无副作用, 但仍可 enter/exit).

        实现方防御: 已 run 过再调应抛错 — lifecycle 不复用.
        """
        ...
