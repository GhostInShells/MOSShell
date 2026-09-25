"""InterleavedVoice — Voice contract 的默认实现.

从 moss_runtime 迁出的装线:
- 说侧 clause → ClauseTopic 桥 (广播 ghost 自己的话) + feed_ghost_clause 联动.
- 说侧 player observe → AudioSampleTopic 桥.
- 听侧 ListenerController 组装 + 生命周期 enter (controller 内部已含 listener-side
  clause 桥, 见 controller.with_topic_service).

设计要点:
- 构造在 matrix bootstrap 之后触发 (IoC factory 从 container 拿 Matrix), 立即完成
  controller 组装. speech()/listener_channel() 此刻即可读, 供 runtime 在 shell
  __aenter__ 之前装线.
- run(speech, listen) 无副作用, 返回 VoiceLifecycle. Lifecycle 的 __aenter__ 才装
  说侧桥 (若 speech=True) + controller enter (若 listen=True).
- Lifecycle 在 shell __aenter__ **之前** 进入 (voice 前起, speech 后起): 桥的回调
  是空转注册 (speech 未起, 没有 clause 产出), shell.__aenter__ 启动 speech 后回调
  才被触发. 反过来会漏最早的 clause.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Optional

import janus
import numpy as np
from typing_extensions import Self

from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.listener import ASRListener
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.speech import (
    PlaybackSample,
    Speech,
    SpeechClause,
    TTSSpeech,
)
from ghoshell_moss.contracts.voice import Voice, VoiceLifecycle
from ghoshell_moss.contracts.audio import (
    AUDIO_SAMPLE_INTERVAL,
    LatestAudioWindow,
    compute_spectrum,
)
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.host.listener.controller import (
    ListenerController,
    build_stop_caller_factory,
)
from ghoshell_moss.types.topics import AudioSampleTopic, ClauseTopic

__all__ = ["InterleavedVoice", "InterleavedVoiceLifecycle"]


class InterleavedVoice(Voice):
    """交错语音总装: 说 + 听 + 联动. IoC provider factory 构造."""

    def __init__(
            self,
            *,
            speech: Optional[Speech],
            listener: Optional[ASRListener],
            matrix: Matrix,
            config_store: Optional[ConfigStore] = None,
            logger: Optional[LoggerItf] = None,
    ) -> None:
        self._speech = speech
        self._matrix = matrix
        self._logger = logger or logging.getLogger("moss")
        self._controller: Optional[ListenerController] = None
        if listener is not None:
            self._controller = ListenerController(
                listener=listener,
                asr=listener.asr(),
                logger=self._logger,
                signal_broadcast=matrix.session.add_signal,
                stop_caller_factory=build_stop_caller_factory(matrix.container),
                cell_name=matrix.this.name,
                config_store=config_store,
                topic_service=matrix.session.topics,
            )
            # 单例注册: TUI voice state / 其它消费面从 container 拿同一个 controller.
            # 保留旧兼容 — 迁移期若有消费方直接 container.get(ListenerController).
            matrix.container.set(ListenerController, self._controller)

    def speech(self) -> Optional[Speech]:
        return self._speech

    def listener_channel(self) -> Optional[Channel]:
        return self._controller.as_channel() if self._controller else None

    def run(self, *, speech: bool, listen: bool) -> VoiceLifecycle:
        return InterleavedVoiceLifecycle(
            speech=self._speech if speech else None,
            controller=self._controller if listen else None,
            matrix=self._matrix,
            logger=self._logger,
        )


class InterleavedVoiceLifecycle(VoiceLifecycle):
    """Voice 的运行时生命周期 — matrix.add_lifecycle_object 托管.

    __aenter__ 装线顺序 (LIFO 退出):
    1. 说侧 clause topic 桥 (若 speech is TTSSpeech): 注册 on_clause 回调 + drain task.
    2. 说侧 audio sample topic 桥: 注册 player.observe + emit task.
    3. 听侧 controller enter (listener + asr + controller 内部 clause 桥).

    speech 未 running 时注册的回调空转 —— shell.__aenter__ 启动 speech 后开始触发.
    """

    def __init__(
            self,
            *,
            speech: Optional[Speech],
            controller: Optional[ListenerController],
            matrix: Matrix,
            logger: LoggerItf,
    ) -> None:
        self._speech = speech
        self._controller = controller
        self._matrix = matrix
        self._logger = logger
        self._exit_stack = contextlib.AsyncExitStack()
        self._entered = False

    async def __aenter__(self) -> Self:
        if self._entered:
            raise RuntimeError("VoiceLifecycle 已 enter, 不复用")
        self._entered = True
        await self._exit_stack.__aenter__()
        # 说侧桥: 只在 speech 为 TTSSpeech 时装 (NullSpeech / MockSpeech 无 clause/sample).
        if isinstance(self._speech, TTSSpeech):
            await self._exit_stack.enter_async_context(self._clause_topic_bridge(self._speech))
            await self._exit_stack.enter_async_context(self._audio_sample_topic_bridge(self._speech))
        # 听侧: controller enter (内部启动 listener + clause topic 桥) + 启动默认礼仪.
        # "启动即听": 默认礼仪常驻, 不停在 stop 等模型/人类手动 activate (e3041d2c 语义).
        if self._controller is not None:
            await self._exit_stack.enter_async_context(self._controller)
            self._controller.start_default_etiquette()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    def pause(self, toggle: bool = True) -> None:
        if self._controller is not None:
            self._controller.pause(toggle)

    @contextlib.asynccontextmanager
    async def _clause_topic_bridge(self, speech: TTSSpeech):
        """说侧 clause → ClauseTopic(role=ghost) + feed_ghost_clause 联动.

        on_clause 在 audio worker 线程触发, 走 janus sync_q 线程安全入队, drain task
        在事件循环出队 pub. 顺路把文本喂进听侧 ASR corpus 的 lines 投影
        (controller.feed_ghost_clause). 无 controller 时跳过喂料.
        """
        env = self._matrix.env
        speaker_id = env.project_id
        speaker_name = env.ghost_name
        publisher = self._matrix.session.topics.model_publisher(
            creator=f"ghost/{speaker_name}",
            model=ClauseTopic,
        )
        queue: janus.Queue = janus.Queue()

        def _on_clause(clause: SpeechClause) -> None:
            queue.sync_q.put_nowait(ClauseTopic(
                text=clause.text,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                role='ghost',
            ))

        controller = self._controller

        async def _drain() -> None:
            while True:
                topic = await queue.async_q.get()
                publisher.pub(topic)
                if controller is not None:
                    controller.feed_ghost_clause(topic.text)

        await publisher.__aenter__()
        disposer = speech.on_clause(_on_clause)
        drain_task = asyncio.create_task(_drain())
        try:
            yield
        finally:
            disposer()
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
            await publisher.__aexit__(None, None, None)

    @contextlib.asynccontextmanager
    async def _audio_sample_topic_bridge(self, speech: TTSSpeech):
        """说侧 player observe → AudioSampleTopic(role=ghost).

        与 clause 桥对称, 但 latest-value-wins (无队列, 用 LatestAudioWindow).
        player.observe 在 audio worker 线程回调, 周期 task 在事件循环取走算频谱发布.
        """
        speaker_name = self._matrix.env.ghost_name
        player = speech.player()
        sample_rate = player.sample_rate
        publisher = self._matrix.session.topics.model_publisher(
            creator=f"ghost/{speaker_name}",
            model=AudioSampleTopic,
        )
        window = LatestAudioWindow()

        def _on_sample(sample: PlaybackSample) -> None:
            if not sample.pcm:
                return
            window.append(np.frombuffer(sample.pcm, dtype=np.int16))

        async def _emit() -> None:
            while True:
                await asyncio.sleep(AUDIO_SAMPLE_INTERVAL)
                pcm = window.take()
                if pcm is None:
                    continue
                spectrum = compute_spectrum(pcm)
                publisher.pub(AudioSampleTopic(
                    role="ghost",
                    sample_rate=sample_rate,
                    duration=len(pcm) / sample_rate if sample_rate else 0.0,
                    rms_db=spectrum.rms_db,
                    peak=spectrum.peak,
                    spectrum_bins=spectrum.spectrum_bins,
                    n_spectrum_bins=len(spectrum.spectrum_bins),
                    waveform=spectrum.waveform,
                    n_waveform=len(spectrum.waveform),
                ))

        await publisher.__aenter__()
        disposer = player.observe(_on_sample)
        emit_task = asyncio.create_task(_emit())
        try:
            yield
        finally:
            disposer()
            emit_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await emit_task
            await publisher.__aexit__(None, None, None)
