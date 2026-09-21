"""Listener Controller — 判停逻辑装线 + listener signal 生产边界 + 运行时自解释.

判停逻辑 (聆听礼仪) 通过 ListenerState.on_event_creating 挂载到识别层 (inline await),
判停时调 state.commit(). commit 机制 (发负序号切段) 已在 recognition 层, 这里只决定
何时调用.

信号发射是独立于判停的第二职责: 本层是 RecognitionEvent (text axis) → listener signal
(first/clause/tail) 的生产边界. 构造时注入 ``signal_broadcast`` (signal sink), 存在时
注册一条 listener 级 on_recognition_result 观察者, 机械地把每个识别事件翻译成 listener
signal 并广播.

第三职责是运行时自解释: 当前礼仪 = ``_active_etiquette`` 单一真值, 提供合成快照
(``snapshot()``) 与随身 channel (``as_channel()``), 让模型能判断"耳朵开没开、什么礼仪".
"""
import asyncio
import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import janus
import numpy as np
from typing_extensions import Self
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.asr import ASR, RecognitionEvent, RecognitionPhase, RecognitionSegment
from ghoshell_moss.contracts.audio import (
    AUDIO_SAMPLE_INTERVAL,
    AudioChunk,
    LatestAudioWindow,
    compute_spectrum,
)
from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.llms import MossLLMCaller
from ghoshell_moss.contracts.listener import ListenLifecycle, Listener, ListenerState
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.core.blueprint.mindflow import ChallengeMode, Priority, Signal
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import Publisher, TopicService
from ghoshell_moss.core.mindflow.listener_nucleus import new_listener_signal
from ghoshell_moss.host.listener.etiquette import (
    DeliverSpec,
    EtiquetteConfig,
    EtiquetteSpec,
    OnsetSpec,
    always as ALWAYS,
    once as ONCE,
    scored as SCORED,
)
from ghoshell_moss.host.listener.segment_buffer import SegmentBuffer
from ghoshell_moss.host.listener.stop_judge import StopJudge, StopScoreObservation
from ghoshell_moss.types.topics import AudioSampleTopic, ClauseTopic

__all__ = [
    "ListenerController",
    "ListenerSnapshot",
    "StopDetectorFactory",
]


#: 出口位点的装配面: 吃 (礼仪, commit 开关), 返回该 session 的判停单元.
#: 默认实现由 ``ListenerController`` 按 ``StopSpec`` + 注入的 caller 组装;
#: 注入自定义 factory 即可整段替换 (container 在 runtime get 一次后塞进来).
StopDetectorFactory = Callable[[EtiquetteSpec, Callable[[], None]], StopJudge]


@dataclass
class ListenerSnapshot:
    """合成快照 — 当前激活礼仪 + listening + ASR 当前参数值.

    ``etiquette`` 是当前激活礼仪的 name (无则 "off"); 与 ``_active_etiquette`` 同源,
    不新增第二真值. 温数据, 进 notice, 变了才重发. 参数 schema 是冷数据 (进 instruction),
    不在此.
    """

    etiquette: str
    listening: bool
    asr_params: dict

    def render_notice(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def render_status(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class ListenerController(ListenLifecycle):
    """判停逻辑装线 + listener signal 生产边界 + 运行时自解释.

    持有 listener (听) + asr (configure vad). once/always 是长时间运行的 async method,
    内部管理一条 listening session 的生命周期; 结果经 listener 的观察面 (on_recognition_*)
    流出.

    继承 ``ListenLifecycle``: moss runtime 只认这个生命周期表面 (enter/exit) 治理听侧,
    不经 IoC / provider 拿完整 concrete — 判停/信号/自解释那面还在演化.

    三个职责:
    - 判停 (on_event_creating 决定何时 commit);
    - 信号发射 (注入 signal_broadcast 时把识别事件翻译成 listener signal 广播);
    - 自解释 (追踪当前礼仪, 提供 ``snapshot()`` 与随身 ``as_channel()``).
    """

    def __init__(
            self,
            *,
            listener: Listener,
            asr: ASR,
            logger: Optional[LoggerItf] = None,
            signal_broadcast: Optional[Callable[[Signal], None]] = None,
            stop_caller_factory: Optional[Callable[[str], MossLLMCaller]] = None,
    ):
        self._listener = listener
        self._asr = asr
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[ListenerController]"
        self._active_task: Optional[asyncio.Task] = None
        self._owns_listener = False
        # 出口位点的 classifier 依赖: 一个 (instruction) -> caller 的装配函数.
        # 没有它时 classifier 静默缺席, 礼仪退回纯 silence/keywords.
        self._stop_caller_factory = stop_caller_factory
        self._stop_detector_factory: Optional[StopDetectorFactory] = None
        self._score_observers: list[Callable[[StopScoreObservation], None]] = []
        # 信号发射: 存在 sink 时注册一条 listener 级观察者 (跨 session 稳定), 机械地把
        # 每个识别事件翻译成 listener signal 并广播. 无 sink 则只做判停, 不发 signal.
        self._signal_broadcast = signal_broadcast
        if signal_broadcast is not None:
            listener.on_recognition_result(self._emit_event)

        self._channel: Optional[Channel] = None
        # 礼仪配置化: 当前激活礼仪 (首包/尾包协议读它) + config store (持久化).
        # ``_active_etiquette`` 是"当前礼仪"的唯一真值 — snapshot / notice / signal
        # 发射都从这里读, 不设第二真值.
        self._active_etiquette: Optional[EtiquetteSpec] = None
        self._config_store: Optional[ConfigStore] = None
        self._etiquette_config_cache: Optional[EtiquetteConfig] = None
        # clause → topic 装线 (懒, 由 with_topic_service 启动).
        self._topic_task: Optional[asyncio.Task] = None
        self._topic_disposer: Optional[Callable[[], None]] = None
        self._topic_publisher: Optional[Publisher] = None
        # audio sample → topic 装线 (懒, 由 with_audio_sample_service 启动).
        self._audio_sample_task: Optional[asyncio.Task] = None
        self._audio_sample_disposer: Optional[Callable[[], None]] = None
        self._audio_sample_publisher: Optional[Publisher] = None
        # segment buffer (感知协议): 跨 session 订阅 text/segment, 拉模式读.
        # 关时不入历史, 开时保留 + 经 notice/pull 暴露; 由 retain.enabled 门控.
        self._buffer = SegmentBuffer()
        self._listener.on_recognition_result(self._on_buffer_event)
        self._listener.on_recognition_segment(self._on_buffer_segment)

    # ── 生命周期: listener 未启动则托管, 已启动则只借用 ──

    async def __aenter__(self) -> Self:
        # 宿主已把 listener 启动 (is_running) → 只借用, 不重复 enter, 退出也不代它关.
        if not self._listener.is_running():
            await self._listener.__aenter__()
            self._owns_listener = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        await self._close_topic_wiring()
        await self._close_audio_sample_wiring()
        if self._owns_listener:
            self._owns_listener = False
            await self._listener.__aexit__(exc_type, exc_val, exc_tb)

    # ── 礼仪配置 (config store + 当前激活礼仪) ──

    def with_config_store(self, store: ConfigStore) -> Self:
        """注册 ConfigStore: 礼仪修改 save=True 时写回环境配置; 否则只内存."""
        self._config_store = store
        return self

    def with_stop_detector(self, factory: StopDetectorFactory) -> Self:
        """替换出口位点的默认装配 (整段换掉判停单元, 含降级件的注册方式)."""
        self._stop_detector_factory = factory
        return self

    def can_stop_judge(self) -> bool:
        """出口位点的 classifier 是否可用 (caller factory 已注入)."""
        return self._stop_caller_factory is not None

    def on_score(self, callback: Callable[[StopScoreObservation], None]) -> Callable[[], None]:
        """注册判停打分观察者 (跨 session) — 每次 llm 打分回调请求+结果, 返回 disposer."""
        self._score_observers.append(callback)
        return lambda: self._score_observers.remove(callback)

    def _notify_score(self, obs: StopScoreObservation) -> None:
        for callback in list(self._score_observers):
            try:
                callback(obs)
            except Exception:
                self._logger.exception("on_score observer failed")

    def _etiquette_config(self) -> EtiquetteConfig:
        """当前礼仪配置: 有 store 则 get_or_create, 否则内存实例."""
        if self._etiquette_config_cache is None:
            self._etiquette_config_cache = (
                self._config_store.get_or_create(EtiquetteConfig())
                if self._config_store is not None
                else EtiquetteConfig()
            )
        return self._etiquette_config_cache

    def etiquette_config(self) -> EtiquetteConfig:
        """开放当前礼仪配置 (所有已定义礼仪 + 默认激活)."""
        return self._etiquette_config()

    def active_etiquette(self) -> EtiquetteSpec | None:
        """当前激活的礼仪 spec (真值)."""
        return self._active_etiquette

    def set_etiquette_spec(self, spec: EtiquetteSpec, *, save: bool = False) -> None:
        """增/改一种礼仪 (内存). save=True 且有 config store 时写回环境配置."""
        config = self._etiquette_config()
        config.upsert(spec)
        if save and self._config_store is not None:
            self._config_store.save(config)

    def _set_active_etiquette(self, spec: EtiquetteSpec | None) -> None:
        """设当前激活礼仪, 并让 segment buffer 容量跟随 spec.retain.history."""
        self._active_etiquette = spec
        if spec is not None:
            self._buffer.resize(spec.retain.history)

    # ── 礼仪驱动状态机 (纯配置, 持续监听 = 另一种 always) ──

    def run_etiquette(
            self,
            etiquette: EtiquetteSpec,
            *,
            timeout: float | None = None,
            until_tail: bool = False,
    ) -> asyncio.Future:
        """传入礼仪配置, 启动持续监听状态机 (由首包/尾包/判停三层驱动).

        ``until_tail=True`` 时会话在尾包处理后结束 (听一次); 否则常驻到 timeout /
        被新礼仪取消.
        """
        self._set_active_etiquette(etiquette)
        self._cancel_active()
        task = asyncio.create_task(
            self._run_etiquette(etiquette, timeout=timeout, until_tail=until_tail)
        )
        self._active_task = task
        return task

    def start_default_etiquette(self) -> asyncio.Future:
        """启动默认礼仪 (config.default); 未定义则 always 配置化并激活."""
        config = self._etiquette_config()
        spec = config.active()
        if spec is None:
            spec = ALWAYS.model_copy(deep=True)
            config.upsert(spec)
            config.activate(spec.name)
        return self.run_etiquette(spec)

    async def _run_etiquette(
            self,
            etiquette: EtiquetteSpec,
            *,
            timeout: float | None,
            until_tail: bool = False,
    ) -> None:
        state = await self._listener.listen()
        judge = self._make_stop_judge(etiquette, state.commit)
        state.on_event_creating(judge.feed)
        done = asyncio.Event()
        if until_tail:
            # 结束条件 = 尾包 (TAIL) 已处理, 不是 segment 切分 — segment 切分早于
            # TAIL 经 _pump 从 queue 取出, 用 segment 判结束会丢尾包 signal.
            def _on_result(result: RecognitionEvent) -> None:
                if result.phase == RecognitionPhase.TAIL:
                    done.set()

            state.on_recognition_result(_on_result)
        async with state:
            try:
                if until_tail:
                    try:
                        await asyncio.wait_for(done.wait(), timeout)
                    except asyncio.TimeoutError:
                        self._logger.warning(
                            "%s no tail within %s — session ends", self._log_prefix, timeout,
                        )
                elif timeout is None:
                    await asyncio.Event().wait()
                else:
                    await asyncio.sleep(timeout)
            finally:
                judge.close()
                # 只在自己仍是当前礼仪时清 — 避免被下一次 run_etiquette 已切走后误清.
                if self._active_etiquette is etiquette:
                    self._set_active_etiquette(None)

    def _make_stop_judge(self, etiquette: EtiquetteSpec, commit: Callable[[], None]) -> StopJudge:
        """出口位点装配: 按 StopSpec 组装判停单元 + 按 classifier.instruction 建 caller."""
        if self._stop_detector_factory is not None:
            return self._stop_detector_factory(etiquette, commit)
        stop = etiquette.stop
        classifier = stop.classifier
        caller = None
        if classifier is not None and self._stop_caller_factory is not None:
            caller = self._stop_caller_factory(classifier.instruction)
        if caller is None:
            return StopJudge(
                caller=None,
                judge=False,
                segment_vad=stop.silence,
                commit=commit,
                keywords=stop.keywords,
                logger=self._logger,
            )
        return StopJudge(
            caller=caller,
            judge=True,
            threshold=classifier.threshold,
            judge_delay=classifier.delay,
            segment_vad=stop.silence,
            commit=commit,
            keywords=stop.keywords,
            context=classifier.context,
            on_score=self._notify_score,
            logger=self._logger,
        )

    # ── 聆听礼仪 ──

    def _resolve_spec(self, name: str, fallback: EtiquetteSpec) -> EtiquetteSpec:
        """按 name 从 config 取礼仪 (支持用户 upsert 的自定义版本), 缺则回退开箱 global.

        便利方法 (once/always/scored) 与 activate command 共用同一份配置真值 —— 用户
        用 set_etiquette_spec 改过的礼仪, 便利方法也读得到.
        """
        spec = self._etiquette_config().get(name)
        return (spec if spec is not None else fallback).model_copy(deep=True)

    def once(
            self,
            *,
            clause_vad: Optional[int] = None,
            keywords: Optional[list[str]] = None,
            timeout: float = 60.0,
    ) -> asyncio.Future:
        """半双工: 拿 clause 立刻 commit, 尾包后结束一次聆听.

        与 ``always`` 走同一条出口位点, 差别只在礼仪配置 (silence=0 → 首个 clause
        即端点) 与会话结束策略 (尾包后收). 立即返回 Future; 新 method 取消旧的状态机.

        Base 走 config.get("once"), 缺则回退开箱 ``once``; ``keywords`` 覆盖 stop.keywords.
        """
        self._apply_clause_vad(clause_vad)
        spec = self._resolve_spec("once", ONCE)
        if keywords:
            spec.stop.keywords = list(keywords)
        return self.run_etiquette(spec, timeout=timeout, until_tail=True)

    def always(
            self,
            *,
            clause_vad: Optional[int] = None,
            silence: float = 1.5,
            keywords: Optional[list[str]] = None,
            timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """持续聆听: clause 后等待 silence 秒静默 commit.

        立即返回 Future; ``timeout=None`` 表示常驻 (直到 ``stop()`` 或新礼仪取消).
        命中 keywords 的 clause 立刻 commit (不等静默).

        Base 走 config.get("always"), 缺则回退开箱 ``always``; ``silence`` / ``keywords``
        覆盖 stop 对应字段.
        """
        self._apply_clause_vad(clause_vad)
        spec = self._resolve_spec("always", ALWAYS)
        spec.stop.silence = silence
        if keywords:
            spec.stop.keywords = list(keywords)
        return self.run_etiquette(spec, timeout=timeout)

    def scored(
            self,
            *,
            clause_vad: Optional[int] = None,
            silence: float = 3.0,
            delay: float = 0.3,
            keywords: Optional[list[str]] = None,
            threshold: int = 7,
            timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """分类器判停: 出口位点挂一个 classifier, 打分到阈值提前 commit, silence 兜底.

        需要构造时注入了 ``stop_caller_factory``, 否则 classifier 静默缺席 (退化为 always).

        Base 走 config.get("scored"), 缺则回退开箱 ``scored``; classifier 缺席时
        ``threshold`` / ``delay`` 无处覆盖, 静默忽略 (退化即接受).
        """
        self._apply_clause_vad(clause_vad)
        spec = self._resolve_spec("scored", SCORED)
        spec.stop.silence = silence
        if spec.stop.classifier is not None:
            spec.stop.classifier.threshold = threshold
            spec.stop.classifier.delay = delay
        if keywords:
            spec.stop.keywords = list(keywords)
        return self.run_etiquette(spec, timeout=timeout)

    def stop(self) -> None:
        """停止聆听: 取消活跃 session."""
        self._set_active_etiquette(None)
        self._cancel_active()

    def pause(self, toggle: bool = True) -> None:
        """急停/恢复 (ListenLifecycle 表面): True 停听, False 恢复默认礼仪."""
        if toggle:
            self.stop()
        else:
            self.start_default_etiquette()

    def snapshot(self) -> ListenerSnapshot:
        """合成当前状态快照 (active etiquette name + listening + ASR 参数值).

        ``etiquette`` = 当前激活礼仪 name, 无激活则 ``"off"`` — 与 ``_active_etiquette``
        同源, 无第二真值.
        """
        info = self._asr.get_info()
        active = self._active_etiquette
        return ListenerSnapshot(
            etiquette=active.name if active is not None else "off",
            listening=self._listener.is_listening(),
            asr_params=dict(info.params),
        )

    # ── 观察面 (供外部消费者) ──

    def on_recognition_result(self, callback: Callable[[RecognitionEvent], None]) -> Callable[[], None]:
        """跨 session 观察识别结果 (CLI 渲染等). 委托给 listener."""
        return self._listener.on_recognition_result(callback)

    async def listen(self) -> ListenerState:
        """开一条裸 listening session (未启动) — 供 enter 等需手动 commit 的消费者."""
        return await self._listener.listen()

    # ── clause → topic 装线 ──

    async def with_topic_service(self, service: TopicService) -> None:
        """懒装线: 把识别到的 CLAUSE 发布成 ClauseTopic 到 ``service``.

        启动一个内部持有的 drain task: ``listener.on_recognition_result`` 的回调把 CLAUSE
        结果线程安全入队, task 出队 pub. 说话人身份 role 固定 user (听侧).
        """
        publisher = service.model_publisher(creator="listener", model=ClauseTopic)
        queue: janus.Queue = janus.Queue()

        def _on_clause(result: RecognitionEvent) -> None:
            if result.phase != RecognitionPhase.CLAUSE:
                return
            clause = result.clause
            text = clause.text if clause else result.text
            queue.sync_q.put_nowait(ClauseTopic(text=text, role="user"))

        async def _drain() -> None:
            while True:
                topic = await queue.async_q.get()
                publisher.pub(topic)

        await publisher.__aenter__()
        self._topic_disposer = self._listener.on_recognition_result(_on_clause)
        self._topic_task = asyncio.create_task(_drain())
        self._topic_publisher = publisher

    async def _close_topic_wiring(self) -> None:
        """关闭 clause→topic 装线 (cancel drain task + dispose 回调 + 退出 publisher)."""
        if self._topic_disposer is not None:
            self._topic_disposer()
            self._topic_disposer = None
        if self._topic_task is not None:
            self._topic_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._topic_task
            self._topic_task = None
        if self._topic_publisher is not None:
            await self._topic_publisher.__aexit__(None, None, None)
            self._topic_publisher = None

    # ── audio sample → topic 装线 ──

    async def with_audio_sample_service(self, service: TopicService, *, sample_rate: int) -> None:
        """把捕获到的音频按 ~200ms 窗口广播成 AudioSampleTopic (role=user).

        与 clause 装线不同: 这是 latest-value-wins, 用 LatestAudioWindow 累积 + stale,
        一个周期 task 取走当前窗口算频谱发布, 无队列.
        """
        publisher = service.model_publisher(creator="listener", model=AudioSampleTopic)
        window = LatestAudioWindow()

        def _on_chunk(chunk: AudioChunk) -> None:
            samples = np.asarray(chunk.samples).ravel().astype(np.int16)
            if samples.size:
                window.append(samples)

        async def _emit() -> None:
            while True:
                await asyncio.sleep(AUDIO_SAMPLE_INTERVAL)
                pcm = window.take()
                if pcm is None:
                    continue
                spectrum = compute_spectrum(pcm)
                publisher.pub(AudioSampleTopic(
                    role="user",
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
        self._audio_sample_disposer = self._listener.on_audio_chunk(_on_chunk)
        self._audio_sample_task = asyncio.create_task(_emit())
        self._audio_sample_publisher = publisher

    async def _close_audio_sample_wiring(self) -> None:
        """关闭 audio sample → topic 装线 (dispose 回调 + cancel task + 退出 publisher)."""
        if self._audio_sample_disposer is not None:
            self._audio_sample_disposer()
            self._audio_sample_disposer = None
        if self._audio_sample_task is not None:
            self._audio_sample_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_sample_task
            self._audio_sample_task = None
        if self._audio_sample_publisher is not None:
            await self._audio_sample_publisher.__aexit__(None, None, None)
            self._audio_sample_publisher = None

    # ── 信号发射 (RecognitionEvent → listener signal) ──

    def _emit_event(self, result: RecognitionEvent) -> None:
        """识别事件 → 打断包/发送包 signal → 广播. 仅在有 sink 时注册本观察者.

        FIRST → 打断包 (complete=False + interrupt + WARNING);
        TAIL → 发送包 (complete=True + notify + INFO);
        CLAUSE/PARTIAL 不上行 (判停已在 listener 侧消化).
        """
        if result.phase == RecognitionPhase.FIRST:
            self._emit_interrupt(result)
        elif result.phase == RecognitionPhase.TAIL:
            self._emit_deliver(result)

    def _emit_interrupt(self, result: RecognitionEvent) -> None:
        """首包打断: 按当前礼仪的 onset 协议发射 (emit 关则不发射)."""
        onset = self._active_etiquette.onset if self._active_etiquette else OnsetSpec()
        if not onset.emit:
            return
        self._signal_broadcast(new_listener_signal(
            result.text,
            segment_id=result.segment_id,
            interrupt=onset.interrupt,
            complete=False,
            priority=onset.priority,
            description="listener:onset",
        ))

    def _emit_deliver(self, result: RecognitionEvent) -> None:
        """尾包发送: 按当前礼仪的 deliver 协议发射 (emit 关则不发射)."""
        deliver = self._active_etiquette.deliver if self._active_etiquette else DeliverSpec()
        if not deliver.emit:
            return
        self._signal_broadcast(new_listener_signal(
            result.text,
            segment_id=result.segment_id,
            interrupt=deliver.interrupt,
            mode=deliver.mode,
            complete=True,
            priority=deliver.priority,
            description="listener:deliver",
        ))

    # ── segment buffer 留存 (retain 协议门控) ──

    def _retain_enabled(self) -> bool:
        """当前激活礼仪是否开启留存槽位 (retain.enabled)."""
        spec = self._active_etiquette
        return spec is not None and spec.retain.enabled

    def _on_buffer_event(self, event: RecognitionEvent) -> None:
        """text axis 观察者: 留存开时更新当前增长全文, 关时不做任何事."""
        if not self._retain_enabled():
            return
        self._buffer.on_event(event)

    def _on_buffer_segment(self, segment: RecognitionSegment) -> None:
        """segment 签发观察者: 留存开时定稿入历史, 关时不做任何事."""
        if not self._retain_enabled():
            return
        self._buffer.on_segment(segment)

    # ── internals ──

    def _cancel_active(self) -> None:
        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()

    # ── internals ──

    def _apply_clause_vad(self, clause_vad: Optional[int]) -> None:
        """用 clause_vad 覆盖 ASR 分句判停时间, 保留其余 params."""
        if clause_vad is None:
            return
        params = dict(self._asr.get_info().params)
        params["end_window_size"] = clause_vad
        self._asr.configure(params)

    # ── 反身 channel ──

    def as_channel(self) -> Channel:
        """随身 channel (惰性构建并持有) — 把聆听礼仪暴露成模型可控制的命令面."""
        if self._channel is None:
            self._channel = self._build_channel()
        return self._channel

    def _build_channel(self) -> Channel:
        chan = new_channel(name="listener", description="语音输入控制 — 开启/关闭聆听、切换礼仪、调 ASR")
        self._register_channel_commands(chan)
        return chan

    def _register_channel_commands(self, chan: MutableChannel) -> None:
        @chan.build.instruction
        def instruction() -> str:
            info = self._asr.get_info()
            return (
                f"ASR audio contract: {info.sample_rate}Hz, {info.bits}-bit, {info.channel}ch.\n"
                f"ASR tunable params schema:\n"
                f"{json.dumps(info.params_schema, ensure_ascii=False)}\n"
                f"EtiquetteSpec json schema (for set_etiquette_spec):\n"
                f"{json.dumps(EtiquetteSpec.model_json_schema(), ensure_ascii=False)}"
            )

        @chan.build.named_notices
        def named_notices() -> dict[str, str | None]:
            # 温数据只暴露当前礼仪名称; 配置详情走 get_etiquette 读接口.
            # 没有激活礼仪时片段缺席 (None): 模型收到 <etiquette removed/> 墓碑, 而不是
            # 一个占位空值 — 空串在这里表示"不变", 会让模型一直以为旧礼仪还激活着.
            active = self._active_etiquette
            notice: dict[str, str | None] = {"etiquette": active.name if active else None}
            # 留存槽位: 开时暴露最近一条定稿 segment 的全文 (变了才重发); 关/空时 None 墓碑.
            if active is not None and active.retain.enabled:
                recent = self._buffer.peek_recent(1)
                notice["last_heard"] = recent[-1].text if recent else None
            else:
                notice["last_heard"] = None
            return notice

        @chan.build.command()
        async def set_etiquette_spec(text__: str, save: bool = False) -> str:
            """Set or update an etiquette; `text__` is a JSON string of EtiquetteSpec
            (schema in the instruction). `save=True` persists it via the config store.
            """
            spec = EtiquetteSpec.model_validate_json(text__)
            self.set_etiquette_spec(spec, save=save)
            return f"etiquette '{spec.name}' set"

        @chan.build.command(blocking=False)
        async def activate(name: str = "") -> str:
            """Run an etiquette — empty name runs the active/default one."""
            config = self.etiquette_config()
            spec = config.get(name) if name else config.active()
            if spec is None:
                return f"etiquette {name!r} not defined"
            if name:
                config.activate(name)
            self.run_etiquette(spec)
            return f"activated '{spec.name}'"

        @chan.build.command(blocking=False)
        async def stop() -> str:
            """Stop listening (run -> stop -> run lifecycle)."""
            self.stop()
            return "stopped"

        @chan.build.command()
        async def get_etiquette(name: str = "") -> str:
            """Read etiquette config: empty = names of all, non-empty = one full spec json."""
            config = self.etiquette_config()
            if name:
                spec = config.get(name)
                return spec.model_dump_json() if spec else f"etiquette {name!r} not defined"
            return json.dumps([s.name for s in config.etiquettes], ensure_ascii=False)

        @chan.build.command()
        async def get_asr_params() -> str:
            """Read ASR params (cold data, pulled on demand — not in notice)."""
            return json.dumps(self._asr.get_info().params, ensure_ascii=False)

        @chan.build.command()
        async def get_transcript(n: int = 0) -> str:
            """Pull the segment buffer: current growing text + recent n heard segments.

            `n=0` uses the active etiquette's retain.history; returns JSON with
            `enabled` (retain switch), `current`, `recent` (tail-n) and `forgotten`.
            Empty current/recent when retain is off or nothing heard yet.
            """
            active = self._active_etiquette
            enabled = active is not None and active.retain.enabled
            if n <= 0:
                n = active.retain.history if active else 8
            current = self._buffer.peek_current() if enabled else None
            recent = self._buffer.peek_recent(n) if enabled else []
            return json.dumps(
                {
                    "enabled": enabled,
                    "current": current.to_dict() if current else None,
                    "recent": [s.to_dict() for s in recent],
                    "forgotten": self._buffer.forgotten() if enabled else 0,
                },
                ensure_ascii=False,
            )

        @chan.build.command()
        async def configure_asr(params: dict) -> str:
            """Change ASR params — pass only the keys to change (schema is in the instruction)."""
            current = dict(self._asr.get_info().params)
            self._asr.configure({**current, **params})
            return "asr configured"


