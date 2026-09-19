"""Listener Controller — 判停逻辑装线 + listener signal 生产边界 + 运行时自解释.

判停逻辑 (聆听礼仪) 通过 ListenerState.on_event_creating 挂载到识别层 (inline await),
判停时调 state.commit(). commit 机制 (发负序号切段) 已在 recognition 层, 这里只决定
何时调用.

信号发射是独立于判停的第二职责: 本层是 RecognitionEvent (text axis) → listener signal
(first/clause/tail) 的生产边界. 构造时注入 ``signal_broadcast`` (signal sink), 存在时
注册一条 listener 级 on_recognition_result 观察者, 机械地把每个识别事件翻译成 listener
signal 并广播.

第三职责是运行时自解释: 追踪当前礼仪 (off/once/always), 提供合成快照 (``snapshot()``)
与随身 channel (``as_channel()``), 让模型能判断"耳朵开没开、什么模式".
"""
import asyncio
import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Optional

import janus
import numpy as np
from typing_extensions import Self
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.asr import ASR, RecognitionEvent, RecognitionPhase
from ghoshell_moss.contracts.audio import (
    AUDIO_SAMPLE_INTERVAL,
    AudioChunk,
    LatestAudioWindow,
    compute_spectrum,
)
from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.llms import MossLLMCaller
from ghoshell_moss.contracts.listener import Listener, ListenerState
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.core.blueprint.mindflow import ChallengeMode, Priority, Signal
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import Publisher, TopicService
from ghoshell_moss.core.mindflow.listener_nucleus import new_listener_signal
from ghoshell_moss.host.listener.etiquette import (
    DeliverSpec,
    EtiquetteConfig,
    EtiquetteSpec,
    FirstPacketSpec,
    new_always_spec,
    new_llm_judge_spec,
    new_once_spec,
)
from ghoshell_moss.host.listener.stop_judge import StopJudge, StopScoreObservation
from ghoshell_moss.types.topics import AudioSampleTopic, ClauseTopic

__all__ = [
    "ListenerController",
    "ModelListenerController",
    "ListenEtiquette",
    "ListenerSnapshot",
]


class ListenEtiquette(str, Enum):
    """聆听礼仪 — 判停策略, 决定"什么时候算说完"."""

    OFF = "off"
    ONCE = "once"
    ALWAYS = "always"
    LLM_JUDGE = "llm_judge"


@dataclass
class ListenerSnapshot:
    """合成快照 — 状态 (mode/listening) + ASR 当前参数值.

    温数据, 进 notice, 变了才重发. 参数 schema 是冷数据 (进 instruction), 不在此.
    """

    mode: str
    listening: bool
    asr_params: dict

    def render_notice(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def render_status(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class ListenerController:
    """判停逻辑装线 + listener signal 生产边界 + 运行时自解释.

    持有 listener (听) + asr (configure vad). once/always 是长时间运行的 async method,
    内部管理一条 listening session 的生命周期; 结果经 listener 的观察面 (on_recognition_*)
    流出.

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
    ):
        self._listener = listener
        self._asr = asr
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[ListenerController]"
        self._active_task: Optional[asyncio.Task] = None
        self._owns_listener = False
        # 信号发射: 存在 sink 时注册一条 listener 级观察者 (跨 session 稳定), 机械地把
        # 每个识别事件翻译成 listener signal 并广播. 无 sink 则只做判停, 不发 signal.
        self._signal_broadcast = signal_broadcast
        if signal_broadcast is not None:
            listener.on_recognition_result(self._emit_event)

        self._mode: ListenEtiquette = ListenEtiquette.OFF
        self._channel: Optional[Channel] = None
        # 礼仪配置化: 当前激活礼仪 (首包/尾包协议读它) + config store (持久化).
        self._active_etiquette: Optional[EtiquetteSpec] = None
        self._config_store: Optional[ConfigStore] = None
        self._etiquette_config: Optional[EtiquetteConfig] = None
        # clause → topic 装线 (懒, 由 with_topic_service 启动).
        self._topic_task: Optional[asyncio.Task] = None
        self._topic_disposer: Optional[Callable[[], None]] = None
        self._topic_publisher: Optional[Publisher] = None
        # audio sample → topic 装线 (懒, 由 with_audio_sample_service 启动).
        self._audio_sample_task: Optional[asyncio.Task] = None
        self._audio_sample_disposer: Optional[Callable[[], None]] = None
        self._audio_sample_publisher: Optional[Publisher] = None

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

    def _etiquette_config(self) -> EtiquetteConfig:
        """当前礼仪配置: 有 store 则 get_or_create, 否则内存实例."""
        if self._etiquette_config is None:
            self._etiquette_config = (
                self._config_store.get_or_create(EtiquetteConfig())
                if self._config_store is not None
                else EtiquetteConfig()
            )
        return self._etiquette_config

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

    # ── 礼仪驱动状态机 (纯配置, 持续监听 = 另一种 always) ──

    def run_etiquette(
            self,
            etiquette: EtiquetteSpec,
            *,
            timeout: float | None = None,
    ) -> asyncio.Future:
        """传入礼仪配置, 启动持续监听状态机 (由首包/尾包/判停三层驱动)."""
        self._active_etiquette = etiquette
        self._cancel_active()
        task = asyncio.create_task(self._run_etiquette(etiquette, timeout=timeout))
        self._active_task = task
        return task

    def start_default_etiquette(self) -> asyncio.Future:
        """启动默认礼仪 (config.default); 未定义则 always 配置化并激活."""
        config = self._etiquette_config()
        spec = config.active()
        if spec is None:
            spec = new_always_spec()
            config.upsert(spec)
            config.activate(spec.name)
        return self.run_etiquette(spec)

    async def _run_etiquette(self, etiquette: EtiquetteSpec, *, timeout: float | None) -> None:
        state = await self._listener.listen()
        judge = self._make_stop_judge(etiquette, state.commit)
        state.on_event_creating(judge.feed)
        async with state:
            try:
                if timeout is None:
                    await asyncio.Event().wait()
                else:
                    await asyncio.sleep(timeout)
            finally:
                judge.close()

    def _make_stop_judge(self, etiquette: EtiquetteSpec, commit: Callable[[], None]) -> StopJudge:
        """判停组件: judge=False 纯 segment_vad, judge=True llm 打分. Model 覆盖加 caller."""
        stop = etiquette.stop
        return StopJudge(
            caller=None,
            judge=stop.judge,
            threshold=stop.threshold,
            segment_vad=stop.segment_vad,
            judge_delay=stop.judge_delay,
            commit=commit,
            keywords=stop.keywords,
            logger=self._logger,
        )

    # ── 聆听礼仪 ──

    def once(
            self,
            *,
            clause_vad: Optional[int] = None,
            keywords: Optional[list[str]] = None,
            timeout: float = 60.0,
    ) -> asyncio.Future:
        """半双工: 拿 clause 立刻 commit, 尾包后结束一次聆听.

        command 语义: 立即返回 Future (外部可 await 阻塞或忽略), 内部 spawn 状态机.
        新 method 调用 cancel 旧的状态机 (同一时刻至多一条 session).

        clause_vad 覆盖 ASR 分句判停时间 (end_window_size). keywords 在 once 语义下
        冗余 (clause 即 commit), 仅为接口一致保留.
        """
        self._mode = ListenEtiquette.ONCE
        self._active_etiquette = new_once_spec()
        self._cancel_active()
        task = asyncio.create_task(self._run_once(clause_vad=clause_vad, keywords=keywords, timeout=timeout))
        self._active_task = task
        return task

    def always(
            self,
            *,
            clause_vad: Optional[int] = None,
            segment_vad: float = 1.5,
            keywords: Optional[list[str]] = None,
            timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """持续聆听: clause 后等待 segment_vad, 活动信号 reset, 静默到 segment_vad commit.

        立即返回 Future; ``timeout=None`` 表示常驻 (直到 ``stop()`` 或新礼仪取消).
        命中 keywords 的 clause 立刻 commit (不等静默).
        """
        self._mode = ListenEtiquette.ALWAYS
        self._apply_clause_vad(clause_vad)
        spec = new_always_spec(segment_vad)
        if keywords:
            spec.stop.keywords = list(keywords)
        return self.run_etiquette(spec, timeout=timeout)

    def stop(self) -> None:
        """停止聆听: 取消活跃 session, 回到 off."""
        self._mode = ListenEtiquette.OFF
        self._cancel_active()

    def snapshot(self) -> ListenerSnapshot:
        """合成当前状态快照 (mode + listening + ASR 参数值)."""
        info = self._asr.get_info()
        return ListenerSnapshot(
            mode=self._mode.value,
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

    # ── 状态机 (内部, 每个 method 一个) ──

    async def _run_once(
            self,
            *,
            clause_vad: Optional[int],
            keywords: Optional[list[str]],
            timeout: float,
    ) -> None:
        self._apply_clause_vad(clause_vad)
        state = await self._listener.listen()
        committed = False
        done = asyncio.Event()

        async def on_event(event: RecognitionEvent) -> None:
            nonlocal committed
            if event.phase == RecognitionPhase.CLAUSE and not committed:
                committed = True
                state.commit()

        def on_result(result: RecognitionEvent) -> None:
            # 结束条件 = 尾包 (TAIL) 已处理, 不是 segment 切分 — segment 切分早于
            # TAIL 经 _pump 从 queue 取出, 用 segment 判结束会丢尾包 signal.
            if result.phase == RecognitionPhase.TAIL:
                done.set()

        state.on_event_creating(on_event)
        state.on_recognition_result(on_result)

        async with state:
            try:
                await asyncio.wait_for(done.wait(), timeout)
            except asyncio.TimeoutError:
                self._logger.warning("%s once: no tail within %.1fs", self._log_prefix, timeout)

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
        """首包打断: 按当前礼仪的 first_packet 协议发射 (barge_in 关则不发射)."""
        fp = self._active_etiquette.first_packet if self._active_etiquette else FirstPacketSpec()
        if not fp.barge_in:
            return
        self._signal_broadcast(new_listener_signal(
            result.text,
            segment_id=result.segment_id,
            interrupt=fp.interrupt,
            complete=False,
            priority=fp.priority,
            description="listener:barge-in",
        ))

    def _emit_deliver(self, result: RecognitionEvent) -> None:
        """尾包发送: 按当前礼仪的 deliver 协议发射."""
        dv = self._active_etiquette.deliver if self._active_etiquette else DeliverSpec()
        self._signal_broadcast(new_listener_signal(
            result.text,
            segment_id=result.segment_id,
            interrupt=dv.interrupt,
            mode=dv.mode,
            complete=True,
            priority=dv.priority,
            description="listener:deliver",
        ))

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
        def named_notices() -> dict[str, str]:
            # 温数据只暴露当前礼仪名称; 配置详情走 get_etiquette 读接口.
            active = self._active_etiquette
            return {"etiquette": active.name if active else ""}

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
        async def configure_asr(params: dict) -> str:
            """Change ASR params — pass only the keys to change (schema is in the instruction)."""
            current = dict(self._asr.get_info().params)
            self._asr.configure({**current, **params})
            return "asr configured"


class ModelListenerController(ListenerController):
    """ListenerController + llm func caller — 智能判停 (llm judge) 高阶礼仪.

    持 MossLLMCaller (外部装配, instruction/model/输出约束已绑定). ``llm_judge``
    是第四种聆听礼仪: clause 后由 llm 打分判「论述讲完了吗」, 打分 >= threshold
    即 commit, segment_vad 静默兜底。第五种 (快捷响应) 是后续礼仪, 不在本类。
    """

    def __init__(self, *, caller: MossLLMCaller, **kwargs) -> None:
        super().__init__(**kwargs)
        self._caller = caller
        self._score_observers: list[Callable[[StopScoreObservation], None]] = []

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

    def llm_judge(
            self,
            *,
            clause_vad: Optional[int] = None,
            segment_vad: float = 3.0,
            judge_delay: float = 0.3,
            keywords: Optional[list[str]] = None,
            threshold: int = 7,
            timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """LLM-judged stop detection: a segment_vad timer (fallback) + a debounced llm judge.

        Commit when the judge scores >= ``threshold`` (early) or after ``segment_vad``
        seconds of quiet past the last clause (baseline). Returns immediately;
        ``timeout=None`` means run until ``stop()`` or another etiquette cancels it.
        """
        self._mode = ListenEtiquette.LLM_JUDGE
        self._apply_clause_vad(clause_vad)
        spec = new_llm_judge_spec(segment_vad, threshold)
        spec.stop.judge_delay = judge_delay
        if keywords:
            spec.stop.keywords = list(keywords)
        return self.run_etiquette(spec, timeout=timeout)

    def _make_stop_judge(self, etiquette: EtiquetteSpec, commit: Callable[[], None]) -> StopJudge:
        """判停组件: 带 llm caller + 打分观察者 (base 无 caller)."""
        stop = etiquette.stop
        return StopJudge(
            caller=self._caller,
            judge=stop.judge,
            threshold=stop.threshold,
            segment_vad=stop.segment_vad,
            judge_delay=stop.judge_delay,
            commit=commit,
            keywords=stop.keywords,
            on_score=self._notify_score,
            logger=self._logger,
        )
