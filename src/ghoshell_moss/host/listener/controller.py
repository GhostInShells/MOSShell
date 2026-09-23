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

from ghoshell_moss.contracts.asr import ASR, ASRWithCorpus, RecognitionClause, RecognitionEvent, RecognitionPhase, RecognitionSegment
from ghoshell_moss.contracts.audio import (
    AUDIO_SAMPLE_INTERVAL,
    AudioChunk,
    LatestAudioWindow,
    compute_spectrum,
)
from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.llms import CallSettings, LLMFuncs, MossLLMCaller, MossLLMFuncs
from ghoshell_moss.contracts.listener import ListenLifecycle, Listener, ListenerState
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.core.blueprint.mindflow import ChallengeMode, Priority, Signal
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import Publisher, TopicService
from ghoshell_moss.core.mindflow.listener_nucleus import new_listener_signal
from ghoshell_moss.core.mindflow.knock_nucleus import new_knock_signal
from ghoshell_moss.host.listener.etiquette import (
    DeliverSpec,
    DetectSpec,
    EtiquetteConfig,
    EtiquetteSpec,
    OnsetSpec,
    always as ALWAYS,
    knock as KNOCK,
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
    "build_stop_caller_factory",
]


#: 出口位点的装配面: 吃 (礼仪, commit 开关), 返回该 session 的判停单元.
#: 默认实现由 ``ListenerController`` 按 ``StopSpec`` + 注入的 caller 组装;
#: 注入自定义 factory 即可整段替换 (container 在 runtime get 一次后塞进来).
StopDetectorFactory = Callable[[EtiquetteSpec, Callable[[], None]], StopJudge]


def build_stop_caller_factory(container) -> Optional[Callable[[str], MossLLMCaller]]:
    """出口位点 classifier 的依赖装配: 容器有模型环境 (MossLLMFuncs) 时返回
    (instruction) -> caller 工厂, 否则 None.

    None 不是错误 —— 默认出口本就不依赖模型; classifier 静默缺席, 礼仪退回
    silence/keywords. runtime 与 listener_node 共用同一装配, 避免两处各写一遍.
    """
    funcs = container.get(LLMFuncs)
    if not isinstance(funcs, MossLLMFuncs):
        return None

    def _factory(instruction: str) -> MossLLMCaller:
        return funcs.caller(
            instruction=instruction,
            tag="small_fast_model",
            settings=CallSettings(max_output_tokens=1),
        )
    return _factory


@dataclass
class ListenerSnapshot:
    """合成快照 — 当前激活礼仪 + listening + 人类锁 + ASR 当前参数值.

    ``etiquette`` = 当前激活礼仪 name (无则 "off") 与 ``_active_etiquette`` 同源;
    ``paused`` = 人类锁 (True 时 channel 从模型面消失, run_etiquette 被门控). 温数据,
    进 notice, 变了才重发. 参数 schema 是冷数据 (进 instruction), 不在此.
    """

    etiquette: str
    listening: bool
    paused: bool
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
            cell_name: str = "",
            config_store: Optional[ConfigStore] = None,
            topic_service: Optional[TopicService] = None,
    ):
        self._listener = listener
        self._asr = asr
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[ListenerController]"
        self._active_task: Optional[asyncio.Task] = None
        self._owns_listener = False
        # cell 标识 (matrix.this.name) — expect 超时 signal 的 source, 让模型区分
        # "这是某 cell 的 listener 系统提示" 与 "asr 识别结果" (source="asr").
        self._cell_name = cell_name
        # expect 状态机 — 纯异步超时提醒. 模型调 expect 立刻拿回执 index, 超时后
        # listener 主动发 notify signal 告知"预期语音输入没发生". 不打断模型行为.
        self._expect_seq = 0
        self._expect_events: dict[str, asyncio.Event] = {}
        self._expect_tasks: set[asyncio.Task] = set()
        # 出口位点的 classifier 依赖: 一个 (instruction) -> caller 的装配函数.
        # 没有它时 classifier 静默缺席, 礼仪退回纯 silence/keywords.
        self._stop_caller_factory = stop_caller_factory
        self._stop_detector_factory: Optional[StopDetectorFactory] = None
        self._score_observers: list[Callable[[StopScoreObservation], None]] = []
        # signal 发射观测 (TUI/debug 面, 非模型 notice): 每次 FIRST/TAIL 触发时回调
        # 一行 hint (发没发 / 发什么), 供观测"为什么 signal 没到" — 与 snapshot 分离.
        self._signal_observers: list[Callable[[str], None]] = []
        # 信号发射: 存在 sink 时注册一条 listener 级观察者 (跨 session 稳定), 机械地把
        # 每个识别事件翻译成 listener signal 并广播. 无 sink 则只做判停, 不发 signal.
        self._signal_broadcast = signal_broadcast
        if signal_broadcast is not None:
            listener.on_recognition_result(self._emit_event)
            # 词级置信度来源: 累积当前 segment 的 clause (emit 时才需要), TAIL 清零 —
            # 注册在 _emit_event 之后, 保证 TAIL 上 _emit_deliver 先读累积值、再被清零.
            listener.on_recognition_result(self._on_accumulate_clause)

        self._channel: Optional[Channel] = None
        # 礼仪配置化: 当前激活礼仪 (首包/尾包协议读它) + config store (持久化).
        # ``_active_etiquette`` 是"当前礼仪"的唯一真值 — snapshot / notice / signal
        # 发射都从这里读, 不设第二真值.
        self._active_etiquette: Optional[EtiquetteSpec] = None
        # 人类锁 — pause(True) 设置, pause(False) 释放. 只 TUI/人类主权面写.
        # 锁着时: run_etiquette 门控 (log + no-op) + as_channel() 的 chan.available
        # 返回 False → 整个 channel 从模型面消失 (无残留 affordance).
        self._human_paused: bool = False
        self._config_store: Optional[ConfigStore] = config_store
        self._etiquette_config_cache: Optional[EtiquetteConfig] = None
        # clause → topic 装线 (懒, 由 with_topic_service 在 __aenter__ 启动).
        self._topic_service: Optional[TopicService] = topic_service
        self._topic_task: Optional[asyncio.Task] = None
        self._topic_disposer: Optional[Callable[[], None]] = None
        self._topic_publisher: Optional[Publisher] = None
        # audio sample → topic 装线 (懒, 由 with_audio_sample_service 启动).
        self._audio_sample_task: Optional[asyncio.Task] = None
        self._audio_sample_disposer: Optional[Callable[[], None]] = None
        self._audio_sample_publisher: Optional[Publisher] = None
        # segment buffer (感知协议): 跨 session 订阅 text/segment, 拉模式读.
        # 关时不入历史, 开时保留 + 经 notice/pull 暴露; 由 deliver.emit=False 门控.
        self._buffer = SegmentBuffer()
        self._current_clauses: list[RecognitionClause] = []
        self._listener.on_recognition_result(self._on_buffer_event)
        self._listener.on_recognition_segment(self._on_buffer_segment)
        # expect 语音回调 (跨 session 稳定): 任何语音识别结果到达 → set 所有 pending
        # expect events, 让"期待输入"立刻满足 (取消超时). 不依赖 signal_broadcast.
        listener.on_recognition_result(self._on_expect_voice)

    # ── 生命周期: listener 未启动则托管, 已启动则只借用 ──

    async def __aenter__(self) -> Self:
        # 宿主已把 listener 启动 (is_running) → 只借用, 不重复 enter, 退出也不代它关.
        if not self._listener.is_running():
            await self._listener.__aenter__()
            self._owns_listener = True
        # topic 装线 (clause → ClauseTopic): 构造时注入的 service 在 async with 里才 await 启动.
        if self._topic_service is not None:
            await self.with_topic_service(self._topic_service)
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
        """设当前激活礼仪, 并让 segment buffer 容量跟随 spec.deliver.history."""
        self._active_etiquette = spec
        if spec is not None:
            self._buffer.resize(spec.deliver.history)

    def activate_etiquette(self, name: str = "") -> str:
        """按 name 激活礼仪并启动聆听; 空 name 用当前 default. 返回人类可读结果.

        channel 的 ``activate`` command 与 TUI 控制面共用此单一真值 — 都走
        ``run_etiquette``, 且受人类锁 (``_human_paused``) 门控.
        """
        config = self._etiquette_config()
        spec = config.get(name) if name else config.active()
        if spec is None:
            return f"etiquette {name!r} not defined"
        if name:
            config.activate(name)
        self.run_etiquette(spec)
        return f"activated '{spec.name}'"

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

        人类锁 (``_human_paused``) 门控: 锁着时记 warning + 返回已完成 future,
        不真启动 —— 保证模型侧 channel 消失的 same-truth 与运行时行为一致.
        """
        if self._human_paused:
            self._logger.warning(
                "%s run_etiquette blocked — listener is paused by human", self._log_prefix,
            )
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut
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
        detect = etiquette.stop.detect
        if detect is not None and detect.enabled:
            # 能量检测礼仪 (knock): 不做 ASR, 只读 meta.rms_db, 检测到声音发 knock signal.
            await self._run_knock(etiquette, detect)
            return

        state = await self._listener.listen()
        judge = self._make_stop_judge(etiquette, state.commit)
        state.on_event_creating(judge.feed)
        done = asyncio.Event()
        # 活动标记: 任何识别事件到达都 set, 供空闲超时重置计时 (until_tail 时顺带判 TAIL).
        activity = asyncio.Event()

        def _on_result(result: RecognitionEvent) -> None:
            activity.set()
            if until_tail and result.phase == RecognitionPhase.TAIL:
                done.set()

        state.on_recognition_result(_on_result)
        idle_triggered = False
        async with state:
            try:
                if until_tail:
                    # 结束条件 = 尾包 (TAIL) 已处理, 不是 segment 切分 — segment 切分早于
                    # TAIL 经 _pump 从 queue 取出, 用 segment 判结束会丢尾包 signal.
                    try:
                        await asyncio.wait_for(done.wait(), timeout)
                    except asyncio.TimeoutError:
                        self._logger.warning(
                            "%s no tail within %s — session ends", self._log_prefix, timeout,
                        )
                else:
                    idle_triggered = await self._wait_for_idle(activity, etiquette.idle_timeout, timeout)
            finally:
                judge.close()
                # 只在自己仍是当前礼仪时清 — 避免被下一次 run_etiquette 已切走后误清.
                if self._active_etiquette is etiquette:
                    self._set_active_etiquette(None)

        # 空闲超时 → 降级到 knock 中间态 (state 已退出, 当前 task 基本结束).
        if idle_triggered:
            self._degrade_to_knock()

    async def _wait_for_idle(
            self,
            activity: asyncio.Event,
            idle_timeout: float,
            timeout: float | None,
    ) -> bool:
        """常驻会话等待: idle_timeout 秒无识别事件 → 发 signal 并结束; timeout 为总时长上限.

        ``idle_timeout <= 0`` 退回旧的常驻 (timeout=None) / 固定总时长 (timeout=值) 语义 —
        空闲超时是叠加在常驻之上的可选层, 不改变旧行为. 返回 True 表示空闲超时触发 (应降级到 knock).
        """
        if idle_timeout <= 0:
            if timeout is None:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(timeout)
            return False
        start = time.monotonic()
        while True:
            if timeout is not None and time.monotonic() - start >= timeout:
                return False
            activity.clear()
            try:
                await asyncio.wait_for(activity.wait(), idle_timeout)
            except asyncio.TimeoutError:
                self._emit_idle_timeout(idle_timeout)
                return True

    async def _run_knock(self, etiquette: EtiquetteSpec, detect: DetectSpec) -> None:
        """knock 中间态: 轻量能量检测 (不做 ASR), 检测到声音发 knock signal.

        listener 不支持能量检测时静默结束 (退化 — 保持旧"彻底结束"语义). 常驻, 直到被
        下一次 run_etiquette (re-activate) cancel.
        """
        if not self._supports_sound_detection():
            return
        disposer = self._listener.on_sound_detected(lambda: self._emit_knock_signal(detect))
        self._listener.start_sound_detection(threshold_db=detect.threshold_db, cooldown=detect.cooldown)
        try:
            await asyncio.Event().wait()
        finally:
            disposer()
            self._listener.stop_sound_detection()

    def _degrade_to_knock(self) -> None:
        """空闲超时后降级到 knock 中间态 (延迟一个 tick, 等当前 task 完全结束)."""
        if not self._supports_sound_detection():
            return

        async def _degrade() -> None:
            await asyncio.sleep(0)
            self.run_etiquette(self._knock_spec())

        asyncio.create_task(_degrade())

    def _knock_spec(self) -> EtiquetteSpec:
        return self._resolve_spec("knock", KNOCK)

    def _supports_sound_detection(self) -> bool:
        """listener 是否支持能量检测 (真实 HostListener 支持, mock/Null 不支持)."""
        return (
            hasattr(self._listener, "start_sound_detection")
            and hasattr(self._listener, "stop_sound_detection")
            and hasattr(self._listener, "on_sound_detected")
        )

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
            if classifier is not None:
                self._logger.warning(
                    "%s classifier mounted (scored) but no stop caller — degrades to silence/keywords",
                    self._log_prefix,
                )
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
        """停止聆听: 取消活跃 session + 清空 pending expect (超时提醒不再发)."""
        self._set_active_etiquette(None)
        self._cancel_active()
        self._clear_expects()

    def pause(self, toggle: bool = True) -> None:
        """人类锁 (ListenLifecycle 表面): True 上锁 + 停听, False 释放 + 恢复默认礼仪.

        锁着期间 ``run_etiquette`` 被门控 + ``as_channel()`` 的 available 返回 False
        → 整个 listener channel 从模型面消失. TUI / 人类主权路径独占, 不给模型钥匙.
        """
        self._human_paused = toggle
        if toggle:
            self.stop()
        else:
            self.start_default_etiquette()

    def is_paused(self) -> bool:
        """当前是否被人类锁定 (channel 消失, run_etiquette 被门控)."""
        return self._human_paused

    def snapshot(self) -> ListenerSnapshot:
        """合成当前状态快照 (active etiquette name + listening + paused + ASR 参数值).

        ``etiquette`` = 当前激活礼仪 name, 无激活则 ``"off"``; ``paused`` = 人类锁 —
        与 ``_active_etiquette`` / ``_human_paused`` 同源, 无第二真值.
        """
        info = self._asr.get_info()
        active = self._active_etiquette
        return ListenerSnapshot(
            etiquette=active.name if active is not None else "off",
            listening=self._listener.is_listening(),
            paused=self._human_paused,
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

    def feed_ghost_clause(self, text: str) -> None:
        """说侧 ghost clause → corpus.tail (lines 投影入口).

        由 moss runtime 的统一 clause 接线在广播 ghost ClauseTopic 的同一处调用; 非
        corpus 能力的 ASR 静默 no-op (可降级)。
        """
        if isinstance(self._asr, ASRWithCorpus):
            self._asr.corpus().tail(text)

    async def with_topic_service(self, service: TopicService) -> None:
        """懒装线: 把识别到的 CLAUSE 发布成 ClauseTopic 到 ``service``.

        启动一个内部持有的 drain task: ``listener.on_recognition_result`` 的回调把 CLAUSE
        结果线程安全入队, task 出队 pub. 听侧固定 ``role="user"``; speaker_id /
        speaker_name 留空 (声纹预留, 启动时未知, 不 mock).
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

    def on_signal_emit(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """注册 signal 发射观测者 (TUI/debug 面) — 每次 FIRST/TAIL 触发回调一行 hint.

        跨 session 稳定; 返回 disposer. hint 形如 ``onset interrupt=True priority=WARNING``
        或 ``onset suppressed (emit=False)``. 独立于 snapshot (不进模型 notice).
        """
        self._signal_observers.append(callback)
        return lambda: self._signal_observers.remove(callback)

    def signal_wired(self) -> bool:
        """signal 发射面是否接通 (signal_broadcast 注入与否) — TUI/debug 读, 不进 snapshot.

        启动装配时即定、运行中不变; False 时 listener signal 静默不产, 只有 clause topic
        走通. 因为不变量而非温数据, 不放入模型 notice 的 snapshot.
        """
        return self._signal_broadcast is not None

    def _notify_signal(self, hint: str) -> None:
        for callback in list(self._signal_observers):
            try:
                callback(hint)
            except Exception:
                self._logger.exception("signal emit observer failed")

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
            self._notify_signal("onset suppressed (emit=False)")
            return
        self._signal_broadcast(new_listener_signal(
            result.text,
            segment_id=result.segment_id,
            interrupt=onset.interrupt,
            complete=False,
            priority=onset.priority,
            description="listener:onset",
        ))
        self._notify_signal(f"onset interrupt={onset.interrupt} priority={onset.priority.name}")

    def _emit_deliver(self, result: RecognitionEvent) -> None:
        """尾包发送: 按当前礼仪的 deliver 协议发射 (emit 关则不发射)."""
        deliver = self._active_etiquette.deliver if self._active_etiquette else DeliverSpec()
        if not deliver.emit:
            self._notify_signal("deliver suppressed (emit=False)")
            return
        self._signal_broadcast(new_listener_signal(
            result.text,
            segment_id=result.segment_id,
            interrupt=deliver.interrupt,
            mode=deliver.mode,
            low_conf=self._low_confidence_chars(deliver.low_confidence),
            complete=True,
            priority=deliver.priority,
            description="listener:deliver",
        ))
        self._notify_signal(
            f"deliver mode={deliver.mode or 'notify'} priority={deliver.priority.name}"
        )

    # ── expect 语音输入超时提醒 (纯异步, 不打断模型) ──

    def expect(self, timeout: float = 10.0) -> str:
        """期待一段语音输入在 ``timeout`` 秒内发生; 立刻返回回执 index.

        纯异步: 不阻塞调用方 (非 observe), 不打断模型当前行为. 若超时前有任何语音
        识别结果到达, 期待即满足 (自动取消, 不发任何 signal); 若超时仍未收到语音,
        listener 主动发一条 notify signal 告知"预期语音输入没发生", source 标记为
        本 cell 名 (``cell_name``), 与 asr 识别结果 (source="asr") 区分.

        回执 index 单调递增, 供模型关联"哪次 expect 超时". 返回字符串形如 ``"#3"``.
        """
        self._expect_seq += 1
        index = str(self._expect_seq)
        event = asyncio.Event()
        self._expect_events[index] = event
        task = asyncio.create_task(self._expect_wait(index, event, timeout))
        self._expect_tasks.add(task)
        task.add_done_callback(self._expect_tasks.discard)
        return index

    async def _expect_wait(self, index: str, event: asyncio.Event, timeout: float) -> None:
        """等语音 (event set) 或超时; 超时发 notify signal, 未超时静默退出."""
        try:
            await asyncio.wait_for(event.wait(), timeout)
            # 语音已到 → 期待满足, 不发任何 signal.
        except asyncio.TimeoutError:
            self._emit_expect_timeout(index, timeout)
        finally:
            # 移除自己的 event; 若语音回调已 discard 过 (见 _on_expect_voice), 这里幂等.
            self._expect_events.pop(index, None)

    def _on_expect_voice(self, result: RecognitionEvent) -> None:
        """语音识别结果到达 → set 所有 pending expect events (期待满足).

        挂 listener 级观察者 (跨 session 稳定). 任何 phase (FIRST/CLAUSE/PARTIAL/TAIL)
        都算"有语音", 因为期待的是"用户开口了", 不是"一句完整的话".
        """
        for event in list(self._expect_events.values()):
            event.set()

    def _clear_expects(self) -> None:
        """清空所有 pending expect (stop/暂停时): 取消 task + 清 event 表.

        不逐个 await — 取消即把 wait 从事件循环摘掉, 超时 signal 不再发.
        """
        for task in list(self._expect_tasks):
            if not task.done():
                task.cancel()
        self._expect_tasks.clear()
        self._expect_events.clear()

    def _emit_expect_timeout(self, index: str, timeout: float) -> None:
        """超时: 发一条 notify signal, 告知模型"预期语音输入没发生".

        source = cell_name (不是 "asr"), complete=True + interrupt=False + mode=notify —
        只旁路知会, 不打断模型. signal_broadcast 未注入时静默跳过.
        """
        if self._signal_broadcast is None:
            return
        self._signal_broadcast(new_listener_signal(
            f"expected voice input (expect #{index}) timed out after {timeout:g}s",
            source=self._cell_name or "listener",
            complete=True,
            interrupt=False,
            mode="notify",
            description="listener:expect-timeout",
        ))
        self._notify_signal(f"expect #{index} timed out after {timeout:g}s")

    def _emit_idle_timeout(self, idle_timeout: float) -> None:
        """空闲超时: 发一条 notify signal, 告知模型"长时间无语音, 降级到 knock 门铃".

        source = cell_name (不是 "asr"), complete=True + interrupt=False + mode=notify —
        只旁路知会, 不打断模型. signal_broadcast 未注入时静默跳过.
        """
        if self._signal_broadcast is None:
            return
        self._signal_broadcast(new_listener_signal(
            f"no speech for {idle_timeout:g}s — degrading to knock (sound detection)",
            source=self._cell_name or "listener",
            complete=True,
            interrupt=False,
            mode="notify",
            description="listener:idle-timeout",
        ))
        self._notify_signal(f"idle timeout after {idle_timeout:g}s")

    def _emit_knock_signal(self, detect: DetectSpec) -> None:
        """检测到声音: 发一条 knock signal, 告知模型"有声音, 但 ASR 关闭中".

        knock 语义 (losable): 模型忙时丢了也无所谓, 门铃的价值是"有空来应门".
        signal_broadcast 未注入时静默跳过.
        """
        if self._signal_broadcast is None:
            return
        self._signal_broadcast(new_knock_signal(
            "sound detected — ASR is off; re-activate listening to engage",
            priority=detect.priority,
            description="listener:knock",
        ))
        self._notify_signal(f"knock (sound detected, priority={detect.priority.name})")

    # ── 人类强发送 (输入法语义) ──

    def send_now(self) -> str:
        """人类强发送按钮 — 现在就把该交的交出去. 按当前是否自动发送分两种形态.

        - emit=False (留存/输入法): drain buffer, 有 sink 时发一条 deliver signal;
          空 buffer 提示空, 不发任何东西.
        - emit=True (自动发送): 强 commit, 走现有 commit 链路 (判停 → TAIL → deliver),
          不替代它自己去发 signal.

        无 signal sink 时副作用只有 drain (清空 buffer), 不发 signal. pause 门控无需
        额外处理 — pause 时 listener 已 stop, state 为 None, 调了也空转.
        """
        if self._retaining():
            return self._drain_and_send()
        return self._force_commit()

    def _drain_and_send(self) -> str:
        """emit=False: 取出 buffer 内容, 有 sink 时发 deliver signal."""
        items = self._buffer.drain()
        if not items:
            return "buffer empty — nothing to send"
        # 发送形状: 当前只拼 text. 未来 clause 会带 speaker_id / speaker_name (声纹),
        # 发送时要能按 speaker 组织, 不要把形状写死成"只有 text".
        text = "\n".join(s.text for s in items if s.text).strip()
        if not text:
            return "buffer empty — nothing to send"
        if self._signal_broadcast is not None:
            self._signal_broadcast(new_listener_signal(
                text,
                source="asr",
                complete=True,
                interrupt=False,
                mode="notify",
                priority=Priority.INFO,
                description="listener:send-now",
            ))
        return "sent"

    def _force_commit(self) -> str:
        """emit=True: 强 commit, 让现有 commit 链路自己发 deliver signal."""
        state = self._listener.state
        if state is None:
            return "no active listening session"
        state.commit()
        return "committed"

    def buffer_status(self) -> str:
        """buffer 里待发送内容的轻量摘要 — TUI 提示用 (非模型 notice).

        返回空串表示 buffer 空; 否则返回当前增长全文 + 定稿条数的单行摘要.
        """
        if not self._retaining():
            return ""
        current = self._buffer.peek_current()
        recent = self._buffer.peek_recent()
        parts: list[str] = []
        if current is not None and current.text:
            parts.append(current.text)
        if recent:
            parts.extend(s.text for s in recent if s.text)
        if not parts:
            return ""
        return " | ".join(parts)

    # ── segment buffer 留存 (deliver.emit=False 门控) ──

    def _retaining(self) -> bool:
        """当前激活礼仪是否留存 (deliver.emit=False 强制留存)."""
        spec = self._active_etiquette
        return spec is not None and not spec.deliver.emit

    def _on_buffer_event(self, event: RecognitionEvent) -> None:
        """text axis 观察者: 留存开时更新当前增长全文, 关时不做任何事."""
        if not self._retaining():
            return
        self._buffer.on_event(event)

    def _on_buffer_segment(self, segment: RecognitionSegment) -> None:
        """segment 签发观察者: 留存开时定稿入历史, 关时不做任何事."""
        if not self._retaining():
            return
        self._buffer.on_segment(segment)

    def _on_accumulate_clause(self, event: RecognitionEvent) -> None:
        """累积当前 segment 的 clause (词级置信度来源), TAIL 时清零.

        CLAUSE 与 TAIL 走同一条 FIFO queue (pump task), 故 TAIL 前本 segment 的 clause
        已全部到达; ``_emit_deliver`` 在 TAIL 上读累积值后由本观察者清零, 供下一 segment.
        """
        if event.phase == RecognitionPhase.CLAUSE and event.clause is not None:
            self._current_clauses.append(event.clause)
        elif event.phase == RecognitionPhase.TAIL:
            self._current_clauses = []

    def _low_confidence_chars(self, threshold: float | None) -> str:
        """当前 segment 里词级置信度 < threshold 的字 (拼接, 供 low_conf 属性); None → 空."""
        if threshold is None:
            return ""
        chars: list[str] = []
        for clause in self._current_clauses:
            for w in clause.words:
                if w.conf is not None and w.conf < threshold:
                    chars.append(w.text)
        return "".join(chars)

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
        # 人类锁门控 — 锁着时整个 channel 从模型面消失 (无残留命令 / notice), 不给
        # 模型撤销人类主权的路径; 解锁后完整回归.
        chan.build.available(lambda: not self._human_paused)

        @chan.build.instruction
        def instruction() -> str:
            info = self._asr.get_info()
            return (
                f"ASR audio contract: {info.sample_rate}Hz, {info.bits}-bit, {info.channel}ch.\n"
                f"Voice input is ASR transcription — it may contain homophones and near-sounds "
                f"(谐音). Judge by meaning and context, infer what the user likely said, and "
                f"ask for clarification when unsure.\n"
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
            if active is not None and not active.deliver.emit:
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
            return self.activate_etiquette(name)

        @chan.build.command(blocking=False)
        async def stop() -> str:
            """Stop listening (run -> stop -> run lifecycle)."""
            self.stop()
            return "stopped"

        @chan.build.command(blocking=False)
        async def expect(timeout: float = 10.0) -> str:
            """Expect voice input within `timeout` seconds; returns a receipt index immediately.

            Pure async — does not block or interrupt your current behavior. If any voice
            arrives before timeout, the expectation is satisfied silently. If it times out,
            the listener sends you a notify signal (`source` = this cell name, not "asr")
            saying the expected voice input did not happen. The receipt index lets you
            correlate which `expect` timed out (e.g. a later signal referencing "#3").
            """
            return self.expect(timeout)

        @chan.build.command(blocking=False)
        async def send_now() -> str:
            """Human-style strong send — commit what should be sent right now.

            When the etiquette does not auto-send (deliver.emit=False, input-method mode),
            this drains the retained buffer and sends it as one signal (empty buffer = no
            signal, just "buffer empty"). When auto-send is on, this force-commits the
            current turn early (before stop judging), letting the normal commit path deliver.
            """
            return self.send_now()

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

        # corpus 是 ASR 的可选能力面 — 没这能力的 ASR 整个 command 不暴露 (可降级).
        @chan.build.command(available=lambda: isinstance(self._asr, ASRWithCorpus))
        async def get_corpus() -> str:
            """Read the runtime ASR corpus (conditioning text): instruction + the ghost
            lines that will ride on the next segment. Cold data, pulled on demand.
            """
            if not isinstance(self._asr, ASRWithCorpus):
                return "corpus not supported by this ASR"
            return self._asr.corpus().model_dump_json()

        @chan.build.command(available=lambda: isinstance(self._asr, ASRWithCorpus))
        async def set_corpus_instruction(text__: str) -> str:
            """Replace the corpus prompt wholesale (set, not update); takes effect at the
            next segment. Runtime-only — not persisted across restarts.
            """
            if not isinstance(self._asr, ASRWithCorpus):
                return "corpus not supported by this ASR"
            self._asr.set_corpus_instruction(text__)
            return "corpus instruction set"

        @chan.build.command()
        async def get_transcript(n: int = 0) -> str:
            """Pull the segment buffer: current growing text + recent n heard segments.

            `n=0` uses the active etiquette's deliver.history; returns JSON with
            `enabled` (deliver.emit=False), `current`, `recent` (tail-n) and `forgotten`.
            Empty current/recent when emit=True or nothing heard yet.
            """
            active = self._active_etiquette
            enabled = active is not None and not active.deliver.emit
            if n <= 0:
                n = active.deliver.history if active else 8
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


