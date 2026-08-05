"""
Listener runtime — 蓝牙耳机近场 ASR 模块.

跟 g1/runtime/asr.py 对称: g1.asr 是 G1 内置远场 (整句 VAD, 非流式),
listener 是蓝牙耳机近场 (流式 partial + 按键 force drain).

设计纪律见同目录 README.md. 关键偏离 asr.py 的点 (有意识的):

  - 数据结构不复用 asr.AsrResult/AsrBatch. 流式 ASR 需要 partial 覆盖式快照 +
    finalized ring buffer 双层状态; asr.py 一条 = 一句的形状塞不进来.
  - 多线程拓扑: miniaudio OS audio 线程 → janus.sync_q → backend daemon (asyncio loop)
    跑 VolcengineASR.recognize. asr.py 是单 reader 线程.
  - start() 永远不抛. 配置文件缺失 / 蓝牙不在 / ws 异常 都是 health 状态而非
    异常. 单进程 g1 主进程启动时调一次, 不戴耳机也不挂 g1.

配置文件: ~/.moss_g1_listener.json
  - 由 _listener_sen_setup.py 探针生成 (或用户手改).
  - 不读环境变量 — 设备相关信息平台无关, 用文件更直观.

调用样例 (g1 main.py 形态):

    from ghoshell_moss_contrib.unitree.g1.runtime import listener
    listener.start()           # 同步, 幂等, 永远不抛
    # ...
    # 只读窥探 (给 context_messages 用, 不消费):
    recent = listener.peek_recent_finalized(10)   # tail-N 历史句
    partial = listener.peek_partial()             # 佩戴者正在说的 partial
    # 消费型 (给 signal 触发时用):
    batch = listener.drain()                            # 拿 finalized
    batch = listener.drain(force_finalize_partial=True) # 按键打断, 拿当前 partial 当 final
    h = listener.health()                               # 状态快照, 不消费
    # 显式开关 (用户按键 / TTS 回声抑制):
    listener.pause()           # 停 ASR 消费, capture 保持连接
    listener.resume()          # 恢复 ASR
    listener.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Callable, Literal, Optional

import janus
import numpy as np
from pydantic import BaseModel, Field

from ghoshell_moss.message import Message, unique_id

logger = logging.getLogger("moss.g1.runtime.listener")

# ── 常量 ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path.home() / ".moss_g1_listener.json"

# end_window_size: 服务端 VAD 静音判停阈值. 800ms 是实测平衡点 —
# 0.5s 太短句间换气会切断, 1s+ 单次回答等待太久. 实际交互不走 is_final 触发
# (走按键 force drain), 这个值只是兜底.
_DEFAULT_END_WINDOW_MS = 800

# 蓝牙断连后, capture 重试间隔. 5s 平衡响应速度 vs 日志噪声.
_DEVICE_RETRY_INTERVAL = 5.0

# ws 异常后, recognize 重试间隔. backoff 但有上限.
_WS_RETRY_MIN = 1.0
_WS_RETRY_MAX = 8.0

# capture 静默 → 怀疑设备死掉的超时. 蓝牙正常时即使无人说话也有静音帧上行.
_PCM_SILENCE_TIMEOUT = 10.0

# janus 队列上限. 50ms 一帧, 512 帧 ≈ 25 秒. 上限只是防内存爆炸, 正常不应到.
_JANUS_MAXSIZE = 512


# ── 数据契约 ──────────────────────────────────────────────────────────────
# Field description 直接进 LLM prompt (channel 层 model_json_schema 时).
# 措辞按对 LLM 解释精度写.

class Utterance(BaseModel):
    """一条 ASR 识别结果. 可能是 partial 也可能是 final."""

    id: str = Field(
        default_factory=unique_id,
        description="本条记录的 ulid. 同一句话的多次 partial 更新共享一个 id, "
                    "因此 partial → final 的过程在模型看来是同一条 utterance 在演进.",
    )
    text: str = Field(
        default="",
        description="识别到的文本. partial 阶段会随说话不断刷新, final 时定稿.",
    )
    is_final: bool = Field(
        default=False,
        description="是否最终结果. False = 还在说; True = 服务端 VAD 判停, 或按键 force drain.",
    )
    received_at: float = Field(
        default_factory=time.time,
        description="本条 (最新一次更新) 到达 backend 线程的本地时间 (time.time 秒).",
    )
    source: str = Field(
        default="g1.listener",
        description="数据来源固定常量. helper 加工成 Message/XML 时作为 tag.",
    )
    forced: bool = Field(
        default=False,
        description="是否因 drain(force_finalize_partial=True) 被强制 finalize 的. "
                    "True 表示模型/用户在 ASR VAD 判停前主动打断, 当前 partial 被当作完成的句子.",
    )


class UtteranceBatch(BaseModel):
    """drain 一次的返回. items 按到达升序, forgotten 是本批累积期间被 ring buffer 挤掉的条数."""

    items: list[Utterance] = Field(default_factory=list)
    forgotten: int = Field(
        default=0,
        description="自上次 drain 起被 ring buffer 挤掉的 utterance 数. 类似人类"
                    "'我有几句话没听清', 应当告知模型让它知道上下文有 gap.",
    )


HealthStatus = Literal[
    "stopped",       # 未 start 或 stop 后
    "no_config",     # ~/.moss_g1_listener.json 不存在, 跑 _listener_sen_setup.py 生成
    "no_device",     # 配置存在但匹配设备不在 (蓝牙未连/未识别)
    "device_down",   # 设备启动失败, 或运行中静默 (蓝牙中断), 后台重试中
    "ws_error",      # VolcengineASR ws 通讯异常, 后台 backoff 重连中
    "ok",            # 正常工作中
]


class ListenerHealth(BaseModel):
    """listener 状态快照. health() 拿到的就是这个, 不消费, 始终最新.

    channel 把这个序列化进 context_messages, 让模型可以回答 '你怎么听到我的声音的'.
    """

    status: HealthStatus = Field(
        default="stopped",
        description="当前主状态. ok 时正在监听; 其余表示原因.",
    )
    device_name: Optional[str] = Field(
        default=None,
        description="实际录音设备名 (miniaudio 报告的). 例: 'AirPods Pro (Hands-Free)'.",
    )
    device_pattern: Optional[str] = Field(
        default=None,
        description="配置文件里的设备匹配模式 (substring, 小写比较).",
    )
    sample_rate_capture: Optional[int] = Field(
        default=None,
        description="设备实际工作的采样率 (Hz). 蓝牙 HFP 常被强制 16kHz 或 8kHz, 与"
                    "配置的 sample_rate 可能不同.",
    )
    sample_rate_asr: Optional[int] = Field(
        default=None,
        description="送给火山引擎 ASR 的采样率 (Hz). 不等于 capture 时由 listener 内部重采样.",
    )
    channels: Optional[int] = Field(default=None)
    paused: bool = Field(
        default=False,
        description="用户是否显式暂停了 listener (pause() 调用). status 反映物理连接, "
                    "paused 反映用户意图, 二者正交. paused=True 时 ASR supervisor 不开新 "
                    "session, 内部 pcm 队列停止消费.",
    )
    started_at: Optional[float] = Field(default=None)
    last_pcm_at: Optional[float] = Field(
        default=None,
        description="最近一帧 PCM 到达 backend 线程的时间. 长时间不更新 = 怀疑设备死了.",
    )
    last_voiced_at: Optional[float] = Field(
        default=None,
        description="最近一次非静音帧的时间. 模型可据此说 '你刚才说话了'.",
    )
    pending_partial: Optional[str] = Field(
        default=None,
        description="当前未 finalize 的 partial 文本快照. 模型可据此说 '你话还没说完'.",
    )
    utterances_pending_drain: int = Field(
        default=0,
        description="finalized buffer 里待 drain 的 utterance 数.",
    )
    forgotten_since_last_drain: int = Field(default=0)
    bt_lost_count: int = Field(
        default=0,
        description="蓝牙设备从 ok 跌到 no_device/device_down 的累计次数. 用于诊断重连质量.",
    )
    ws_error_count: int = Field(default=0)
    error_count: int = Field(
        default=0,
        description="未预期异常累计. 高频增长 = 有未识别的失败模式, 看日志.",
    )
    last_error_msg: Optional[str] = Field(default=None)


# ── 模块级私有状态 ────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_listeners_lock = threading.Lock()

_finalized_dq: deque[Utterance] = deque(maxlen=32)
_partial: Optional[Utterance] = None
_forgotten_since_last_drain: int = 0

_sentence_listeners: dict[str, Callable[[Utterance], None]] = {}
_partial_listeners: dict[str, Callable[[Utterance], None]] = {}
_health_change_listeners: dict[str, Callable[[ListenerHealth], None]] = {}

_health: ListenerHealth = ListenerHealth()

_backend_thread: Optional[threading.Thread] = None
_backend_loop: Optional[asyncio.AbstractEventLoop] = None
_stop_event: Optional[asyncio.Event] = None  # asyncio Event, lives on backend loop
_started: bool = False

# 当前 recognize session. drain(force=True) 时设 abort, backend 立即结束本轮.
_current_session: Optional["_Session"] = None

# 用户显式暂停标志. 与 start/stop 生命周期正交, 与 status (物理连接) 正交.
# True 时 ASR supervisor 不开新 session; capture 仍在运行, janus 队列自然 ring-drop.
# resume 时 ASR supervisor 会在开新 session 前 flush 队列避免播老音频.
_paused: bool = False

# 配置 (start 时填入)
_config: Optional["_ListenerConfig"] = None


class _ListenerConfig(BaseModel):
    device_pattern: str
    device_name: Optional[str] = None  # setup 写入的精确设备名, _find_device 优先匹配
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 50
    asr_sample_rate: int = 16000
    end_window_ms: int = _DEFAULT_END_WINDOW_MS


class _Session:
    """一次 VolcengineASR.recognize() 调用的生命周期对象.

    backend 每开一次 recognize 创建新 session. force drain 时设 abort, 当前 session 内
    后续 partial 全部丢弃, 服务端送来的 final 也丢弃 (因为已经被强制 finalize 过了),
    然后开下一个 session.
    """

    __slots__ = ("id", "abort", "utterance_id")

    def __init__(self) -> None:
        self.id = unique_id()
        self.abort = asyncio.Event()
        # 一个 session 对应一句话, 共享一个 utterance_id, 跨多次 partial 更新.
        self.utterance_id = unique_id()


# ── 公开接口 ─────────────────────────────────────────────────────────────

def start(*, config_path: Optional[Path] = None) -> None:
    """启动 listener. 同步, 幂等, **永远不抛**.

    配置不存在 / 蓝牙不在 / ws 异常 → status 字段反映原因, 不抛.

    :param config_path: 默认 ~/.moss_g1_listener.json. 用 _listener_sen_setup.py 生成.
    """
    global _backend_thread, _started, _config

    with _state_lock:
        if _started:
            logger.debug("start() 重入 — 已 started, 跳过.")
            return

        # 1. 读配置. 失败 = no_config 状态, 不起 backend, 不抛.
        path = config_path or DEFAULT_CONFIG_PATH
        cfg = _load_config(path)
        if cfg is None:
            _set_status_locked("no_config", last_error_msg=f"config not found: {path}")
            _started = True  # 标记为 started, stop() 才能 clean reset
            logger.warning(
                "listener: config %s not found. run "
                "`python -m ghoshell_moss_contrib.unitree.g1.runtime._listener_sen_setup` to generate.",
                path,
            )
            return

        _config = cfg
        _set_status_locked(
            "stopped",  # 还没真正 ok, supervisor 启动后会改
            device_pattern=cfg.device_pattern,
            sample_rate_asr=cfg.asr_sample_rate,
            channels=cfg.channels,
            started_at=time.time(),
        )
        _started = True

    # 2. 起 backend daemon 线程. 内部维护 asyncio loop + supervisor 任务.
    _backend_thread = threading.Thread(
        target=_backend_thread_main,
        name="g1-listener-backend",
        daemon=True,
    )
    _backend_thread.start()
    logger.info("listener started (config=%s)", path)


def stop(timeout: float = 3.0) -> None:
    """停止 listener. 同步, 幂等. join backend 线程, 关 capture, 关 ws.

    daemon 线程随进程退出而死, timeout 内未 join 完成只 warn 不 raise.
    """
    global _backend_thread, _backend_loop, _stop_event, _started, _config, _current_session, _paused

    with _state_lock:
        if not _started:
            logger.debug("stop() 重入 — 未 started, 跳过.")
            return
        _started = False
        loop = _backend_loop
        stop_event = _stop_event
        thread = _backend_thread

    # 触发 backend 退出. _stop_event 是 asyncio.Event, 必须从 backend loop 内 set.
    if loop is not None and stop_event is not None and not loop.is_closed():
        try:
            loop.call_soon_threadsafe(stop_event.set)
        except RuntimeError:
            logger.debug("listener stop: loop already stopped")

    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "listener backend thread 未在 %.1fs 内 join 完成 (daemon, 随进程退出).",
                timeout,
            )

    with _state_lock:
        _backend_thread = None
        _backend_loop = None
        _stop_event = None
        _config = None
        _current_session = None
        _paused = False
        _set_status_locked("stopped", paused=False)

    logger.info("listener stopped.")


def is_running() -> bool:
    """主状态是否 ok. 注意: started 但 no_device 时也返回 False."""
    with _state_lock:
        return _health.status == "ok"


def drain() -> UtteranceBatch:
    """拿走 finalized buffer 内全部 utterance + 当前 partial (若有) + forgotten 计数.

    始终 abort 当前 recognize session, 无论 _partial 是否存在:
    - _partial 存在: 强制作为 forced=True utterance 拿走, abort session, 后续同 id 更新丢弃.
    - _partial 为 None: 用户刚停止说话但 is_final 还在 pipeline 里 (ASR 延迟窗口).
      此时 finalized 和 partial 都为空, drain 返回空 batch. 但 session 仍然 abort —
      in-flight is_final 被丢弃, 不会在下次 drain 时以"上一轮内容"重新出现.

    这解决了 "第二次按 A 又是上一轮的内容" bug: 不 abort session 时, is_final 在
    drain 后到达, 写入 _finalized_dq, 下次 drain 拿到.

    线程安全. 与 backend 线程的 _partial 覆盖 / _finalized_dq.append 串行共用 _state_lock.
    """
    global _forgotten_since_last_drain, _partial

    with _state_lock:
        items = list(_finalized_dq)
        _finalized_dq.clear()
        forgotten = _forgotten_since_last_drain
        _forgotten_since_last_drain = 0

        if _partial is not None:
            forced = _partial.model_copy(update={"is_final": True, "forced": True})
            items.append(forced)
            _partial = None

        # 始终 abort 当前 session — 无论 partial 是否存在.
        # 这是防止 in-flight is_final 污染下次 drain 的关键.
        abort_session = _current_session

        _refresh_pending_partial_locked()
        _refresh_utterances_pending_drain_locked()

    # abort 必须 lock 外做 — call_soon_threadsafe 自己拿 loop 内锁, 跟 _state_lock 没冲突.
    if abort_session is not None and _backend_loop is not None:
        try:
            _backend_loop.call_soon_threadsafe(abort_session.abort.set)
        except RuntimeError:
            logger.debug("drain: backend loop already stopped, abort skipped")

    return UtteranceBatch(items=items, forgotten=forgotten)


def peek_partial() -> Optional[Utterance]:
    """看当前未 finalize 的 partial 快照. 不消费, 不影响 forgotten."""
    with _state_lock:
        if _partial is None:
            return None
        return _partial.model_copy()


def peek_latest_finalized() -> Optional[Utterance]:
    """看 finalized buffer 末尾一条. 不消费."""
    with _state_lock:
        if not _finalized_dq:
            return None
        return _finalized_dq[-1]


def peek_recent_finalized(n: int = 10) -> list[Utterance]:
    """看 finalized buffer 末尾最近 n 条 (只读, 不消费, 不影响 forgotten).

    channel 层的 context_messages 用这个拿"最近听到的几句"塞给 ghost, tail -n 语义.
    与 drain() 是两条正交路径 — drain 是"消费型, 触发 signal 时把整批交出去",
    peek 是"只读窥探, 每回合装配时拿快照". 二者不互斥, 但同一 channel 内一般只走一条.
    """
    if n <= 0:
        return []
    with _state_lock:
        items = list(_finalized_dq)
    return [u.model_copy() for u in items[-n:]]


def pause() -> None:
    """暂停 listener 采集消费 — ASR supervisor 不开新 session, 当前 session abort.

    与 stop() 不同: capture 线程继续跑 (设备保持连接), 幂等. resume() 即刻恢复.
    未 start / no_config 状态下调用也合法, 只标记 _paused, resume 时行为一致.

    典型用途:
    - TTS 播放前 pause, 播完 resume — 避免自己听自己 (回声抑制的最硬门槛)
    - 用户按耳机 "关" 键 pause, 按 "开" 键 resume
    """
    global _paused
    abort_session: Optional[_Session] = None

    with _state_lock:
        if _paused:
            return
        _paused = True
        abort_session = _current_session
        _set_status_locked(_health.status, paused=True)

    if abort_session is not None and _backend_loop is not None:
        try:
            _backend_loop.call_soon_threadsafe(abort_session.abort.set)
        except RuntimeError:
            logger.debug("pause: backend loop already stopped, abort skipped")

    logger.info("listener paused.")


def resume() -> None:
    """恢复 listener. 幂等. ASR supervisor 下一轮循环起新 session.

    resume 前若 janus 队列里堆了 pause 期间的 PCM (最多 _JANUS_MAXSIZE 帧 ≈ 25s),
    supervisor 会在开 session 前 flush 队列, 避免拿老音频当新一句起头.
    """
    global _paused
    with _state_lock:
        if not _paused:
            return
        _paused = False
        _set_status_locked(_health.status, paused=False)
    logger.info("listener resumed.")


def is_paused() -> bool:
    """用户是否已 pause. 与 status (物理连接) 正交."""
    with _state_lock:
        return _paused


def health() -> ListenerHealth:
    """暴露当前状态快照. 不消费, channel/monitor 任何时候读."""
    with _state_lock:
        return _health.model_copy()


def register_sentence_listener(cb: Callable[[Utterance], None]) -> str:
    """注册 'is_final 时' 触发的回调. **跑在 backend asyncio 线程**, 不能阻塞.

    跨线程需求 (推回 ghost asyncio loop / queue) 由 cb 自行处理. 跟 sdk/_buttons.py
    范式一致 — runtime 不替 listener 决定线程模型.

    :return: handle (ulid str), 用于 unregister.
    """
    return _register(_sentence_listeners, cb)


def register_partial_listener(cb: Callable[[Utterance], None]) -> str:
    """注册 'partial 刷新时' 触发的回调. 高频 — 流式 ASR 一句话 20+ 次.

    channel 一般只用第一次 partial 触发 SPEECH_STARTED 早期 signal, 后续靠 peek_partial
    + health 拿状态, 不靠这里. 提供这个 API 是为完备性 + 可视化场景.

    **跑在 backend asyncio 线程**, 不能阻塞.
    """
    return _register(_partial_listeners, cb)


def register_health_change_listener(cb: Callable[[ListenerHealth], None]) -> str:
    """注册 'status 字段跳变时' 触发的回调. 只在 ok ↔ no_device ↔ device_down ↔
    ws_error 之间转换时触发, 不在内容更新时触发. 防刷屏.

    **跑在 backend asyncio 线程**, 不能阻塞.
    """
    return _register(_health_change_listeners, cb)


def unregister_listener(handle: str) -> None:
    """反注册任意一类 listener. 未知 handle 静默忽略."""
    with _listeners_lock:
        _sentence_listeners.pop(handle, None)
        _partial_listeners.pop(handle, None)
        _health_change_listeners.pop(handle, None)


# ── 内部: 配置加载 ────────────────────────────────────────────────────────

def _load_config(path: Path) -> Optional[_ListenerConfig]:
    """读 ~/.moss_g1_listener.json. 不存在或解析失败返回 None."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 过滤掉 _note 等仅供人类阅读的字段.
        # device_name_resolved → device_name: setup 写入的精确设备名, 用于 _find_device 优先匹配.
        if "device_name_resolved" in data and "device_name" not in data:
            data["device_name"] = data.pop("device_name_resolved")
        return _ListenerConfig.model_validate({
            k: v for k, v in data.items()
            if k in _ListenerConfig.model_fields
        })
    except Exception:
        logger.exception("listener: failed to load config %s", path)
        return None


# ── 内部: 状态管理 ────────────────────────────────────────────────────────

def _set_status_locked(status: HealthStatus, **fields) -> None:
    """更新 _health. 必须在持 _state_lock 时调. 不触发 callback."""
    global _health
    update = {"status": status, **fields}
    _health = _health.model_copy(update=update)


def _refresh_pending_partial_locked() -> None:
    global _health
    _health = _health.model_copy(update={
        "pending_partial": _partial.text if _partial is not None else None,
    })


def _refresh_utterances_pending_drain_locked() -> None:
    global _health
    _health = _health.model_copy(update={
        "utterances_pending_drain": len(_finalized_dq),
        "forgotten_since_last_drain": _forgotten_since_last_drain,
    })


def _transition_status(new_status: HealthStatus, **fields) -> None:
    """更新 status, 若实际跳变则触发 health_change_listeners. backend 线程内调."""
    with _state_lock:
        old = _health.status
        _set_status_locked(new_status, **fields)
        if old == new_status:
            return
        snapshot = _health.model_copy()
        if old == "ok" and new_status in ("no_device", "device_down"):
            _set_status_locked(new_status, bt_lost_count=_health.bt_lost_count + 1)
            snapshot = _health.model_copy()

    with _listeners_lock:
        callbacks = list(_health_change_listeners.values())
    for cb in callbacks:
        try:
            cb(snapshot)
        except Exception:
            logger.exception("listener health_change callback raised (isolated).")


def _register(registry: dict[str, Callable], cb: Callable) -> str:
    handle = unique_id()
    with _listeners_lock:
        registry[handle] = cb
    return handle


# ── 内部: backend 线程 + asyncio loop ─────────────────────────────────────

def _backend_thread_main() -> None:
    """backend 线程入口. 拥有自己的 asyncio loop, 永生直到 stop_event."""
    global _backend_loop, _stop_event

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    with _state_lock:
        _backend_loop = loop
        # asyncio.Event 必须在它将被 await 的 loop 下创建
        _stop_event = asyncio.Event()

    try:
        loop.run_until_complete(_backend_main())
    except Exception:
        logger.exception("listener backend asyncio main crashed")
    finally:
        try:
            loop.close()
        except Exception:
            pass


async def _backend_main() -> None:
    """backend 总入口. 起 capture + asr 两个 supervisor 任务, 等 stop_event."""
    # config 在 start() 时填好, 这里直接用
    cfg = _config
    assert cfg is not None, "_backend_main called without config"

    pcm_queue: janus.Queue = janus.Queue(maxsize=_JANUS_MAXSIZE)

    capture_task = asyncio.create_task(
        _capture_supervisor(pcm_queue, cfg),
        name="g1-listener-capture-supervisor",
    )
    asr_task = asyncio.create_task(
        _asr_supervisor(pcm_queue, cfg),
        name="g1-listener-asr-supervisor",
    )

    try:
        await _stop_event.wait()
    finally:
        capture_task.cancel()
        asr_task.cancel()
        await asyncio.gather(capture_task, asr_task, return_exceptions=True)
        # janus.Queue 也要关一下
        pcm_queue.close()
        await pcm_queue.wait_closed()


# ── 内部: capture supervisor ─────────────────────────────────────────────

async def _capture_supervisor(pcm_queue: janus.Queue, cfg: _ListenerConfig) -> None:
    """循环: 探测设备 → 启动 capture → 监视 → 失败重试. 永生直到取消."""
    while not _stop_event.is_set():
        try:
            await _capture_one_lifecycle(pcm_queue, cfg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("capture supervisor unexpected error: %s", e)
            with _state_lock:
                _set_status_locked(
                    _health.status,
                    error_count=_health.error_count + 1,
                    last_error_msg=f"capture: {e}",
                )

        if _stop_event.is_set():
            return
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=_DEVICE_RETRY_INTERVAL)
        except asyncio.TimeoutError:
            pass


async def _capture_one_lifecycle(pcm_queue: janus.Queue, cfg: _ListenerConfig) -> None:
    """一次 capture 生命周期: 探设备 → 起 → 监视静默 → 关. 任意阶段失败 raise 给上层重试."""
    import miniaudio

    # 1. 设备探测
    device_id, device_name = await asyncio.get_running_loop().run_in_executor(
        None, _find_device, cfg.device_pattern, cfg.device_name,
    )
    if device_id is None:
        _transition_status("no_device", device_name=None, sample_rate_capture=None)
        logger.debug("listener: no device matching pattern '%s'", cfg.device_pattern)
        raise RuntimeError(f"no device matching '{cfg.device_pattern}'")

    # 2. 启动 CaptureDevice
    try:
        capture = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=cfg.channels,
            sample_rate=cfg.sample_rate,
            buffersize_msec=cfg.frame_ms,
            device_id=device_id,
        )
    except Exception as e:
        _transition_status("device_down", last_error_msg=f"capture init: {e}")
        logger.warning("listener: CaptureDevice init failed: %s", e)
        raise

    # miniaudio 报告的实际采样率可能与请求不同 (蓝牙 HFP 可能强制 8k/16k)
    actual_sr = getattr(capture, "sample_rate", cfg.sample_rate)
    actual_name = getattr(capture, "name", "") or device_name or "<unknown>"

    last_pcm_at_ref = [time.time()]
    capture_gen = _make_capture_generator(
        sync_q=pcm_queue.sync_q,
        channels=cfg.channels,
        last_pcm_at_ref=last_pcm_at_ref,
    )
    next(capture_gen)  # prime

    try:
        capture.start(capture_gen)
    except Exception as e:
        _transition_status("device_down", last_error_msg=f"capture start: {e}")
        logger.warning("listener: capture.start failed: %s", e)
        capture.close()
        raise

    _transition_status(
        "ok",
        device_name=actual_name,
        sample_rate_capture=actual_sr,
        last_pcm_at=last_pcm_at_ref[0],
    )
    logger.info(
        "listener capture started: device=%r sr=%d ch=%d frame=%dms",
        actual_name, actual_sr, cfg.channels, cfg.frame_ms,
    )

    try:
        # 3. 监视: 周期检查最近 PCM 时间. 长时间静默 = 设备死了 (蓝牙断).
        while not _stop_event.is_set():
            try:
                await asyncio.wait_for(_stop_event.wait(), timeout=1.0)
                break  # stop_event 触发
            except asyncio.TimeoutError:
                pass

            now = time.time()
            silence = now - last_pcm_at_ref[0]
            with _state_lock:
                _set_status_locked(_health.status, last_pcm_at=last_pcm_at_ref[0])

            if silence > _PCM_SILENCE_TIMEOUT:
                logger.warning(
                    "listener: no PCM for %.1fs, suspecting device down", silence,
                )
                _transition_status(
                    "device_down",
                    last_error_msg=f"no PCM for {silence:.1f}s",
                )
                raise RuntimeError("capture silent timeout")
    finally:
        try:
            capture.stop()
        except Exception:
            logger.exception("capture.stop failed (ignored)")
        try:
            capture.close()
        except Exception:
            logger.exception("capture.close failed (ignored)")


def _find_device(pattern: str, device_name: Optional[str] = None):
    """返回 (device_id, device_name) 或 (None, None).

    匹配优先级:
      1. device_name (精确设备名, 来自 setup 写入的 device_name_resolved)
      2. pattern (配置里的模糊匹配 pattern, 小写 substring)

    miniaudio.Devices().get_captures() 返回 list[dict] (本仓库装的版本),
    dict 字段: name / type / id (cdata, 可直接传 CaptureDevice) / formats.
    host/voice/capture/miniaudio_capture.py 里写的 .capture 是更老 API 的 typo,
    本仓库当前版本会 AttributeError, 那条路径事实上 fallback 到默认设备.
    """
    import miniaudio
    pat = pattern.lower() if pattern else ""
    try:
        devs = miniaudio.Devices().get_captures()
    except Exception as e:
        logger.warning("device enumeration failed: %s", e)
        return None, None

    # pass 1: 精确设备名匹配
    if device_name:
        for d in devs:
            name = d.get("name", "") if isinstance(d, dict) else getattr(d, "name", "")
            if name == device_name:
                dev_id = d.get("id") if isinstance(d, dict) else getattr(d, "id", None)
                return dev_id, name

    # pass 2: pattern 子串匹配 (不包含 "Monitor of" 的设备优先, 避免 Monitor of X
    # 抢在 X 之前被命中 — PulseAudio monitor 设备总是排在实体设备前面)
    best = None
    best_is_monitor = False
    for d in devs:
        name = d.get("name", "") if isinstance(d, dict) else getattr(d, "name", "")
        if pat in name.lower():
            is_monitor = name.startswith("Monitor of")
            if not is_monitor:
                dev_id = d.get("id") if isinstance(d, dict) else getattr(d, "id", None)
                return dev_id, name  # 非 monitor 匹配立即返回
            if best is None:
                best = (d.get("id") if isinstance(d, dict) else getattr(d, "id", None), name)
                best_is_monitor = True
    if best is not None:
        return best
    return None, None


def _make_capture_generator(
    *,
    sync_q,
    channels: int,
    last_pcm_at_ref: list,
):
    """miniaudio capture 喂数据的 generator coroutine. 跑在 miniaudio 的 OS audio 线程内.

    last_pcm_at_ref 是单元素 list, 让 supervisor 能看到最近 PCM 到达时间 (不上锁,
    单写多读, time.time 写 float 在 CPython 是原子的).
    """
    def _gen():
        while True:
            data = yield
            try:
                samples = np.frombuffer(data, dtype=np.int16)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
                last_pcm_at_ref[0] = time.time()
                try:
                    sync_q.put_nowait(samples)
                except janus.SyncQueueFull:
                    # 队列满 (backend 跟不上 / asr_supervisor 暂停). 丢最旧.
                    # janus 没有 deque 语义, 手动 get 一帧再 put.
                    try:
                        sync_q.get_nowait()
                        sync_q.put_nowait(samples)
                    except Exception:
                        pass
            except Exception:
                logger.exception("listener capture callback failed")
    return _gen()


# ── 内部: asr supervisor ──────────────────────────────────────────────────

async def _asr_supervisor(pcm_queue: janus.Queue, cfg: _ListenerConfig) -> None:
    """循环: 等 capture ok → 开 VolcengineASR → 跑 recognize → 失败 backoff → 重开.

    VolcengineASR.recognize() 是 per-utterance — 一次调用消费一段音频直到服务端 is_final.
    我们在外层无限循环, 一句话一句话地开 recognize, 中间 janus 缓冲不丢.
    """
    from ghoshell_moss.host.listener.volcengine_asr import VolcengineASR, VolcengineASRConfig

    asr_cfg = VolcengineASRConfig(
        sample_rate=cfg.asr_sample_rate,
        end_window_size=cfg.end_window_ms,
    )
    asr = VolcengineASR(config=asr_cfg, logger=logger)
    backoff = _WS_RETRY_MIN

    try:
        was_paused = False
        while not _stop_event.is_set():
            # 等 capture 出 ok 才开 ws — 设备没就绪开 ws 也是白开
            # 检查 _paused — 用户显式暂停时不开新 session, 但设备保持连接
            if _paused or not _is_capture_ready():
                if _paused:
                    was_paused = True
                try:
                    await asyncio.wait_for(_stop_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                continue

            # 从 paused 转出 → flush 队列, 避免拿到 pause 期间堆积的老 PCM
            if was_paused:
                _flush_sync_queue(pcm_queue.sync_q)
                was_paused = False

            session = _Session()
            with _state_lock:
                global _current_session
                _current_session = session

            try:
                audio_gen = _pull_pcm(
                    async_q=pcm_queue.async_q,
                    session=session,
                    capture_sr=_health.sample_rate_capture or cfg.sample_rate,
                    asr_sr=cfg.asr_sample_rate,
                )
                async for result in asr.recognize(audio_gen):
                    if session.abort.is_set():
                        # force drain 已经把 partial 拿走了, 服务端送的 final 也要丢
                        continue
                    _handle_asr_result(result, session.utterance_id)
                # recognize 正常 end (is_final), 重置 backoff
                backoff = _WS_RETRY_MIN
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("listener: asr.recognize failed: %s", e)
                _transition_status(
                    "ws_error",
                    ws_error_count=_health.ws_error_count + 1,
                    last_error_msg=f"ws: {e}",
                )
                try:
                    await asyncio.wait_for(_stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, _WS_RETRY_MAX)
                continue

            # session 正常结束, 恢复 ok (如果不是因为 capture 出问题)
            if _health.status == "ws_error":
                _transition_status("ok")
    finally:
        try:
            await asr.close()
        except Exception:
            logger.exception("asr.close failed (ignored)")


def _is_capture_ready() -> bool:
    with _state_lock:
        return _health.status == "ok"


def _flush_sync_queue(sync_q) -> None:
    """Drain janus.sync_q non-blocking. resume 转 active 前调用, 避免拿老音频起头."""
    dropped = 0
    while True:
        try:
            sync_q.get_nowait()
            dropped += 1
        except Exception:
            break
    if dropped:
        logger.debug("listener: flushed %d stale PCM frames on resume.", dropped)


async def _pull_pcm(
    *,
    async_q: janus._AsyncQueueProxy,
    session: _Session,
    capture_sr: int,
    asr_sr: int,
) -> AsyncIterable[np.ndarray]:
    """一次 recognize 的音频源. 从 janus 拉 PCM, 必要时重采样, 直到 session.abort.

    abort 触发 → 生成器优雅退出 → recognize 发尾包 → 干净结束.
    """
    while not session.abort.is_set() and not _stop_event.is_set():
        try:
            samples = await asyncio.wait_for(async_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise

        if capture_sr != asr_sr:
            samples = _resample_int16(samples, capture_sr, asr_sr)
        yield samples


def _resample_int16(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样 int16 PCM. scipy.signal.resample_poly, 用 gcd 算 up/down 比例."""
    # scipy 是 host.voice.capture 已用的依赖, 不新增. 延迟 import.
    from scipy import signal as _sig

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    f32 = samples.astype(np.float32)
    out = _sig.resample_poly(f32, up, down)
    return out.astype(np.int16)


# ── 内部: ASR 结果处理 ────────────────────────────────────────────────────

def _handle_asr_result(result, utterance_id: str) -> None:
    """处理一次 ASRResult — partial 覆盖 _partial, final append _finalized_dq.

    触发 listeners 在 lock 外, snapshot 防 register/unregister 并发.
    跑在 backend asyncio 线程.
    """
    global _partial, _forgotten_since_last_drain

    text = result.text or ""
    if not text and not result.is_final:
        # 空 partial — 服务端 keepalive, 不更新 _partial 也不触发 listener
        return

    now = time.time()
    utterance = Utterance(
        id=utterance_id,
        text=text,
        is_final=bool(result.is_final),
        received_at=now,
    )

    triggered_sentence: Optional[Utterance] = None
    triggered_partial: Optional[Utterance] = None

    with _state_lock:
        if utterance.is_final:
            if text:  # 空 final 不入 buffer
                was_full = len(_finalized_dq) == _finalized_dq.maxlen
                _finalized_dq.append(utterance)
                if was_full:
                    _forgotten_since_last_drain += 1
                triggered_sentence = utterance
            _partial = None
        else:
            _partial = utterance
            triggered_partial = utterance

        # 更新 health 视图
        global _health
        _health = _health.model_copy(update={"last_voiced_at": now})
        _refresh_pending_partial_locked()
        _refresh_utterances_pending_drain_locked()

    # 触发 callbacks (lock 外)
    if triggered_sentence is not None:
        _fire_listeners(_sentence_listeners, triggered_sentence)
    if triggered_partial is not None:
        _fire_listeners(_partial_listeners, triggered_partial)


def _fire_listeners(registry: dict[str, Callable], payload) -> None:
    with _listeners_lock:
        callbacks = list(registry.values())
    for cb in callbacks:
        try:
            cb(payload)
        except Exception:
            logger.exception("listener callback raised (isolated, other cbs continue)")


# ── 无状态 helper (channel 层用) ─────────────────────────────────────────
# Runtime 不直接产 Message. channel 拿现成 helper 包装. 跟 asr.py 同范式.

def to_xml_text(u: Utterance) -> str:
    """单条 utterance → 紧凑 XML. 多条拼接用 batch_to_xml."""
    attrs = [f'id="{u.id}"', f'ts="{u.received_at:.3f}"']
    if not u.is_final:
        attrs.append('partial="true"')
    if u.forced:
        attrs.append('forced="true"')
    return f'<{u.source} {" ".join(attrs)}>{u.text}</{u.source}>'


def batch_to_xml(batch: UtteranceBatch) -> str:
    """整批 → 单段 XML, 含 forgotten 元信息. forgotten>0 = 模型知道上下文有 gap."""
    lines = [f'<g1.listener forgotten="{batch.forgotten}" count="{len(batch.items)}">']
    for u in batch.items:
        lines.append("  " + to_xml_text(u))
    lines.append("</g1.listener>")
    return "\n".join(lines)


def to_message(u: Utterance) -> Message:
    """单条 utterance → ghoshell_moss Message. channel 入 context_messages."""
    attrs: dict = {"id": u.id}
    if not u.is_final:
        attrs["partial"] = True
    if u.forced:
        attrs["forced"] = True
    return Message.new(
        tag=u.source,
        attributes=attrs,
        timestamp=True,
    ).with_content(u.text)


def batch_to_message(batch: UtteranceBatch) -> Message:
    """整批 → 单条 Message. forgotten 进 attributes."""
    msg = Message.new(
        tag="g1.listener",
        attributes={"forgotten": batch.forgotten, "count": len(batch.items)},
        timestamp=True,
    )
    for u in batch.items:
        msg = msg.with_content(to_xml_text(u))
    return msg


def health_to_message(h: ListenerHealth) -> Message:
    """health snapshot → Message. channel 把它放进 system context, 模型可据此自述听感.

    只导出对 LLM 有意义的字段, 不导出累计计数等技术指标 (那些在 health() / log 里).
    """
    attrs: dict = {"status": h.status}
    if h.paused:
        attrs["paused"] = True
    if h.device_name:
        attrs["device"] = h.device_name
    if h.sample_rate_capture:
        attrs["sample_rate"] = h.sample_rate_capture
    if h.pending_partial:
        attrs["partial"] = h.pending_partial
    if h.utterances_pending_drain:
        attrs["pending"] = h.utterances_pending_drain
    if h.last_voiced_at:
        attrs["last_voiced_ago_sec"] = round(time.time() - h.last_voiced_at, 1)
    return Message.new(
        tag="g1.listener.health",
        attributes=attrs,
        timestamp=True,
    )
