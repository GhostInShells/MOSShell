"""
ASR runtime — G1 内置语音识别服务的运行时模块.

本模块是 runtime 层的首个样例. 设计纪律见同目录 README.md.
后续 arms keyframe / locomotion / 各类轨迹模块照此范式实现.

物理事实 (来自 docs/index.md + scripts/sdk/23_asr_api_probe.py 实测):
  - 启动协议: AudioClient._Call(1002, json.dumps({"action":"start"}))
  - 停止协议: AudioClient._Call(1002, json.dumps({"action":"stop"}))
  - 数据流:   DDS topic `rt/audio_msg` (std_msgs/String_), 内容是 JSON 字符串
  - 已知字段: text / index / speaker_id / angle (°) / confidence / emotion / language
  - G1 默认非流式模式, 无 is_final; play_state 消息被过滤.
  - G1 报文本身无 id 字段, runtime 自生成 ulid
  - 实机验证 (2026-07-01): angle 始终 0, speaker_id 始终 0 — G1 当前固件不启用声源定位与说话人识别.
  - is_final 始终 false — 默认非流式模式. 但 G1 VAD 结束时 LED 变色, 需另寻判停路径.
  - emotion 字段名实机为 emotion (官方文档写 sense), 值 e.g. '<|NEUTRAL|>'.
  - play_state 消息 ({"play_state": 0/1}) 与 ASR 结果共用同一 topic, _parse 过滤.

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import asr

    bootstrap(nic="eth0")        # SDK 生命周期由 sdk/ 负责
    asr.start()                  # 启动 ASR 服务 + reader 线程
    # ...
    batch = asr.drain()          # 拿走累积 AsrResult, 含 forgotten 计数
    asr.stop()
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ghoshell_moss.message import Message, unique_id

from ghoshell_moss_contrib.unitree.g1.sdk import get_audio_client

logger = logging.getLogger("moss.g1.runtime.asr")


# ── 数据契约 ──────────────────────────────────────────────────────────────
# 字段 description 直接进 LLM prompt (channel 层用 model_json_schema 时).
# 措辞按"对 LLM 解释精度"写, 不只是给开发者看.

class AsrResult(BaseModel):
    """G1 一次 ASR 识别结果. 字段对齐官方文档 + 实机验证 (2026-07-01)."""

    id: str = Field(
        default_factory=unique_id,
        description="本条记录的 ulid. 由 runtime 生成.",
    )
    text: str = Field(
        default="",
        description="识别到的文本内容.",
    )
    index: Optional[int] = Field(
        default=None,
        description="消息唯一序号 (G1 原始字段).",
    )
    speaker_id: Optional[int] = Field(
        default=None,
        description="说话人编号. G1 默认不启用, 通常为 0.",
    )
    angle: Optional[int] = Field(
        default=None,
        description="声源方位角 0-180°. G1 默认不启用, 通常为 0.",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="识别置信度 [0,1].",
    )
    emotion: Optional[str] = Field(
        default=None,
        description="情绪识别结果 (e.g. '<|NEUTRAL|>'). 实机字段名 emotion, 官方文档写 sense.",
    )
    language: Optional[str] = Field(
        default=None,
        description="语言种类 (e.g. '<|zh|>').",
    )
    received_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S"),
        description="reader 线程收到本条的本地时间 (HH:MM:SS).",
    )
    source: str = Field(
        default="g1.asr",
        description="数据来源固定常量.",
    )
    raw_json: Optional[str] = Field(
        default=None,
        description="DDS 原始 JSON. 仅供 debug, 不导出给模型.",
    )


class AsrBatch(BaseModel):
    """drain 一次的返回. items 按到达时间升序, forgotten 是本批累积期间被 ring buffer 挤掉的数量."""

    items: list[AsrResult] = Field(default_factory=list)
    forgotten: int = Field(
        default=0,
        description="自上次 drain 起被 ring buffer 挤掉的条数. 类似人类'我没听清几句', "
                    "应当告知模型让它知道上下文有 gap.",
    )


# ── 模块级私有状态 ────────────────────────────────────────────────────────
# Runtime 模块 = 单例服务. 所有可变状态在模块级, 由 _state_lock 保护.
# listeners 单独 _listeners_lock 防止 callback 触发期间 register/unregister 死锁.

_state_lock = threading.Lock()
_listeners_lock = threading.Lock()

_dq: deque[AsrResult] = deque(maxlen=32)
_listeners: dict[str, Callable[[AsrResult], None]] = {}

_thread: Optional[threading.Thread] = None
_subscriber: object = None  # cyclonedds ChannelSubscriber, 延迟构造
_running: bool = False
_stop_event: Optional[threading.Event] = None

_last_recv: float = 0.0
_error_count: int = 0
_forgotten_since_last_drain: int = 0
_total_count: int = 0  # 累计入队条数, start 后递增, 不随 drain 归零


# ── 公开接口 ─────────────────────────────────────────────────────────────

def start(*, buffer_size: int = 32) -> None:
    """启动 ASR 服务. 幂等 — 已运行则直接 return.

    前置: sdk.bootstrap() 已完成 (否则 get_audio_client() raise).

    步骤:
      1. _Call(1002, {"action":"start"}) — 失败仅 warn, 不阻塞.
         实测发现 _Call 可能失败 (G1 ASR 服务侧未知协议), 但 DDS topic 仍有流.
      2. ChannelSubscriber("rt/audio_msg", String_) Init.
         订阅失败必须 raise — 没有 subscriber 等于没有数据流, 模块无意义.
      3. 起 daemon reader 线程, 进入 poll 循环.

    :param buffer_size: ring buffer 容量. 满则挤掉最旧, forgotten 计数 +1.
    """
    global _dq, _thread, _subscriber, _running, _stop_event
    global _last_recv, _error_count, _forgotten_since_last_drain

    with _state_lock:
        if _running:
            logger.debug("start() 重入 — 已在运行, 跳过.")
            return

        # 1. 启动 ASR 服务 (fire-and-forget, 不阻塞).
        # 实测 _Call(1002, "start") 全部返回 3104, DDS 流不依赖此 RPC.
        # 卸到 daemon 线程避免阻塞 startup 路径 (RPC 超时几秒).
        def _fire_asr_rpc() -> None:
            try:
                code, data = get_audio_client()._Call(1002, json.dumps({"action": "start"}))
                if code != 0:
                    logger.debug("ASR _Call(1002, start) code=%s (expected 3104)", code)
            except Exception as e:
                logger.debug("ASR _Call(1002, start) 抛异常 (expected): %s", e)
        threading.Thread(target=_fire_asr_rpc, name="g1-asr-rpc", daemon=True).start()

        # 2. 订阅 DDS topic. 必须成功.
        # TODO 实测: cyclonedds ChannelSubscriber Close 后能否重新 Init.
        #   如不能, stop() 改为不关 sub 只停 reader 线程.
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        try:
            sub = ChannelSubscriber("rt/audio_msg", String_)
            sub.Init()
            _subscriber = sub
        except Exception:
            logger.exception("订阅 rt/audio_msg 失败")
            raise

        # 3. 重置状态 + 起 reader 线程.
        _dq = deque(maxlen=buffer_size)
        _last_recv = 0.0
        _error_count = 0
        _forgotten_since_last_drain = 0
        _total_count = 0
        _stop_event = threading.Event()
        _running = True

        _thread = threading.Thread(
            target=_reader_loop,
            name="g1-asr-reader",
            daemon=True,
        )
        _thread.start()
        logger.info("ASR runtime started (buffer_size=%d)", buffer_size)


def stop(timeout: float = 2.0) -> None:
    """停止 ASR 服务并 join reader 线程. 幂等.

    timeout 内未 join 完成 → log warning 但 return (daemon 线程随进程退出而死).

    :param timeout: 等待 reader 线程退出的秒数.
    """
    global _thread, _subscriber, _running, _stop_event

    # 拿到 thread / sub 引用后立即出 lock — 否则 join 跟 reader 线程内
    # 的 _state_lock 互相等待会死锁 (reader 在 _enqueue 内拿 lock).
    with _state_lock:
        if not _running:
            logger.debug("stop() 重入 — 未在运行, 跳过.")
            return
        _running = False
        if _stop_event is not None:
            _stop_event.set()
        thread = _thread
        sub = _subscriber

    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "ASR reader 线程未在 %.1fs 内 join 完成 (daemon, 随进程退出).",
                timeout,
            )

    if sub is not None:
        try:
            sub.Close()
        except Exception:
            logger.exception("DDS subscriber Close 失败 (忽略, 进程退出会回收)")

    try:
        client = get_audio_client()
        client._Call(1002, json.dumps({"action": "stop"}))
    except Exception as e:
        logger.warning("ASR _Call(1002, stop) 抛异常 %s (忽略)", e)

    with _state_lock:
        _thread = None
        _subscriber = None
        _stop_event = None

    logger.info("ASR runtime stopped.")


def is_running() -> bool:
    """当前是否运行."""
    with _state_lock:
        return _running


def drain() -> AsrBatch:
    """拿走 buffer 内全部 AsrResult, 同时取走 forgotten 计数并归零.

    返回 AsrBatch — items 按到达时间升序, forgotten 是本次 drain 之间被 ring buffer
    挤掉的条数. 线程安全, 与 reader 线程的 _enqueue 串行 (共用 _state_lock).
    """
    global _forgotten_since_last_drain

    with _state_lock:
        items = list(_dq)
        _dq.clear()
        forgotten = _forgotten_since_last_drain
        _forgotten_since_last_drain = 0
    return AsrBatch(items=items, forgotten=forgotten)


def peek_latest() -> Optional[AsrResult]:
    """看 buffer 末尾一条, 不出栈, 不影响 forgotten 计数. 无数据返回 None."""
    with _state_lock:
        if not _dq:
            return None
        return _dq[-1]


def register_listener(cb: Callable[[AsrResult], None]) -> str:
    """注册新数据回调. cb 在 reader 线程内同步触发, 不能阻塞.

    跨线程需求 (推回 asyncio loop / queue) 由 cb 自行处理. 跟 sdk/_buttons.py
    范式一致 — runtime 不替 listener 决定线程模型, 把控制权留给注册方.

    :param cb: 接收 AsrResult 的 callable. 异常会被隔离, 不影响其他 listener.
    :return: handle (ulid str), 用于 unregister.
    """
    handle = unique_id()
    with _listeners_lock:
        _listeners[handle] = cb
    return handle


def unregister_listener(handle: str) -> None:
    """反注册 listener. 未知 handle 静默忽略, 允许调用方安全 cleanup."""
    with _listeners_lock:
        _listeners.pop(handle, None)


def health() -> dict:
    """暴露 runtime 内部状态. 供 monitor 脚本 / channel debug / 实机校验用."""
    with _state_lock:
        return {
            "running": _running,
            "last_recv": _last_recv,
            "buffer_len": len(_dq),
            "buffer_max": _dq.maxlen,
            "total_count": _total_count,
            "forgotten_since_last_drain": _forgotten_since_last_drain,
            "error_count": _error_count,
        }


# ── 内部: reader 线程 ────────────────────────────────────────────────────

def _reader_loop() -> None:
    """reader 线程主循环.

    异常处理纪律 (Runtime README §4):
      - 任何异常 log.exception + 短 sleep + 继续, 绝不 raise 出循环.
      - 信号断流比异常传播严重得多.
    """
    global _error_count

    logger.info("ASR reader loop entered.")
    sub = _subscriber  # 局部引用避免每次属性查找
    stop_event = _stop_event
    while not stop_event.is_set():
        try:
            msg = sub.Read(timeout=500)  # cyclonedds: ms
            if msg is None:
                # 无匹配 publisher 时 Read 可能立即返回. 短暂 sleep 防 CPU 空转.
                time.sleep(0.05)
                continue
            raw = getattr(msg, "data", None)
            if not raw:
                continue
            result = _parse(raw)
            if result is None:
                continue
            _enqueue(result)
        except Exception:
            _error_count += 1
            logger.exception("ASR reader 异常 (累计 %d, 保持线程活).", _error_count)
            time.sleep(0.1)
    logger.info("ASR reader loop exited.")


def _parse(raw: str) -> Optional[AsrResult]:
    """把 DDS String_ 报文 (JSON 字符串) 解析为 AsrResult.

    过滤: play_state 消息 (不含 text 字段)、空文本、keepalive 包.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("ASR 报文非 JSON, 跳过: %r", raw[:200])
        return None

    # 过滤 ASR 播放状态消息 (不含 text 字段).
    if "text" not in data:
        return None

    text = data.get("text") or data.get("transcript") or ""
    if not text:
        return None

    # emotion: 实机字段名 emotion, 官方文档写 sense — 两者都尝试.
    emotion = data.get("emotion") or data.get("sense")

    return AsrResult(
        text=text,
        index=data.get("index"),
        speaker_id=data.get("speaker_id"),
        angle=data.get("angle"),
        confidence=data.get("confidence"),
        emotion=emotion,
        language=data.get("language"),
        raw_json=raw,
    )


def _enqueue(result: AsrResult) -> None:
    """放入 buffer + 触发 listeners. 在 reader 线程内调用."""
    global _last_recv, _forgotten_since_last_drain, _total_count

    with _state_lock:
        was_full = len(_dq) == _dq.maxlen
        _dq.append(result)  # deque(maxlen=N): 满则自动挤掉最旧
        if was_full:
            _forgotten_since_last_drain += 1
        _last_recv = time.time()
        _total_count += 1

    # listener 触发在 lock 外 — 避免 callback 期间持锁阻塞 drain/peek.
    # snapshot 防 register/unregister 并发期间的 "dict changed during iteration".
    with _listeners_lock:
        snapshot = list(_listeners.values())
    for cb in snapshot:
        try:
            cb(result)
        except Exception:
            logger.exception("ASR listener 回调异常 (隔离, 不影响其他 listener).")


def peek_window(max_count: int = 3) -> list[AsrResult]:
    """不出栈地返回 buffer 内最新的 max_count 条. 供 context_messages 每帧读取.

    不清空 buffer, 不影响 drain 的 forgotten 计数.
    """
    with _state_lock:
        items = list(_dq)
    return items[-max_count:] if len(items) > max_count else items


def peek_total_count() -> int:
    """返回累计入队条数 (start 后递增, 不随 drain 归零). 供 channel 层构建 context."""
    with _state_lock:
        return _total_count


# ── 无状态 helper (channel 层用) ─────────────────────────────────────────
# Runtime 不直接产 Message — Message 是 channel 层 "何时/怎么/要不要喂模型"
# 的事. 这里只提供无状态转换器, channel 按需调.

def to_xml_text(r: AsrResult) -> str:
    """单条结果 → 紧凑 XML. 多条拼接用 results_to_message."""
    attrs = [f'id="{r.id}"', f'ts="{r.received_at}"']
    if r.speaker_id is not None and r.speaker_id != 0:
        attrs.append(f'speaker="{r.speaker_id}"')
    if r.angle is not None and r.angle != 0:
        attrs.append(f'angle="{r.angle}"')
    if r.confidence is not None:
        attrs.append(f'conf="{r.confidence:.2f}"')
    if r.emotion:
        attrs.append(f'emotion="{r.emotion}"')
    return f'<{r.source} {" ".join(attrs)}>{r.text}</{r.source}>'


def batch_to_xml(batch: AsrBatch) -> str:
    """整批 → 单段 XML, 含 forgotten 元信息. 模型看到 forgotten>0 即知上下文有 gap."""
    lines = [f'<g1.asr forgotten="{batch.forgotten}" count="{len(batch.items)}">']
    for r in batch.items:
        lines.append("  " + to_xml_text(r))
    lines.append("</g1.asr>")
    return "\n".join(lines)


def to_message(r: AsrResult) -> Message:
    """单条结果 → ghoshell_moss Message. channel 把它入 context_messages.

    只带对模型有用的字段: 时间戳 + 非零可选字段. id 是 ulid 字符串, 对 LLM 无用,
    不放 attributes. source 由 tag 承载, 不重复.
    """
    attrs: dict = {"ts": r.received_at}
    if r.speaker_id is not None and r.speaker_id != 0:
        attrs["speaker"] = r.speaker_id
    if r.angle is not None and r.angle != 0:
        attrs["angle"] = r.angle
    if r.confidence is not None:
        attrs["conf"] = round(r.confidence, 2)
    if r.emotion:
        attrs["emotion"] = r.emotion
    msg = Message.new(
        tag=r.source,
        attributes=attrs,
        timestamp=True,
    ).with_content(r.text)
    # 用 ASR 结果自己的到达时间, 而非 Message 构造时间.
    # received_at 是 "HH:MM:SS" 字符串, 补上今日日期 + 本地时区.
    if r.received_at:
        try:
            h, m, s = map(int, r.received_at.split(":"))
            msg.meta.created = datetime.now().astimezone().replace(
                hour=h, minute=m, second=s, microsecond=0,
            )
        except (ValueError, TypeError):
            pass
    return msg


def results_to_message(items: list[AsrResult], total: int = 0) -> Message:
    """最近几条结果 + 累计总数 → 单条 Message, 供 context_messages 用.

    total 是 peek_total_count() 的值, 让模型知道 "共收到 N 条, 现在看到的是最后 M 条".
    """
    msg = Message.new(
        tag="g1.asr-batch",
        attributes={"total": total, "showing": len(items)},
        timestamp=True,
    )
    for r in items:
        msg = msg.with_content(to_xml_text(r))
    return msg


def batch_to_message(batch: AsrBatch) -> Message:
    """整批 → 单条 Message, forgotten 进 attributes. channel pop 后入 context."""
    msg = Message.new(
        tag="g1.asr-batch",
        attributes={"forgotten": batch.forgotten, "count": len(batch.items)},
        timestamp=True,
    )
    for r in batch.items:
        msg = msg.with_content(to_xml_text(r))
    return msg
