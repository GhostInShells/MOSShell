"""
control_pad — G1 遥控器按键的语义层 runtime 模块.

站在 sdk._buttons 单键边沿之上, 提供:
  - KeyBinding 注册: 物理键组合 (frozenset) → 命名语义事件 + per-binding debounce
  - fallthrough listener: 任何按下边沿但未命中 binding 的全局兜底 (全局 debounce)
  - event listener: 任何事件 (binding 命中 OR fallthrough) 通知, 给 FSM 推 context_messages
  - 事件历史 ring buffer: drain/peek, 准 asr 范式, 可灌入 context_messages

设计纪律: 见同目录 README.md.

它**不做**:
  - 不内置组合键语义 (ESTOP / GHOST_TRIGGER 等). 那是 FSM 任务的事 (story-2026-07
    §2 提到的"按键状态机由人类工程师 7-01 起牵头重新设计", 不在本模块表达).
  - 不知道哪些组合 "合法" / "非法". fallthrough listener 让 FSM 自己判.
  - 不替 FSM 决策权限归零. 控制器物理失效保护 (黏键反复触发) 是 FSM 策略层 —
    本模块 health() 暴露触发率, FSM 自己监测.

判定语义:
  - 触发只在按下边沿 (pressed=True), 不在松开.
  - pressed_keys = state.remote() 当前所有按下的键的 frozenset.
  - **exact match**: binding.keys == pressed_keys 命中. 不用 subset —
    避免 {l2} binding 被 l2+b 组合意外触发.
  - 顺序自然容错: 最后一个按下的键的边沿触发命中 ({l2}, 再 b → b 边沿
    pressed={l2,b} 命中 {l2,b} binding).
  - 没命中任何 binding 的按下边沿 → fallthrough 兜底 (受全局 debounce 节流).

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import control_pad

    bootstrap(nic)
    control_pad.start()
    h = control_pad.register_binding(
        name="my_estop",
        keys={"l2", "b"},
        callback=lambda evt: print(f"ESTOP! {evt.pressed_keys}"),
        debounce_sec=0.05,
    )
    # ...
    batch = control_pad.drain()  # 事件历史
    control_pad.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ghoshell_moss.message import Message, unique_id

from ghoshell_moss_contrib.unitree.g1.sdk import (
    CallbackHandle,
    register_button_callback,
    remote as _get_remote_state,
    unregister_button_callback,
)
from ghoshell_moss_contrib.unitree.g1.sdk._buttons import VALID_BUTTONS

logger = logging.getLogger("moss.g1.runtime.control_pad")


# ── 数据契约 ──────────────────────────────────────────────────────────────
# Field description 直接进 LLM prompt (channel 层用 model_json_schema 时).
# 措辞按"对 LLM 解释精度"写.

class KeyBinding(BaseModel):
    """一个物理键组合的语义封装. 由 register_binding 创建, 不直接构造."""

    name: str = Field(
        description="注册方起的语义名 (FSM 决定语义). 触发事件时回显, 进 ring buffer.",
    )
    keys: frozenset[str] = Field(
        description="物理键集合 (小写名). 按下边沿瞬间当前所有按下的键 == 本集合时触发. "
                    "exact match — {'l2'} 不会被 l2+b 组合意外触发.",
    )
    debounce_sec: float = Field(
        description="本 binding 触发后, 该秒数内不再触发 (monotonic 时间). "
                    "实际值 = max(注册时传入, 模块 global_min_debounce_sec).",
        ge=0.0,
    )


class KeyEvent(BaseModel):
    """一次按键事件 (binding 命中或 fallthrough). 进 ring buffer, 模型感知用."""

    id: str = Field(
        default_factory=unique_id,
        description="本条事件的 ulid. 由 runtime 生成 — 用于去重/追溯.",
    )
    binding_name: str = Field(
        description="触发的 binding 名称. fallthrough 事件填空串 ''.",
    )
    pressed_keys: tuple[str, ...] = Field(
        description="触发瞬间所有按下的物理键, 按字母升序排列, 便于解析.",
    )
    is_fallthrough: bool = Field(
        default=False,
        description="True = 未匹配任何 binding 的按下边沿. False = 命中某 binding.",
    )
    triggered_at: float = Field(
        default_factory=time.time,
        description="触发时刻 (time.time() 秒, wall clock).",
    )


class KeyEventBatch(BaseModel):
    """drain 一次的返回. items 按到达时间升序, forgotten 是本批被 ring buffer 挤掉的数量."""

    items: list[KeyEvent] = Field(default_factory=list)
    forgotten: int = Field(
        default=0,
        description="自上次 drain 起被 ring buffer 挤掉的条数. 模型看到 >0 即知有 gap.",
    )


# ── 模块级私有状态 ────────────────────────────────────────────────────────
# Runtime 模块 = 单例服务. 状态在模块级, 两把锁保护.

_state_lock = threading.Lock()
_listeners_lock = threading.Lock()

# binding handle (ulid) -> KeyBinding (数据) + callback (实现)
# 分开存: KeyBinding 是 pydantic 数据契约, 不应包含 callable.
_bindings: dict[str, KeyBinding] = {}
_binding_callbacks: dict[str, Callable[[KeyEvent], None]] = {}
# binding handle -> 最近一次触发的 monotonic 时间. per-binding debounce 记账.
_last_triggered: dict[str, float] = {}

# fallthrough listeners: handle (ulid) -> cb(pressed_keys, event)
_fallthrough_listeners: dict[str, Callable[[tuple[str, ...], "KeyEvent"], None]] = {}
# 全局 fallthrough 最后触发时刻 — 单一时间戳, 所有 fallthrough 共享 debounce.
# 防颤抖键反复触发权限归零 (story-2026-07: F3 归零是大动作, 不能高频).
_last_fallthrough_triggered: float = 0.0

# event listeners: handle (ulid) -> cb(event)
# 任何事件 (binding 命中 OR fallthrough) 都通知一次. FSM 用来灌 context_messages.
_event_listeners: dict[str, Callable[["KeyEvent"], None]] = {}

# Ring buffer for events. README §5 — 满则挤旧, forgotten 计数.
_dq: deque[KeyEvent] = deque(maxlen=64)
_forgotten_since_last_drain: int = 0

# SDK callback handles (start 时给 16 键各注册一个, stop 时反注册).
_sdk_handles: list[CallbackHandle] = []

# 生命周期 + 配置
_running: bool = False
_default_debounce_sec: float = 0.20
_global_min_debounce_sec: float = 0.05
_fallthrough_debounce_sec: float = 1.0

# health 统计 (累积, 在 start 时重置)
_press_edge_count: int = 0           # 按下边沿到达 _on_sdk_edge 的总次数
_fired_count: int = 0                # binding 命中并经 debounce 通过的次数
_fallthrough_fired_count: int = 0    # fallthrough 经 debounce 通过的次数
_debounce_suppressed_count: int = 0  # 被 debounce 静默的次数 (binding + fallthrough 合计)


# ── 公开接口 ─────────────────────────────────────────────────────────────

def start(
    *,
    buffer_size: int = 64,
    default_debounce_sec: float = 0.20,
    global_min_debounce_sec: float = 0.05,
    fallthrough_debounce_sec: float = 1.0,
) -> None:
    """启动 control_pad. 幂等 — 已运行则直接 return.

    前置: sdk.bootstrap() 已完成 (否则 register_button_callback 会 raise).

    步骤:
      1. 重置模块状态 (清空 bindings / ring buffer / debounce 时间戳 / health 计数).
      2. 给 sdk._buttons 16 键各注册一个 callback. 边沿到达时通过 _on_sdk_edge 分发.

    :param buffer_size: ring buffer 容量. 满则挤旧, forgotten 计数 +1.
    :param default_debounce_sec: register_binding 时 debounce_sec=None 用本值.
    :param global_min_debounce_sec: floor — register_binding 取 max(provided, floor).
        防止注册方填 0 漏防抖.
    :param fallthrough_debounce_sec: 全局 fallthrough cooldown. 比 binding 长一个量级
        因为 fallthrough = 权限归零, 大动作不能高频.
    """
    global _running, _dq, _default_debounce_sec, _global_min_debounce_sec
    global _fallthrough_debounce_sec, _last_fallthrough_triggered
    global _press_edge_count, _fired_count, _fallthrough_fired_count
    global _debounce_suppressed_count, _forgotten_since_last_drain

    with _state_lock:
        if _running:
            logger.debug("start() 重入 — 已在运行, 跳过.")
            return

        # 1. 重置状态
        _dq = deque(maxlen=buffer_size)
        _bindings.clear()
        _binding_callbacks.clear()
        _last_triggered.clear()
        _last_fallthrough_triggered = 0.0
        _press_edge_count = 0
        _fired_count = 0
        _fallthrough_fired_count = 0
        _debounce_suppressed_count = 0
        _forgotten_since_last_drain = 0
        _default_debounce_sec = default_debounce_sec
        _global_min_debounce_sec = global_min_debounce_sec
        _fallthrough_debounce_sec = fallthrough_debounce_sec

        # 2. 给 16 键各注册 sdk callback.
        # closure 默认参数 _btn=btn 防 late binding (Python 经典坑).
        # 中途 raise 时回滚已注册的 handle — 现实中不会触发 (is_started 是一次性
        # 状态, 16 次注册要么全过要么全 fail), 但防御性写好, README §4.
        _sdk_handles.clear()
        try:
            for btn in VALID_BUTTONS:
                h = register_button_callback(
                    btn,
                    lambda pressed, _btn=btn: _on_sdk_edge(_btn, pressed),
                )
                _sdk_handles.append(h)
        except Exception:
            for partial in _sdk_handles:
                try:
                    unregister_button_callback(partial)
                except Exception:
                    pass
            _sdk_handles.clear()
            raise

        _running = True

    logger.info(
        "control_pad started (buffer_size=%d, default_debounce=%.3f, "
        "min=%.3f, fallthrough=%.3f)",
        buffer_size, default_debounce_sec, global_min_debounce_sec,
        fallthrough_debounce_sec,
    )


def stop(timeout: float = 2.0) -> None:
    """停止 control_pad. 反注册全部 sdk callback. 幂等.

    timeout 当前未用 (本模块不起自有线程, sdk callback 跑在 reader 线程,
    反注册即停). 保留参数对齐 README §3 标准签名.
    """
    global _running

    with _state_lock:
        if not _running:
            logger.debug("stop() 重入 — 未在运行, 跳过.")
            return
        _running = False
        # snapshot handles, 释放 lock 后反注册 — 否则 reader 线程内的 _on_sdk_edge
        # 持 _state_lock 进 _dispatch_press 时, 跟我们 unregister 互等死锁.
        handles_snapshot = list(_sdk_handles)
        _sdk_handles.clear()

    for h in handles_snapshot:
        try:
            unregister_button_callback(h)
        except Exception:
            logger.exception("unregister_button_callback failed (忽略).")

    logger.info("control_pad stopped.")


def is_running() -> bool:
    """当前是否运行."""
    with _state_lock:
        return _running


def register_binding(
    name: str,
    keys: frozenset[str] | set[str] | list[str] | tuple[str, ...],
    callback: Callable[[KeyEvent], None],
    *,
    debounce_sec: Optional[float] = None,
) -> str:
    """注册一个 KeyBinding. 返回 handle (ulid) 供 unregister_binding.

    :param name: 语义名 (FSM 决定). 事件中回显, 模型可见.
    :param keys: 物理键集合 (任何 iterable). 元素必须在 VALID_BUTTONS, 非空.
    :param callback: 命中时调 cb(event: KeyEvent). **跑在 sdk reader 线程**, 不能阻塞.
                     异常被隔离 log, 不影响其它 binding/listener.
    :param debounce_sec: 本 binding 的 cooldown. None 时用模块 default_debounce_sec.
                         实际值 = max(传入值, global_min_debounce_sec) 兜底.
    :return: handle (ulid str), 用于 unregister.
    :raises ValueError: name 空 / keys 空 / keys 含非法按键名.
    """
    if not name:
        raise ValueError("binding name must not be empty")
    keys_fs = frozenset(keys)
    if not keys_fs:
        raise ValueError("binding keys must not be empty")
    invalid = keys_fs - VALID_BUTTONS
    if invalid:
        raise ValueError(
            f"invalid button names: {sorted(invalid)}. valid: {sorted(VALID_BUTTONS)}"
        )

    requested = debounce_sec if debounce_sec is not None else _default_debounce_sec
    actual_debounce = max(requested, _global_min_debounce_sec)

    binding = KeyBinding(name=name, keys=keys_fs, debounce_sec=actual_debounce)
    handle = unique_id()

    with _state_lock:
        # _default_debounce_sec / _global_min_debounce_sec 读取本应在 lock 内
        # 才严格无 race (start/_configure_for_testing 写它们), 但 float 读取
        # GIL 原子, register_binding 现实中只在 start 之后调, 不并发. 这里
        # 用 lock 保护写入即足够.
        _bindings[handle] = binding
        _binding_callbacks[handle] = callback
        _last_triggered[handle] = 0.0  # 初始 0, 第一次触发立即可通过 debounce

    logger.debug(
        "registered binding: handle=%s name=%s keys=%s debounce=%.3f",
        handle, name, sorted(keys_fs), actual_debounce,
    )
    return handle


def unregister_binding(handle: str) -> None:
    """反注册 binding. 未知 handle 静默忽略 (允许调用方安全 cleanup)."""
    with _state_lock:
        _bindings.pop(handle, None)
        _binding_callbacks.pop(handle, None)
        _last_triggered.pop(handle, None)


def register_fallthrough_listener(
    callback: Callable[[tuple[str, ...], KeyEvent], None],
) -> str:
    """注册 fallthrough listener — 任何按下边沿但未命中任何 binding 时触发.

    用途: FSM 用此实现"非法键组合 → 权限归零". control_pad 不知道白名单,
    fallthrough cb 自己判.

    全局 debounce: 所有 fallthrough 共享一个时间戳, 任意一次 fallthrough 触发后,
    fallthrough_debounce_sec 内的其它 fallthrough 全部静默. 防颤抖键反复归零.

    :param callback: cb(pressed_keys: tuple[str, ...], event: KeyEvent) -> None.
                     pressed_keys 已字母排序. event 是构造好的 fallthrough KeyEvent
                     (binding_name='', is_fallthrough=True), 跟 ring buffer 一致.
                     **跑在 sdk reader 线程**, 不能阻塞.
    :return: handle (ulid).
    """
    handle = unique_id()
    with _listeners_lock:
        _fallthrough_listeners[handle] = callback
    return handle


def unregister_fallthrough_listener(handle: str) -> None:
    """反注册 fallthrough listener."""
    with _listeners_lock:
        _fallthrough_listeners.pop(handle, None)


def register_event_listener(
    callback: Callable[[KeyEvent], None],
) -> str:
    """注册全局 event listener — 任何事件 (binding 命中 OR fallthrough) 都通知.

    用途: FSM 把所有事件灌入 context_messages, 让模型感知发生了什么.
    跟 binding callback 不同, 不需要为每个 binding 单独订阅.

    通知发生在事件已 debounce 通过并入 ring buffer 之后. 被 debounce 静默
    的边沿**不通知** event listener (调用方看到的事件流跟 ring buffer 一致).

    :param callback: cb(event: KeyEvent) -> None. **跑在 sdk reader 线程**, 不能阻塞.
    :return: handle (ulid).
    """
    handle = unique_id()
    with _listeners_lock:
        _event_listeners[handle] = callback
    return handle


def unregister_event_listener(handle: str) -> None:
    """反注册 event listener."""
    with _listeners_lock:
        _event_listeners.pop(handle, None)


def drain() -> KeyEventBatch:
    """拿走 ring buffer 全部事件 + forgotten 计数, 清空 + 归零.

    线程安全, 与 _dispatch_press 串行 (共用 _state_lock).
    """
    global _forgotten_since_last_drain
    with _state_lock:
        items = list(_dq)
        _dq.clear()
        forgotten = _forgotten_since_last_drain
        _forgotten_since_last_drain = 0
    return KeyEventBatch(items=items, forgotten=forgotten)


def peek_latest() -> Optional[KeyEvent]:
    """看 ring buffer 末尾一条, 不出栈, 不影响 forgotten 计数. 无数据返回 None."""
    with _state_lock:
        if not _dq:
            return None
        return _dq[-1]


def list_bindings() -> dict[str, KeyBinding]:
    """现场调试: handle -> KeyBinding. 不含 cb 与 listeners."""
    with _state_lock:
        return dict(_bindings)


def health() -> dict:
    """暴露 runtime 内部状态.

    fallthrough_fired_count 在短时间内快速增长 = 控制器可能黏键或失效,
    FSM 可据此进 safe mode (但 control_pad 自身不做决策).
    """
    with _state_lock:
        return {
            "running": _running,
            "bindings_count": len(_bindings),
            "fallthrough_listeners": len(_fallthrough_listeners),
            "event_listeners": len(_event_listeners),
            "buffer_len": len(_dq),
            "buffer_max": _dq.maxlen,
            "forgotten_since_last_drain": _forgotten_since_last_drain,
            "press_edge_count": _press_edge_count,
            "fired_count": _fired_count,
            "fallthrough_fired_count": _fallthrough_fired_count,
            "debounce_suppressed_count": _debounce_suppressed_count,
            "default_debounce_sec": _default_debounce_sec,
            "global_min_debounce_sec": _global_min_debounce_sec,
            "fallthrough_debounce_sec": _fallthrough_debounce_sec,
        }


# ── 内部: sdk callback → _dispatch_press 分发 ─────────────────────────────

def _on_sdk_edge(button: str, pressed: bool) -> None:
    """sdk._buttons 调的 callback. 跑在 cyclonedds reader 线程.

    只处理按下边沿. 松开边沿不触发 binding / fallthrough.

    异常处理: 任何异常 log + return, 不 raise (README §4 — 信号断流比异常传播严重).
    """
    if not pressed:
        return  # 松开不触发

    try:
        remote = _get_remote_state()
        pressed_keys = frozenset(
            btn for btn in VALID_BUTTONS if getattr(remote, btn)
        )
        # Defensive: monitor 现行顺序保证 state.remote() 在 cb 调用前已更新,
        # button 必在 pressed_keys 里. 但若未来 monitor 顺序变化, 补一下防漏.
        if button not in pressed_keys:
            pressed_keys = pressed_keys | {button}

        _dispatch_press(button, pressed_keys)
    except Exception:
        logger.exception("control_pad _on_sdk_edge 异常 (button=%s, 隔离).", button)


def _dispatch_press(button: str, pressed_keys: frozenset[str]) -> None:
    """核心调度逻辑. 接受已知 pressed_keys, 不读 state.

    这是测试 hook 入口 — tes 脚本通过 _dispatch_press_for_testing 直接调本函数,
    跳过 sdk._buttons 链路, 不依赖 monitor 启动.

    步骤:
      1. lock 内: 找 exact-match bindings → debounce 判 → 入 ring buffer → 收集 (cb, event).
      2. lock 内: 没命中任何 binding → fallthrough 判 → 入 ring buffer → 收集 event.
      3. lock 外: 触发 binding cb / fallthrough listener / event listener.

    cb 在 lock 外触发, 防 cb 内调 register/unregister/drain 死锁.
    """
    global _press_edge_count, _fired_count, _fallthrough_fired_count
    global _debounce_suppressed_count, _last_fallthrough_triggered
    global _forgotten_since_last_drain

    now_mono = time.monotonic()
    pressed_keys_tuple = tuple(sorted(pressed_keys))

    # 阶段 1: lock 内, 算 + 入队 + 收集要触发的
    binding_invocations: list[tuple[Callable[[KeyEvent], None], KeyEvent]] = []
    fallthrough_event: Optional[KeyEvent] = None

    with _state_lock:
        _press_edge_count += 1
        matched_any = False
        for handle, binding in _bindings.items():
            if binding.keys != pressed_keys:
                continue
            matched_any = True
            last = _last_triggered.get(handle, 0.0)
            if (now_mono - last) < binding.debounce_sec:
                _debounce_suppressed_count += 1
                continue
            _last_triggered[handle] = now_mono
            _fired_count += 1

            event = KeyEvent(
                binding_name=binding.name,
                pressed_keys=pressed_keys_tuple,
                is_fallthrough=False,
            )
            was_full = len(_dq) == _dq.maxlen
            _dq.append(event)
            if was_full:
                _forgotten_since_last_drain += 1

            cb = _binding_callbacks.get(handle)
            if cb is not None:
                binding_invocations.append((cb, event))

        if not matched_any:
            if (now_mono - _last_fallthrough_triggered) >= _fallthrough_debounce_sec:
                _last_fallthrough_triggered = now_mono
                _fallthrough_fired_count += 1
                fallthrough_event = KeyEvent(
                    binding_name="",
                    pressed_keys=pressed_keys_tuple,
                    is_fallthrough=True,
                )
                was_full = len(_dq) == _dq.maxlen
                _dq.append(fallthrough_event)
                if was_full:
                    _forgotten_since_last_drain += 1
            else:
                _debounce_suppressed_count += 1

    # 阶段 2: lock 外触发 cb
    # binding cb
    for cb, event in binding_invocations:
        try:
            cb(event)
        except Exception:
            logger.exception(
                "binding callback raised (binding=%s, 隔离).", event.binding_name,
            )

    # fallthrough listeners
    if fallthrough_event is not None:
        with _listeners_lock:
            ft_snapshot = list(_fallthrough_listeners.values())
        for ft_cb in ft_snapshot:
            try:
                ft_cb(pressed_keys_tuple, fallthrough_event)
            except Exception:
                logger.exception("fallthrough listener raised (隔离).")

    # event listeners (任何 binding 命中 + fallthrough 都通知)
    all_events = [e for _, e in binding_invocations]
    if fallthrough_event is not None:
        all_events.append(fallthrough_event)
    if all_events:
        with _listeners_lock:
            ev_snapshot = list(_event_listeners.values())
        for ev in all_events:
            for ev_cb in ev_snapshot:
                try:
                    ev_cb(ev)
                except Exception:
                    logger.exception("event listener raised (隔离).")


# ── 无状态 helper (channel 层用) ─────────────────────────────────────────
# Runtime 不直接产 Message — Message 是 channel 层 "何时/怎么/要不要喂模型"
# 的事. 这里只提供无状态转换器, channel 按需调.

def to_xml_text(e: KeyEvent) -> str:
    """单条事件 → 紧凑 XML. 多条用 batch_to_xml."""
    keys_str = "+".join(e.pressed_keys)
    attrs = [f'id="{e.id}"', f'ts="{e.triggered_at:.3f}"', f'keys="{keys_str}"']
    if e.is_fallthrough:
        attrs.append('fallthrough="true"')
        tag = "g1.control_pad.fallthrough"
        body = keys_str
    else:
        attrs.append(f'binding="{e.binding_name}"')
        tag = "g1.control_pad.event"
        body = e.binding_name
    return f'<{tag} {" ".join(attrs)}>{body}</{tag}>'


def batch_to_xml(batch: KeyEventBatch) -> str:
    """整批 → 单段 XML, 含 forgotten 元信息. 模型看到 forgotten>0 即知有 gap."""
    lines = [
        f'<g1.control_pad forgotten="{batch.forgotten}" count="{len(batch.items)}">'
    ]
    for e in batch.items:
        lines.append("  " + to_xml_text(e))
    lines.append("</g1.control_pad>")
    return "\n".join(lines)


def to_message(e: KeyEvent) -> Message:
    """单条事件 → ghoshell_moss Message. channel 把它入 context_messages."""
    attrs: dict = {"id": e.id, "keys": "+".join(e.pressed_keys)}
    if e.is_fallthrough:
        attrs["fallthrough"] = True
        tag = "g1.control_pad.fallthrough"
    else:
        attrs["binding"] = e.binding_name
        tag = "g1.control_pad.event"
    return Message.new(
        tag=tag,
        attributes=attrs,
        timestamp=True,
    ).with_content("+".join(e.pressed_keys))


def batch_to_message(batch: KeyEventBatch) -> Message:
    """整批 → 单条 Message, forgotten 进 attributes. channel pop 后入 context."""
    msg = Message.new(
        tag="g1.control_pad",
        attributes={"forgotten": batch.forgotten, "count": len(batch.items)},
        timestamp=True,
    )
    for e in batch.items:
        msg = msg.with_content(to_xml_text(e))
    return msg


# ── 测试 hooks (内部, 不暴露到 __init__.py) ───────────────────────────────
# tes 脚本走这两个 hook, 跳过 sdk._buttons 链路, 不依赖 G1 实机按按键.
# sen 脚本走真实 start() + 实机按按键, 双工体验.

def _configure_for_testing(
    *,
    buffer_size: int = 64,
    default_debounce_sec: float = 0.20,
    global_min_debounce_sec: float = 0.05,
    fallthrough_debounce_sec: float = 1.0,
) -> None:
    """测试 hook — 跳过 start() (不调 sdk register), 直接配置模块级状态.

    用法: tes 脚本顶部调本函数, 然后 register_binding + _dispatch_press_for_testing.
    """
    global _dq, _default_debounce_sec, _global_min_debounce_sec
    global _fallthrough_debounce_sec, _running, _last_fallthrough_triggered
    global _press_edge_count, _fired_count, _fallthrough_fired_count
    global _debounce_suppressed_count, _forgotten_since_last_drain

    with _state_lock:
        _dq = deque(maxlen=buffer_size)
        _bindings.clear()
        _binding_callbacks.clear()
        _last_triggered.clear()
        _last_fallthrough_triggered = 0.0
        _press_edge_count = 0
        _fired_count = 0
        _fallthrough_fired_count = 0
        _debounce_suppressed_count = 0
        _forgotten_since_last_drain = 0
        _default_debounce_sec = default_debounce_sec
        _global_min_debounce_sec = global_min_debounce_sec
        _fallthrough_debounce_sec = fallthrough_debounce_sec
        _running = True  # 让 register_binding 正常工作
    with _listeners_lock:
        _fallthrough_listeners.clear()
        _event_listeners.clear()


def _dispatch_press_for_testing(
    button: str,
    pressed_keys: set[str] | frozenset[str] | list[str] | tuple[str, ...],
) -> None:
    """测试 hook — 直接驱动 _dispatch_press, 跳过 sdk 链路.

    button: 触发本次边沿的"主键"名 (用于 race defensive, 加进 pressed_keys 防漏).
    pressed_keys: 当前按下的物理键集合.
    """
    pks = frozenset(pressed_keys) | {button}
    _dispatch_press(button, pks)
