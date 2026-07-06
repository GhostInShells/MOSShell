"""
G1 FSM 模块 — 状态格 + 按键路由 + 变化分发.

═══════════════════════════════════════════════════════════════════════════════
定位
═══════════════════════════════════════════════════════════════════════════════

本模块是 MOSS 在 G1 上的授权 / 状态中枢. 三件事:

1. **持有三元组状态**: (ai_mode, sport_mode, auth_level)
2. **订阅两个上游**: control_pad 按键事件 + sdk.state sport_mode 回调
3. **分发两种下游回调**: 状态变化 + AI 模式按键 (X/A/Y)

关键设计约束: **本模块不定义 "状态机" transition 表**. 按键 → 状态字段的映射是
procedural 的 if/elif, 由 handler 函数直接写. 不做"哪些迁移合法/非法"校验.
授权与能力的关系由各 channel 在自己的 available() 里通过 need_fsm_state 查询.

═══════════════════════════════════════════════════════════════════════════════
三元组语义
═══════════════════════════════════════════════════════════════════════════════

- **ai_mode: bool** — MOSS 是否被授权控制 G1. 根阀门. False 时按键 binding 常驻,
  但语义 dispatch 关闸: `_set_auth_level` / `_exit_ai_mode` 内部早退, `_dispatch_button`
  跳过下游 listener. 按键 history 无条件写入 ring buffer, 模型能通过 recent_events
  看到"人按了 A 但没生效", 主动教人类先按 L1+Start.

- **sport_mode: FsmMode** — G1 主机当前 FSM 模式 (Sit/Stand/WalkRun/...).
  来源 sdk.state.register_sport_mode_callback. MOSS 只读, 不写.

- **auth_level: int** — MOSS 授权档 0..3. AI 模式激活时才有意义. 直选模型 —
  按 L1+方向直接跳到目标档, 不做升降. 每档具体授权哪些能力由 channel 自己在
  need_fsm_state 里声明所需档位集合, 无中央表.

═══════════════════════════════════════════════════════════════════════════════
按键映射 (binding 常驻注册, 效果按 _ai_mode 关闸)
═══════════════════════════════════════════════════════════════════════════════

**根阀门键** (无授权关闸, 任何时候都生效):
- L1+Start → 进 AI 模式 (ai_mode=True, auth_level=0)

**AI 模式语义键** (binding 常驻, 语义效果需 ai_mode=True):
- L1+Select → 显式退 AI 模式 (`_exit_ai_mode` 内部早退)
- 摇杆任一轴 |v| > 0.15 → 退 AI 模式 (20Hz 轮询, 表示人类接管)
- L1+上   → auth_level = 0 (`_set_auth_level` 内部早退)
- L1+右   → auth_level = 1
- L1+下   → auth_level = 2
- L1+左   → auth_level = 3
- X → interrupt (`_dispatch_button` 按 _ai_mode 关闸下游 listener)
- A → trigger
- Y → audio_toggle (自由对话切换)
- F1 → listener_toggle (ASR 硬开关, 等价于耳机中键)

方向键映射固化. 修改点集中在 _AI_MODE_BUTTONS + BTN_AUTH_* 常量.

═══════════════════════════════════════════════════════════════════════════════
使用形态
═══════════════════════════════════════════════════════════════════════════════

**channel.available() 查询式**:

    from ghoshell_moss_contrib.unitree.g1.runtime import story_202607_fsm as fsm

    @arms.build.command(
        available=lambda: fsm.need_fsm_state([
            ([FsmMode.STAND, FsmMode.WALK_RUN], [1, 2]),
        ]),
    )
    async def wave() -> Observe: ...

**副作用响应 (LED / TTS 提示)**:

    def _on_state_change(snapshot):
        ai, sm, lv = snapshot
        janus_q.sync_q.put(("state", snapshot))

    handle = fsm.register_change_callback(_on_state_change)

**AI 按键钩子 (X/A/Y 分发)**:

    def _on_button(name):
        # name in {"interrupt", "trigger", "audio_toggle", "listener_toggle"}
        janus_q.sync_q.put(("button", name))

    handle = fsm.register_button_callback(_on_button)

**cb 跑在 cyclonedds reader 线程**, 不能阻塞. 复杂业务通过 janus / threading.Event
卸载到调用方自己的执行环境 (channel asyncio loop / runtime daemon thread).

═══════════════════════════════════════════════════════════════════════════════
生命周期
═══════════════════════════════════════════════════════════════════════════════

- start(): 一次性. 挂 sport_mode 回调 + 常驻注册全部 button binding + 起摇杆轮询 daemon.
- stop(): 反注册 + 停轮询. 幂等.
- 前置: sdk.bootstrap() + control_pad.start() 必须先跑.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional
from uuid import uuid4

from ghoshell_moss.core.blueprint.channel_builder import ObserveError

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad
from ghoshell_moss_contrib.unitree.g1.sdk import (
    FsmMode,
    register_sport_mode_callback,
    unregister_sport_mode_callback,
    remote as _get_remote_state,
)

logger = logging.getLogger("moss.g1.runtime.story_202607_fsm")


# ═══════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════

AUTH_LEVEL_MIN = 0
AUTH_LEVEL_MAX = 3
"""授权档整数值域. 0=零权限 (仅 AI 在线, 无运控), 1/2/3 由各能力自己解释.

刻意用整数而不是枚举: 每档具体含义 case-by-case, 各 channel 在 available()
里通过 need_fsm_state 声明允许的档位集合. 不做集中枚举.
"""


# 按键 binding — 全部用 frozenset (control_pad 精确匹配, 非子集)
BTN_AI_ENTER = frozenset({"l1", "start"})       # 进 AI 模式 (根阀门)
BTN_AI_EXIT = frozenset({"l1", "select"})       # 退 AI 模式 (显式)
BTN_AUTH_0 = frozenset({"l1", "up"})            # 授权直选 → 0
BTN_AUTH_1 = frozenset({"l1", "right"})         # 授权直选 → 1
BTN_AUTH_2 = frozenset({"l1", "down"})          # 授权直选 → 2
BTN_AUTH_3 = frozenset({"l1", "left"})          # 授权直选 → 3
BTN_INTERRUPT = frozenset({"x"})                # 中断当前 command loop (下游)
BTN_TRIGGER = frozenset({"a"})                  # 触发模型立即回复 (下游)
BTN_AUDIO_TOGGLE = frozenset({"y"})             # 自由对话模式切换 (下游)
BTN_LISTENER_TOGGLE = frozenset({"f1"})         # listener ASR 硬开关 (下游, 等价耳机中键)

# 参考: 观测但不 bind (L2+B = G1 硬件急停, 我们只知情, 不处理)
BTN_HARDWARE_ESTOP = frozenset({"l2", "b"})

# AI 模式激活时新增监听的 binding: 语义名 → 键集
# 命名 auth_N 直接编码目标档位, handler 一行 lambda 即可.
_AI_MODE_BUTTONS: dict[str, frozenset[str]] = {
    "ai_exit": BTN_AI_EXIT,
    "auth_0": BTN_AUTH_0,
    "auth_1": BTN_AUTH_1,
    "auth_2": BTN_AUTH_2,
    "auth_3": BTN_AUTH_3,
    "interrupt": BTN_INTERRUPT,
    "trigger": BTN_TRIGGER,
    "audio_toggle": BTN_AUDIO_TOGGLE,
    "listener_toggle": BTN_LISTENER_TOGGLE,
}

# 下游 button listener 收到的语义名 (以上 keys 的子集, 其余是内部状态迁移)
_DOWNSTREAM_BUTTONS: frozenset[str] = frozenset({
    "interrupt", "trigger", "audio_toggle", "listener_toggle",
})

# 摇杆退出阈值 (任一轴 |v| > threshold → 退 AI). 4 轴: lx / ly / rx / ry.
JOYSTICK_EXIT_THRESHOLD = 0.15

# 摇杆轮询频率
_JOYSTICK_POLL_HZ = 20.0

# 事件历史 ring buffer 容量. channel 层 context_messages 只出最近若干条,
# 这里保留稍多余量给 debug / 未来其他消费者.
_HISTORY_MAXLEN = 20


# ═══════════════════════════════════════════════════════════════════════════════
# 类型别名
# ═══════════════════════════════════════════════════════════════════════════════

# 三元组快照
StateSnapshot = tuple[bool, FsmMode, int]

# 下游状态变化回调 — 任何字段变化后触发一次, 拿到新的完整三元组
ChangeCallback = Callable[[StateSnapshot], None]

# 下游 AI 按键回调 — 收到语义名 ("interrupt" / "trigger" / "audio_toggle")
ButtonCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class FsmEvent:
    """状态迁移或按键事件. ring buffer 每条一个.

    - ts: 事件时刻 (time.time() 秒).
    - kind: "ai_mode" | "auth_level" | "sport_mode" | "button".
    - source: 触发源 — 按键组合 (`l1+start` / `l1+up` / `x` / ...) 或 `g1_host` (主机 FSM 切) 或 `joystick`.
    - text: 已格式化的变化描述, 直接可给模型看
      (`AI on` / `0→1` / `1→0` / `STAND→WALK_RUN` / `interrupt`).
    """
    ts: float
    kind: str
    source: str
    text: str


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级私有状态 — 单例, 两把锁
# ═══════════════════════════════════════════════════════════════════════════════

_state_lock = threading.Lock()      # 保护三元组 + 上游句柄
_listeners_lock = threading.Lock()  # 保护下游 listener 注册表

# 三元组. GIL 保证单字段引用赋值原子, 但复合读需锁.
_ai_mode: bool = False
_sport_mode: FsmMode = FsmMode.UNKNOWN
_auth_level: int = AUTH_LEVEL_MIN

# 下游回调注册表 (handle → cb)
_change_listeners: dict[str, ChangeCallback] = {}
_button_listeners: dict[str, ButtonCallback] = {}

# 事件历史 ring buffer. 状态迁移点内部写, channel 层通过 recent_events 读.
# 满了自动挤旧, 不告警.
_history: deque[FsmEvent] = deque(maxlen=_HISTORY_MAXLEN)

# 上游 binding handle. name → control_pad handle (str).
# "ai_enter" 常驻; AI 模式其他 binding 进 AI 时增, 退 AI 时删.
_control_pad_handles: dict[str, str] = {}

# sdk sport_mode 回调 handle
_sport_mode_handle: Optional[str] = None

# 摇杆轮询 daemon
_joystick_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None

_running: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# 生命周期
# ═══════════════════════════════════════════════════════════════════════════════

def start() -> None:
    """启动 FSM 模块. 幂等.

    前置: sdk.bootstrap() + control_pad.start() 已完成.

    步骤:
      1. 重置三元组 → (False, UNKNOWN, L0).
      2. 注册 sdk sport_mode 回调.
      3. 常驻注册全部 button binding (L1+Start / L1+Select / L1+方向 / X / A / Y / F1).
         AI 模式外这些 binding 仍然收得到按键事件, history ring buffer 会写, 但
         语义 dispatch 由各 handler / _dispatch_button 内部按 _ai_mode 关闸.
         设计意图: binding 常驻, 效果按授权关闸 (人按了 A 但没在 AI 模式 →
         history 有事件, 但不 dispatch 到下游 listener).
      4. 起摇杆轮询 daemon.
    """
    global _running, _ai_mode, _sport_mode, _auth_level
    global _sport_mode_handle, _joystick_thread, _stop_event

    with _state_lock:
        if _running:
            logger.debug("start() re-entered — already running, skip.")
            return
        _ai_mode = False
        _sport_mode = FsmMode.UNKNOWN
        _auth_level = AUTH_LEVEL_MIN
        _control_pad_handles.clear()
        _history.clear()
        _stop_event = threading.Event()
        _running = True

    # sdk sport_mode 回调 (若已有已知 mode, 会立即以 (-1, current) fire 一次)
    _sport_mode_handle = register_sport_mode_callback(_on_sport_mode_change)

    # 根阀门 binding
    handle = control_pad.register_binding(
        name="fsm_ai_enter",
        keys=BTN_AI_ENTER,
        callback=_on_ai_enter_pressed,
    )
    with _state_lock:
        _control_pad_handles["ai_enter"] = handle

    # AI 模式内 button binding — 常驻注册, 关闸在 handler / _dispatch_button 内做.
    # 各 handler 的授权语义:
    #   - ai_exit → _exit_ai_mode 内部 `if not _ai_mode: return` 早退
    #   - auth_N → _set_auth_level 内部 `if not _ai_mode: return` 早退
    #   - interrupt / trigger / audio_toggle → _dispatch_button 按 _ai_mode 关闸
    for name, keys in _AI_MODE_BUTTONS.items():
        try:
            binding_handle = control_pad.register_binding(
                name=f"fsm_{name}",
                keys=keys,
                callback=_make_ai_button_handler(name),
            )
            with _state_lock:
                _control_pad_handles[name] = binding_handle
        except Exception:
            logger.exception("register AI-mode binding %s failed", name)

    # 摇杆轮询
    _joystick_thread = threading.Thread(
        target=_joystick_poll_loop,
        name="g1-fsm-joystick",
        daemon=True,
    )
    _joystick_thread.start()

    logger.info("story_202607_fsm started.")


def stop(timeout: float = 2.0) -> None:
    """停止 FSM 模块. 反注册全部 binding + 停轮询. 幂等."""
    global _running, _sport_mode_handle, _joystick_thread, _stop_event

    with _state_lock:
        if not _running:
            logger.debug("stop() re-entered — not running, skip.")
            return
        _running = False
        if _stop_event is not None:
            _stop_event.set()
        handles_snapshot = dict(_control_pad_handles)
        _control_pad_handles.clear()
        sport_handle = _sport_mode_handle
        _sport_mode_handle = None
        thread = _joystick_thread
        _joystick_thread = None

    # 反注册 (锁外, 防 reader 线程死锁)
    for name, h in handles_snapshot.items():
        try:
            control_pad.unregister_binding(h)
        except Exception:
            logger.exception("unregister control_pad binding %s failed (ignored)", name)
    if sport_handle is not None:
        try:
            unregister_sport_mode_callback(sport_handle)
        except Exception:
            logger.exception("unregister sport_mode callback failed (ignored)")
    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("joystick poller join timeout.")

    logger.info("story_202607_fsm stopped.")


def is_running() -> bool:
    with _state_lock:
        return _running


# ═══════════════════════════════════════════════════════════════════════════════
# 查询 API — channel available() / command available() 用
# ═══════════════════════════════════════════════════════════════════════════════

def read() -> StateSnapshot:
    """快照三元组 (ai_mode, sport_mode, auth_level).

    加锁读, 保证三个字段互相一致 (无中间态). 无 I/O.
    """
    with _state_lock:
        return (_ai_mode, _sport_mode, _auth_level)


def get_ai_mode() -> bool:
    """当前是否已进入 AI 模式. GIL 原子读."""
    return _ai_mode


def get_sport_mode() -> FsmMode:
    """G1 主机当前 FSM 模式. GIL 原子读."""
    return _sport_mode


def get_auth_level() -> int:
    """当前 MOSS 授权档 (整数 0..3). GIL 原子读."""
    return _auth_level


def need_fsm_state(
    requirements: list[tuple[list[FsmMode], list[int]]],
) -> bool:
    """判断当前三元组是否命中给定要求组的任一个 — channel/command available() 用.

    语义:
      - AI 模式未激活 → 无条件 False (根阀门).
      - AI 模式激活 → 遍历 requirements, 任一 (sport_mode ∈ list[0]
        AND auth_level ∈ list[1]) 命中即返回 True.

    Args:
        requirements: 允许状态组列表. 每组是 (sport_modes, auth_levels).
                      auth_levels 是整数列表 (0..3), 每档具体授权哪些能力由
                      各能力自己在此表里声明.

    Returns:
        命中 True, 否则 False.

    使用示例:
        # 站姿或走跑, auth 1 或 2
        need_fsm_state([
            ([FsmMode.STAND, FsmMode.WALK_RUN], [1, 2]),
        ])

        # 多组合并
        need_fsm_state([
            ([FsmMode.STAND],    [1, 2, 3]),
            ([FsmMode.WALK_RUN], [2, 3]),
        ])

    非运动 channel (led / speaker / system / sensors) 不应调用本函数 —
    它们不受授权门控.
    """
    ai, sport, auth = read()
    if not ai:
        return False
    for allowed_sport, allowed_auth in requirements:
        if sport in allowed_sport and auth in allowed_auth:
            return True
    return False


def recent_events(limit: int = 5) -> list[FsmEvent]:
    """近期事件, 最新在末尾. channel context_messages 用.

    :param limit: 最多返回条数. 超过 buffer 容量则给 buffer 全量.
    :return: FsmEvent 列表, 时间升序 (老 → 新).
    """
    with _state_lock:
        if limit <= 0:
            return []
        if limit >= len(_history):
            return list(_history)
        return list(_history)[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# 下游注册 API — channel startup 用
# ═══════════════════════════════════════════════════════════════════════════════

def register_change_callback(cb: ChangeCallback) -> str:
    """注册状态变化回调.

    任何字段 (ai_mode / sport_mode / auth_level) 变化后触发一次,
    cb 拿到新的完整三元组快照.

    **cb 跑在 cyclonedds reader 线程 (或摇杆轮询线程)**, 不能阻塞.
    典型用法: cb 内部 push 到 janus.Queue 或 threading.Event, 主线程消费.

    Returns:
        handle (uuid hex), 用于 unregister.
    """
    handle = uuid4().hex
    with _listeners_lock:
        _change_listeners[handle] = cb
    return handle


def unregister_change_callback(handle: str) -> None:
    """反注册. 未知 handle 静默忽略."""
    with _listeners_lock:
        _change_listeners.pop(handle, None)


def register_button_callback(cb: ButtonCallback) -> str:
    """注册 AI 模式按键回调.

    binding 常驻注册, 按键事件在 AI 模式外仍进 history ring buffer, 但下游
    dispatch (即本 cb 被调用) 只在 AI 模式激活期间发生. AI 模式外按下
    X / A / Y, 模型可通过 `recent_events()` 看到 "人按了 A 但没生效" 事件,
    从而主动教人类先按 L1+Start.

    收到 cb 时对应语义名: "interrupt" / "trigger" / "audio_toggle".

    **cb 跑在 cyclonedds reader 线程**, 不能阻塞.
    """
    handle = uuid4().hex
    with _listeners_lock:
        _button_listeners[handle] = cb
    return handle


def unregister_button_callback(handle: str) -> None:
    """反注册. 未知 handle 静默忽略."""
    with _listeners_lock:
        _button_listeners.pop(handle, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 内部: 状态迁移 (procedural, 不建 transition 表)
# ═══════════════════════════════════════════════════════════════════════════════

def _enter_ai_mode() -> None:
    """L1+Start → 进 AI 模式. 翻状态位 + 写 history + 分发 change.

    binding 在 start() 中常驻注册, 本函数不管 binding 生命周期. AI 模式外
    X/A/Y 会照常触发 handler → _dispatch_button, 后者按 _ai_mode 关闸
    dispatch 到下游 listener. history 无论如何都写.
    """
    global _ai_mode, _auth_level
    snapshot: Optional[StateSnapshot] = None

    with _state_lock:
        if _ai_mode:
            return
        _ai_mode = True
        _auth_level = AUTH_LEVEL_MIN
        _record_event_locked("ai_mode", "l1+start", "AI on")
        snapshot = (_ai_mode, _sport_mode, _auth_level)

    logger.info("AI mode entered → %s", _fmt(snapshot))
    _dispatch_change(snapshot)


def _exit_ai_mode(source: str = "unknown") -> None:
    """L1+Select / 摇杆 → 退 AI 模式. 翻状态位 + 写 history + 分发 change.

    :param source: 触发源, 进 history 事件的 source 字段. "l1+select" / "joystick".

    binding 常驻注册 (见 start()), 本函数不管 binding 生命周期.
    """
    global _ai_mode, _auth_level
    snapshot: Optional[StateSnapshot] = None

    with _state_lock:
        if not _ai_mode:
            return
        _ai_mode = False
        _auth_level = AUTH_LEVEL_MIN
        _record_event_locked("ai_mode", source, "AI off")
        snapshot = (_ai_mode, _sport_mode, _auth_level)

    logger.info("AI mode exited → %s", _fmt(snapshot))
    _dispatch_change(snapshot)


def _set_auth_level(target: int, source: str) -> None:
    """AI 模式内直选授权档 (L1+方向). target 必须在 [MIN, MAX] 内.

    :param target: 目标档位整数 (0..3).
    :param source: 触发源 (`l1+up` / `l1+right` / `l1+down` / `l1+left`),
                   进 history event 的 source 字段.
    """
    global _auth_level
    snapshot: Optional[StateSnapshot] = None
    if target < AUTH_LEVEL_MIN or target > AUTH_LEVEL_MAX:
        logger.error("_set_auth_level: invalid target %d (allowed %d..%d)",
                     target, AUTH_LEVEL_MIN, AUTH_LEVEL_MAX)
        return
    with _state_lock:
        if not _ai_mode:
            return
        if _auth_level == target:
            return
        old = _auth_level
        _auth_level = target
        _record_event_locked("auth_level", source, f"{old}→{target}")
        snapshot = (_ai_mode, _sport_mode, _auth_level)

    logger.info("auth level → %d", snapshot[2])
    _dispatch_change(snapshot)


# ═══════════════════════════════════════════════════════════════════════════════
# 内部: 上游回调
# ═══════════════════════════════════════════════════════════════════════════════

def _on_ai_enter_pressed(_evt) -> None:
    """L1+Start binding 触发 (跑在 reader 线程)."""
    _enter_ai_mode()


def _make_ai_button_handler(name: str) -> Callable:
    """为每个 AI 模式 binding 生成 control_pad handler.

    - ai_exit: 显式退 AI 模式
    - auth_N (N in 0..3): 直选授权档 N
    - interrupt / trigger / audio_toggle: 分发到下游 button listener
    """
    if name == "ai_exit":
        return lambda _evt: _exit_ai_mode("l1+select")
    if name.startswith("auth_"):
        try:
            target = int(name.split("_", 1)[1])
        except ValueError:
            logger.error("bad auth binding name: %s", name)
            return lambda _evt: None
        # L1+方向 → 映射源
        _source_map = {0: "l1+up", 1: "l1+right", 2: "l1+down", 3: "l1+left"}
        src = _source_map.get(target, f"l1+auth_{target}")
        return lambda _evt, _t=target, _s=src: _set_auth_level(_t, _s)
    if name in _DOWNSTREAM_BUTTONS:
        return lambda _evt, _n=name: _dispatch_button(_n)
    logger.error("unknown AI mode button name: %s", name)
    return lambda _evt: None


def _on_sport_mode_change(_old: int, new: int) -> None:
    """sdk sport_mode 回调 (跑在 reader 线程).

    仅更新三元组的 sport_mode 字段, 不做授权自动降级 — auth 的合法性
    由各 channel 在 need_fsm_state 里判定. 保持"FSM 无内建 transition 表"约束.
    """
    global _sport_mode
    snapshot: Optional[StateSnapshot] = None
    try:
        new_mode = FsmMode(new)
    except ValueError:
        new_mode = FsmMode.UNKNOWN

    with _state_lock:
        if _sport_mode == new_mode:
            return
        old = _sport_mode
        _sport_mode = new_mode
        _record_event_locked("sport_mode", "g1_host", f"{old.name}→{new_mode.name}")
        snapshot = (_ai_mode, _sport_mode, _auth_level)

    logger.info("sport_mode → %s (%d)", new_mode.name, new)
    _dispatch_change(snapshot)


# ═══════════════════════════════════════════════════════════════════════════════
# 内部: 摇杆轮询
# ═══════════════════════════════════════════════════════════════════════════════

def _joystick_poll_loop() -> None:
    """摇杆退出检测 daemon. 20Hz 读 sdk remote() 快照, 任一轴超阈值 → 退 AI.

    实机验证前的行为约定 (可能需实测调整):
    - 仅在 AI 模式激活时检测 (否则遥控器归 G1 主机, MOSS 不参与)
    - 单帧超阈值即触发 (不做时间窗防抖 — 摇杆是有意的物理动作, 抖动模型可后续加)
    - _get_remote_state() 抛 RuntimeError (monitor 未启动) 时静默跳过
    """
    interval = 1.0 / _JOYSTICK_POLL_HZ
    stop_event = _stop_event
    threshold = JOYSTICK_EXIT_THRESHOLD
    logger.info("joystick poller started (%.1fHz, threshold=%.2f).", _JOYSTICK_POLL_HZ, threshold)

    while not stop_event.is_set():
        try:
            if _ai_mode:
                r = _get_remote_state()
                if (abs(r.lx) > threshold or abs(r.ly) > threshold or
                        abs(r.rx) > threshold or abs(r.ry) > threshold):
                    logger.info(
                        "joystick displacement (lx=%.2f ly=%.2f rx=%.2f ry=%.2f) → exit AI",
                        r.lx, r.ly, r.rx, r.ry,
                    )
                    _exit_ai_mode("joystick")
        except RuntimeError:
            pass  # monitor 未启动或首帧未到, 忽略
        except Exception:
            logger.exception("joystick poll error (isolated)")
            time.sleep(0.1)
        stop_event.wait(interval)

    logger.info("joystick poller exited.")


# ═══════════════════════════════════════════════════════════════════════════════
# 内部: 下游分发
# ═══════════════════════════════════════════════════════════════════════════════

def _dispatch_change(snapshot: StateSnapshot) -> None:
    """通知全部 change listener. 锁外调用, cb 异常隔离."""
    with _listeners_lock:
        cbs = list(_change_listeners.values())
    for cb in cbs:
        try:
            cb(snapshot)
        except Exception:
            logger.exception("change listener raised (isolated)")


def _dispatch_button(button_name: str) -> None:
    """通知全部 button listener. 锁外调用, cb 异常隔离.

    History 无论授权与否都写 — 模型能看到"人按了 X 但没生效", 主动教人类
    先按 L1+Start 进 AI 模式. dispatch 到下游 listener 按 _ai_mode 关闸,
    授权外 no-op (符合 "binding 常驻, 没授权时按键不生效" 设计).

    Source 就是按键本身 (x / a / y), 与 name 对应关系固定.
    """
    _source_map = {"interrupt": "x", "trigger": "a", "audio_toggle": "y"}
    with _state_lock:
        _record_event_locked("button", _source_map.get(button_name, "?"), button_name)
        authorized = _ai_mode

    if not authorized:
        logger.info("button %s pressed but AI mode off, dispatch skipped", button_name)
        return

    with _listeners_lock:
        cbs = list(_button_listeners.values())
    for cb in cbs:
        try:
            cb(button_name)
        except Exception:
            logger.exception("button listener (%s) raised (isolated)", button_name)


def _record_event_locked(kind: str, source: str, text: str) -> None:
    """写一条事件到 history ring buffer. **调用方必须持 _state_lock**.

    满了自动挤旧 — deque(maxlen) 天然行为, 不告警不阻塞.
    """
    _history.append(FsmEvent(ts=time.time(), kind=kind, source=source, text=text))


def _fmt(snapshot: StateSnapshot) -> str:
    """三元组日志格式化."""
    ai, sm, lv = snapshot
    return f"(ai={ai}, sport={sm.name}, auth={lv})"


# ═══════════════════════════════════════════════════════════════════════════════
# 现场调试
# ═══════════════════════════════════════════════════════════════════════════════

def health() -> dict:
    """暴露内部状态, 供 monitor / channel debug."""
    with _state_lock:
        return {
            "running": _running,
            "ai_mode": _ai_mode,
            "sport_mode": _sport_mode.name,
            "sport_mode_id": int(_sport_mode),
            "auth_level": _auth_level,
            "control_pad_bindings": list(_control_pad_handles.keys()),
            "change_listeners": len(_change_listeners),
            "button_listeners": len(_button_listeners),
            "history_len": len(_history),
            "history_max": _history.maxlen,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Warrant — 状态变化的硬中断上下文管理器
# ═══════════════════════════════════════════════════════════════════════════════

class FsmWarrant:
    """状态守卫 async 上下文管理器. 状态变化时打断当前 task, 转成 ObserveError.

    用途: 动作命令 (arms.wave / move.walk 等) 在物理动作执行期间, 若 FSM 三元组
    发生变化且 guard 谓词 fail, 立刻 cancel 当前 task + 抛 ObserveError.

    使用形态:

        # 显式的授权谓词, 同一个函数复用给 available 和 warrant.
        def _wave_available() -> bool:
            return fsm.need_fsm_state([([FsmMode.STAND, FsmMode.WALK_RUN],
                                         [1, 2])])

        @arms.build.command(available=_wave_available)
        async def wave() -> Observe:
            async with fsm.warrant(_wave_available):
                await _run_wave_keyframes()
                # 若中途 auth 变化 → CancelledError → __aexit__ 转 ObserveError
            # 到这里 = 动作跑完, 命令自己决定返回什么
            return CommandUtil.observe("wave completed")

    生命周期:
      - __aenter__ 时抓 asyncio loop + current_task, 注册 change_callback,
        guard() 立刻查一次 (快速失败: 状态在入 warrant 前就已失效, 直接 raise).
      - 状态变化 → change_callback fire (跑 reader / joystick 线程)
        → guard() 若返回 False, self._loop.call_soon_threadsafe(task.cancel)
      - task 下次 await 拿到 CancelledError, 沿路径展开到 __aexit__
      - __aexit__ 反注册 callback; 若 CancelledError 且是**我们自己**造成的,
        raise ObserveError; 否则不拦截.

    区分内外中断:
      - warrant 自己触发的 cancel → _interrupted_by_us = True → ObserveError
      - 外部 (shell / channel Start 键) 触发的 cancel → _interrupted_by_us = False
        → CancelledError 原样传播, 不拦截

    Guard 语义: 谓词返回 True = 仍然授权, 不中断. False = 撤销, 中断.
    通常传入的就是 command available() 里同一个函数.
    """

    __slots__ = (
        "_guard", "_loop", "_task", "_handle",
        "_interrupted_by_us", "_interrupt_reason",
    )

    def __init__(self, guard: Callable[[], bool]):
        self._guard = guard
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._handle: Optional[str] = None
        self._interrupted_by_us: bool = False
        self._interrupt_reason: str = ""

    async def __aenter__(self) -> "FsmWarrant":
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        if self._task is None:
            # defensive — 理论上 async with 内 current_task 必非 None
            raise RuntimeError("FsmWarrant must be entered inside an asyncio task")

        # Fast fail: guard 在 warrant 建立时就 False, 不进入
        # 注意先查后注册, 否则 register + immediate change 会打乱顺序
        if not self._guard():
            snapshot = read()
            raise ObserveError(
                f"action rejected — guard fails at warrant entry: {_fmt(snapshot)}"
            )

        self._handle = register_change_callback(self._on_change)
        return self

    async def __aexit__(self, exc_type, exc_val, tb) -> None:
        if self._handle is not None:
            unregister_change_callback(self._handle)
            self._handle = None

        # 只在"我们自己 cancel 的 + 确实收到 CancelledError" 时转成 ObserveError.
        # 其他情况 (正常完成 / 外部 cancel / 命令自己抛别的异常) 一律不拦截.
        if self._interrupted_by_us and exc_type is asyncio.CancelledError:
            logger.info("warrant interrupt: %s", self._interrupt_reason)
            raise ObserveError(
                f"interrupted by state change: {self._interrupt_reason}"
            )
        # 返回 None (等价 False) — 不抑制任何异常

    def _on_change(self, snapshot: StateSnapshot) -> None:
        """状态变化 callback. 跑在 reader / joystick 线程."""
        try:
            still_ok = self._guard()
        except Exception:
            logger.exception("warrant guard raised (isolated); treating as failed")
            still_ok = False
        if still_ok:
            return

        # guard 失败 → cancel task. 记录原因 + 线程安全 cancel.
        # 幂等: 多次状态变化只 cancel 一次 (后续 task.done() 为 True, cancel no-op)
        self._interrupted_by_us = True
        ai, sp, lv = snapshot
        self._interrupt_reason = f"ai={ai} sport={sp.name} auth={lv}"
        if self._task is not None and not self._task.done():
            self._loop.call_soon_threadsafe(self._task.cancel)


def warrant(guard: Callable[[], bool]) -> FsmWarrant:
    """构造一个 FsmWarrant. guard 是无参谓词 (通常 == command 的 available 函数).

    见 FsmWarrant docstring 详解.
    """
    return FsmWarrant(guard)


# ═══════════════════════════════════════════════════════════════════════════════
# 测试 hook — 依赖无关场景脚本使用, 不暴露到 __init__.py
# ═══════════════════════════════════════════════════════════════════════════════

def _configure_for_testing() -> None:
    """跳过 sdk 依赖的启动步骤: 不注册 sport_mode 回调, 不起摇杆轮询.

    要求: control_pad._configure_for_testing() 先跑 (否则 register_binding 失败).

    binding 侧跟 start() 一致 — 常驻注册全部 button binding (ai_enter +
    _AI_MODE_BUTTONS), 让 _dispatch_button 的 _ai_mode 关闸得到覆盖.

    用法: 在 _fsm_sen_*.py / _fsm_tes_*.py 里调用, 然后用 control_pad
    的 _dispatch_press_for_testing 注入按键, 用 _inject_sport_mode_for_testing
    注入 FSM 变化.
    """
    global _running, _ai_mode, _sport_mode, _auth_level, _stop_event

    with _state_lock:
        _ai_mode = False
        _sport_mode = FsmMode.UNKNOWN
        _auth_level = AUTH_LEVEL_MIN
        _control_pad_handles.clear()
        _history.clear()
        _stop_event = threading.Event()
        _running = True
    with _listeners_lock:
        _change_listeners.clear()
        _button_listeners.clear()

    # 注册根阀门 binding (要求 control_pad 已 configure_for_testing)
    handle = control_pad.register_binding(
        name="fsm_ai_enter",
        keys=BTN_AI_ENTER,
        callback=_on_ai_enter_pressed,
    )
    with _state_lock:
        _control_pad_handles["ai_enter"] = handle

    # 常驻注册 AI 模式 binding, 跟 start() 一致
    for name, keys in _AI_MODE_BUTTONS.items():
        binding_handle = control_pad.register_binding(
            name=f"fsm_{name}",
            keys=keys,
            callback=_make_ai_button_handler(name),
        )
        with _state_lock:
            _control_pad_handles[name] = binding_handle


def _inject_sport_mode_for_testing(new: int) -> None:
    """测试 hook: 注入一次 sport_mode 变化, 走跟 sdk 回调完全一样的路径."""
    _on_sport_mode_change(-1, new)
