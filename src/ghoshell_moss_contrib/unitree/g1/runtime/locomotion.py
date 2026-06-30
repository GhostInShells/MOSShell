"""
Locomotion runtime — G1 空间移动命令的运行时模块.

七个 async 命令面 (前进/后退/横移左右/转身左右 + stop), 全部映射到 LocoClient.Move
基本 RPC. 互斥语义: 单一 _current_version + 抢占切换. 任何新命令进入立即拿到新
version, 老 publish loop 在下一 tick (~50ms) 自己发现 my_version != _current_version,
break + 不发 StopMove (让位给新 session, 避免抢占瞬间 StopMove→Move 抽搐).

设计纪律见同目录 README.md. 关键偏离 asr.py / control_pad.py 的点 (有意识的):

  - 锁拓扑: 全程在 asyncio 内, 用 asyncio.Lock 保护"version 申领 + session 切换"
    复合操作. threading.Lock 仅保护 start/stop/_running flag (因 main.py 可能从非
    asyncio 上下文调). asr.py 是 reader 线程跨 asyncio, threading.Lock 全包.
  - 不实装 drain / register_listener — locomotion 是动作执行器, 没有"累积数据需
    要被 listen"的语义. 命令的"完成"由 async 函数返回值传递, 不走 listener.
  - 返回 Observe 文本 — 命令面契约就是 LLM 看见的 Observe. 不走 to_message helper
    路径 (asr/listener 那一套是被动数据流). reason 字段在 Observe 里显式告诉 LLM
    "你的命令为什么停" (duration/preempted_by:X/stopped/exception), 这是诚实告知.

底层 wrapper (_loco_move_versioned / _loco_stop_versioned) 内做 version 校验 ——
老 session 在 break 之前或 finally 内若仍想 publish, 会被静默挡掉. 这一层防御让
抢占切换不依赖"老 loop 立刻退出"的精确时序.

物理事实参考:
  - V_FORWARD = 0.25 m/s : script 19 实测稳定值. 0.15 低于启动阈值.
  - V_LATERAL = 0.15 m/s : story-2026-07 速度上限. 待实测细化.
  - V_YAW low/medium/high: 三档全是猜值, 实机标定后写回常量.
  - LocoClient.Move(vx, vy, vyaw) 调用持久性 ("一次 Move 持续走" vs "需要 keepalive
    重发"): script 19 暗示一次 Move 持续走, 但保险起见 publish loop 每 tick 重发.
    实测确认不需要后可改为"仅起始时发一次".

不做 (本期范围之外, 有意推迟):
  - 空气墙 / 软边界守护 — 人类工程师定不做, 物理急停走 L2+B 硬件路径.
  - velocity ramp (启停减速曲线) — LocoClient 内部是否平滑未知, 第一版直接发 v.
  - 闭环 turn (用 imu.rpy[2] 真值控制转角) — 等里程计 sdk 接入后再加入.
  - roll_toward_speaker — story P2, 后期闭环.

调用样例 (g1 main.py 形态):

    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import locomotion

    bootstrap(nic="eth0")
    locomotion.start()
    # 在 asyncio 内:
    result = await locomotion.walk_forward(2.0)        # "walk_forward duration after 2.00s"
    result = await locomotion.turn_left(1.5, "medium") # "turn_left_medium duration after 1.50s"
    result = await locomotion.stop()                   # 任何时刻显式停 + 回 stand
    locomotion.stop_runtime()                           # 进程退出前清理
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

from ghoshell_moss_contrib.unitree.g1.sdk import get_loco_client

logger = logging.getLogger("moss.g1.runtime.locomotion")


# ── 物理常量 ──────────────────────────────────────────────────────────────
# 都是 "速度 (m/s 或 rad/s)" 值. 改这些值不影响命令面 (LLM 看到的接口),
# 但改变物理行为. 标定后写回这里.

V_FORWARD: float = 0.25      # 前/后直行, m/s. script 19 稳定值.
V_LATERAL: float = 0.25      # 左/右横移, m/s. 0.15 不动（低于启动阈值）, 0.5 恐怖. 0.25 待明天实测.

V_YAW: dict[str, float] = {  # 转身角速度, rad/s. 全是猜值, 实机标定.
    "low":    0.3,
    "medium": 0.6,
    "high":   1.0,
}

# publish loop tick. 20Hz = 50ms 反应窗口. version 抢占的"最坏延迟"就是这个值.
_TICK_SEC: float = 0.05


# ── 数据契约 ──────────────────────────────────────────────────────────────

@dataclass
class _MoveSession:
    """一次 locomotion 命令的生命周期对象.

    version: 申领时的全局单调递增整数. publish loop 每 tick 比对 my_version
             vs _current_version, 不等就 break (让位给抢占者).
    done:    finally 块跑完后 set. stop() 等待它确保老 loop 真退出.
    reason:  retro-填: "duration" / "preempted_by:<new_cmd>" / "stopped" /
             "exception:<Type>" / "cancelled". 进 Observe 文本.
    """
    version: int
    cmd_name: str
    started_at: float
    done: asyncio.Event
    reason: Optional[str] = None


# ── 模块级私有状态 ────────────────────────────────────────────────────────
# 单例服务. 锁拓扑:
#   _module_lock (threading.Lock):
#     保护 _running flag. start/stop/is_running 跨线程调用安全.
#   _version_lock (asyncio.Lock, 延迟创建):
#     保护 "申领 _next_version + 设置 _current_version/_current_session" 复合操作.
#     单独的引用赋值 (e.g. _current_version = 0) 借 GIL 原子, 不上锁.

_module_lock = threading.Lock()
_running: bool = False

_version_lock: Optional[asyncio.Lock] = None  # 延迟到首次命令时, 绑当前 loop

_next_version: int = 1                         # 单调递增 version 池
_current_version: int = 0                      # 0 = 无 active session (或被 stop 强制失效)
_current_session: Optional[_MoveSession] = None


# ── 公开生命周期 ─────────────────────────────────────────────────────────

def start() -> None:
    """启动 locomotion runtime. 幂等. 不需要 loop 参数 — asyncio.Lock 延迟创建.

    前置: sdk.bootstrap() 已完成 (否则 get_loco_client() raise).

    实际"启动"只是 set _running flag + 清状态. 没有子线程, 没有 DDS subscribe ——
    locomotion 是动作执行器, 不需要持续 reader. 命令 await 进来才有事干.
    """
    global _running, _next_version, _current_version, _current_session
    with _module_lock:
        if _running:
            logger.debug("start() 重入 — 已在运行, 跳过.")
            return
        # 验证 sdk 已 bootstrap (raise 早于第一次命令). get_loco_client 自身会 raise.
        get_loco_client()
        _next_version = 1
        _current_version = 0
        _current_session = None
        _running = True
    logger.info("locomotion runtime started.")


def stop_runtime(timeout: float = 2.0) -> None:
    """停止 runtime. 取消任何 active session + StopMove 兜底. 幂等.

    命名为 stop_runtime 而非 stop, 避开跟命令面 stop() 重名 (那是 async).

    :param timeout: 等待 active session 自退的秒数.
    """
    global _running, _current_version, _current_session
    with _module_lock:
        if not _running:
            logger.debug("stop_runtime() 重入 — 未在运行, 跳过.")
            return
        _running = False
        old = _current_session
        # 失效任何 active session, 让它 break + 不 publish 残帧.
        _current_version = 0

    # 兜底 StopMove —— 不管有无 active session, 进程退出前让 G1 回 stand.
    try:
        get_loco_client().StopMove()
    except Exception:
        logger.exception("StopMove failed in stop_runtime (ignored)")

    # 等老 session 自退 (它 publish loop 下一 tick 会发现 version=0 失效 → break)
    if old is not None and not old.done.is_set() and _version_lock is not None:
        # 不能在非 asyncio 上下文 await — 仅当调用方在 asyncio 内, 老 session 自会退.
        # 同步上下文走 timeout 轮询.
        deadline = time.monotonic() + timeout
        while not old.done.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not old.done.is_set():
            logger.warning(
                "stop_runtime: session %s did not finish in %.1fs (will continue, daemon).",
                old.cmd_name, timeout,
            )

    with _module_lock:
        _current_session = None
    logger.info("locomotion runtime stopped.")


def is_running() -> bool:
    """当前是否运行."""
    with _module_lock:
        return _running


def current_command() -> Optional[str]:
    """active session 的 cmd_name. 无 active 返回 None. 给 channel.idle / debug 看."""
    s = _current_session
    if s is None or s.done.is_set():
        return None
    return s.cmd_name


def health() -> dict:
    """暴露 runtime 内部状态. 供 monitor 脚本 / channel debug 用."""
    s = _current_session
    return {
        "running": _running,
        "current_version": _current_version,
        "next_version": _next_version,
        "current_command": s.cmd_name if (s and not s.done.is_set()) else None,
        "current_command_elapsed_sec": (
            (time.monotonic() - s.started_at) if (s and not s.done.is_set()) else None
        ),
    }


# ── 命令面 (LLM 看到这 7 个 async 函数) ─────────────────────────────────
# Field-style docstrings — 后续 channel 层 reflect 时, 第一句 + :param: 直接进 prompt.

async def walk_forward(duration: float) -> str:
    """G1 向前直行 (机身正面方向). 固定速度 0.25 m/s.

    :param duration: 持续时间 (秒). 命令在此时间到 / 被新命令抢占 / 被 stop 时返回.
    :return: Observe 文本, 形如 "walk_forward duration after 2.00s" 或
             "walk_forward preempted_by:turn_left after 0.34s".
    """
    return await _run_session("walk_forward", duration, vx=V_FORWARD, vy=0.0, vyaw=0.0)


async def walk_backward(duration: float) -> str:
    """G1 向后直行 (机身正面方向的反方向). 固定速度 0.25 m/s.

    :param duration: 持续时间 (秒).
    :return: Observe 文本.
    """
    return await _run_session("walk_backward", duration, vx=-V_FORWARD, vy=0.0, vyaw=0.0)


async def strafe_left(duration: float) -> str:
    """G1 向左横移 (机身正面不变, 整体向左侧平移). 固定速度 0.15 m/s.

    :param duration: 持续时间 (秒).
    :return: Observe 文本.
    """
    return await _run_session("strafe_left", duration, vx=0.0, vy=V_LATERAL, vyaw=0.0)


async def strafe_right(duration: float) -> str:
    """G1 向右横移. 固定速度 0.15 m/s.

    :param duration: 持续时间 (秒).
    :return: Observe 文本.
    """
    return await _run_session("strafe_right", duration, vx=0.0, vy=-V_LATERAL, vyaw=0.0)


async def turn_left(
    duration: float,
    speed: Literal["low", "medium", "high"] = "low",
) -> str:
    """G1 原地左转 (yaw 增加方向). 三档角速度, 通过 duration 控制转角.

    粗略对照: low ≈ 17°/s, medium ≈ 34°/s, high ≈ 57°/s. 实机标定后修订.
    例: low + duration=1.0s ≈ 转 17°; medium + 2.0s ≈ 68°.

    :param duration: 持续时间 (秒).
    :param speed: "low" 谨慎打量, "medium" 一般转身, "high" 应急回头.
    :return: Observe 文本, cmd_name 包含档位 (e.g. "turn_left_medium").
    """
    vyaw = V_YAW[speed]
    return await _run_session(f"turn_left_{speed}", duration, vx=0.0, vy=0.0, vyaw=vyaw)


async def turn_right(
    duration: float,
    speed: Literal["low", "medium", "high"] = "low",
) -> str:
    """G1 原地右转 (yaw 减少方向). 三档角速度, 通过 duration 控制转角.

    :param duration: 持续时间 (秒).
    :param speed: "low" / "medium" / "high".
    :return: Observe 文本.
    """
    vyaw = -V_YAW[speed]
    return await _run_session(f"turn_right_{speed}", duration, vx=0.0, vy=0.0, vyaw=vyaw)


async def stop() -> str:
    """强制停止任何 active 移动. 独立接口 — 不参与互斥队列, 立即生效.

    机制: bump _current_version=0 失效所有 publish loop + 直接 LocoClient.StopMove().
    与命令式 walk/turn 不同, stop 不依赖老 session finally (后者也被 version 校验挡掉),
    自己负责发 StopMove. 等老 session 自退最多 0.5s.

    :return: Observe 文本, 如 "stopped walk_forward" 或 "no active move; stopped anyway".
    """
    global _current_version, _current_session

    if not _running:
        return "<observe>locomotion not running</observe>"
    if _version_lock is None:
        # 还没有任何命令跑过 — 没有 lock 也没有 session. 直发 StopMove 兜底退出.
        try:
            get_loco_client().StopMove()
        except Exception:
            logger.exception("StopMove failed in stop() (no prior session)")
        return "<observe>no active move; stopped anyway</observe>"

    async with _version_lock:
        old = _current_session
        _current_version = 0
        if old is not None and not old.done.is_set():
            old.reason = "stopped"

    # 直接 StopMove —— 不靠老 session finally (后者被 version 校验挡掉).
    try:
        get_loco_client().StopMove()
    except Exception:
        logger.exception("StopMove failed in stop()")

    if old is not None and not old.done.is_set():
        try:
            await asyncio.wait_for(old.done.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            logger.warning("stop: session %s did not finish in 0.5s", old.cmd_name)
        return f"<observe>stopped {old.cmd_name}</observe>"
    return "<observe>no active move; stopped anyway</observe>"


# ── 内部: version 化的底层 wrapper ───────────────────────────────────────
# 这一层是 version 互斥模型的核心防御 —— 老 session 在 break 之前或 finally
# 内若仍想 publish, 会被静默挡掉. 不依赖"老 loop 立刻退出"的精确时序.

def _loco_move_versioned(version: int, vx: float, vy: float, vyaw: float) -> bool:
    """带 version 校验的 Move. 不匹配则 no-op 返回 False. 跨 asyncio 调用安全."""
    if version != _current_version:
        return False
    try:
        get_loco_client().Move(vx, vy, vyaw)
        return True
    except Exception:
        logger.exception("LocoClient.Move(%.2f, %.2f, %.2f) raised", vx, vy, vyaw)
        return False


def _loco_stop_versioned(version: int) -> bool:
    """带 version 校验的 StopMove. 不匹配则 no-op (说明已被抢占, 让位).

    显式 stop() 不走这里 —— 它直接调 get_loco_client().StopMove() 不经校验.
    """
    if version != _current_version:
        return False
    try:
        get_loco_client().StopMove()
        return True
    except Exception:
        logger.exception("LocoClient.StopMove raised")
        return False


# ── 内部: 通用 session 执行体 ────────────────────────────────────────────

async def _run_session(
    cmd_name: str,
    duration: float,
    *,
    vx: float,
    vy: float,
    vyaw: float,
) -> str:
    """所有 walk/turn 命令收敛到这里. 一处实现, 处处一致.

    流程:
      1. 申领新 version + 设置 _current_session (lock 内, 标记老 session 被抢占)
      2. publish loop (20Hz): 每 tick 校验 version + 检查 duration + Move
      3. finally: _loco_stop_versioned(my_version) + session.done.set()
         (被抢占则 no-op, 让位给新 session; duration 正常退则发 StopMove 回 stand)
    """
    global _current_session, _current_version, _next_version, _version_lock

    if not _running:
        raise RuntimeError(
            "locomotion not started; call ghoshell_moss_contrib.unitree.g1.runtime."
            "locomotion.start() first."
        )
    if duration <= 0:
        return f"<observe>{cmd_name} skipped: duration must be > 0</observe>"

    # asyncio.Lock 延迟创建 — 绑当前 loop. 单次 if-set 即可, 后续命令复用.
    if _version_lock is None:
        _version_lock = asyncio.Lock()

    # ── 阶段 1: 申领 version + 切换 session ────────────────────────────
    async with _version_lock:
        my_version = _next_version
        _next_version += 1
        old = _current_session
        if old is not None and not old.done.is_set():
            old.reason = f"preempted_by:{cmd_name}"
            # 老 session 不需要 await — 它下一 tick (≤ 50ms) 发现 version 失效自退.
            # finally 内的 _loco_stop_versioned 会被新 version 挡掉, 不抽搐.
        session = _MoveSession(
            version=my_version,
            cmd_name=cmd_name,
            started_at=time.monotonic(),
            done=asyncio.Event(),
        )
        _current_session = session
        _current_version = my_version

    logger.info(
        "locomotion: %s started (v=%d, vx=%.2f, vy=%.2f, vyaw=%.2f, duration=%.2fs)",
        cmd_name, my_version, vx, vy, vyaw, duration,
    )

    elapsed = 0.0
    try:
        # 首发 Move (无需 version 校验 — 我们刚拿的 version 必是 current)
        _loco_move_versioned(my_version, vx, vy, vyaw)

        # ── 阶段 2: publish loop ────────────────────────────────────
        while True:
            # version 校验是 publish loop 的核心终止条件之一
            if my_version != _current_version:
                if session.reason is None:
                    session.reason = "preempted"
                break
            elapsed = time.monotonic() - session.started_at
            if elapsed >= duration:
                if session.reason is None:
                    session.reason = "duration"
                break
            # keepalive Move — 实测确认 LocoClient.Move "一次保持" 后可改为仅首发.
            if not _loco_move_versioned(my_version, vx, vy, vyaw):
                # 被抢占了, 下次 while 头部 version 校验会再发现一次
                if session.reason is None:
                    session.reason = "preempted"
                break
            await asyncio.sleep(_TICK_SEC)
    except asyncio.CancelledError:
        session.reason = "cancelled"
        raise
    except Exception as e:
        logger.exception("locomotion %s failed", cmd_name)
        session.reason = f"exception:{type(e).__name__}"
        elapsed = time.monotonic() - session.started_at
        return f"<observe>{cmd_name} failed: {e}</observe>"
    finally:
        # ── 阶段 3: finally StopMove + done ──────────────────────────
        # 仅当我仍是 current 时才 StopMove. 被抢占的 session 让位, 不抽搐 G1.
        _loco_stop_versioned(my_version)
        session.done.set()
        # 清 _current_session 仅当还指向我 (防抢占覆盖后又被我清)
        if _current_session is session:
            with _module_lock:
                if _current_session is session:
                    _current_session = None

    return f"<observe>{cmd_name} {session.reason} after {elapsed:.2f}s</observe>"
