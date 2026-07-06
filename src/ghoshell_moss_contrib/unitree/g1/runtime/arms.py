"""
Arms runtime — G1 手臂动作 (L2 命名调用形态) — L2 能力金字塔的第一层.

设计范式:
  - **L2 命名调用**: LLM 只看动作名 (wave / clap / heart), 不接触关节坐标.
    动作实现走 G1 主板 ExecuteAction(id) RPC. 见 FEATURE.md "arms 能力金字塔".
  - **不做 preempt/version**: ExecuteAction 是 RPC 阻塞 (script 21 实测互斥 +
    不可中断), 单一 _busy flag 拒绝新命令. 忙时返回 "busy, ignored".
  - **不做完成信号**: script 28 (action state probe) 未跑, 无 done callback.
    用保守 sleep 估时 + 自动 release_arm 复位.
  - **中断三基础缺失**: 碰撞反馈 / 复位可靠性 / 首帧过渡都未达成. 本模块只做
    "发出去 + 等一段时间 + 复位" 的最简形态, 高级形态见 FEATURE.md.

物理事实 (来自 scripts/sdk/09_arm_preset.py + 21_arm_action_interruption.py):
  - G1ArmActionClient.ExecuteAction(id) 是同步 RPC. RPC 卸载到线程池.
  - action id map: 见 _ACTION_MAP. 99 = release_arm (温柔复位).
  - A 中发 B → 7401 / 3104 拒绝 (script 21). ExecuteAction 99 在 arm 忙时
    code=0 但排队, 不中断当前动作.
  - 每个动作物理时长约 3-5s (估值, 未逐个标定).

安全:
  - **必须 FSM STAND 或 WALK_RUN + L2 授权** (channel 层门控).
  - 首次实测建议吊架状态, 站立. arm 动作幅度大, 手臂前方需保持 1m 空间.

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import arms

    bootstrap(nic="eth0")
    arms.start()
    result = await arms.wave()          # "<observe>wave started, ~4s + release</observe>"
    # arm 忙时:
    result = await arms.clap()          # "<observe>arm busy (wave), ignored</observe>"
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from ghoshell_moss_contrib.unitree.g1.sdk import get_arm_client

logger = logging.getLogger("moss.g1.runtime.arms")


# ── 常量: 动作 ID map + 保守时长估算 ────────────────────────────────────
# id 来自 SDK unitree_sdk2py.g1.arm.g1_arm_action_client.action_map.
# duration 秒是 09 脚本实测经验值 + 保守放大, 未逐个标定.

_ACTION_RELEASE = 99   # 温柔复位帧, 空闲时可用作复位

_ACTION_MAP: dict[str, tuple[int, float]] = {
    # name       (action_id, duration_seconds)
    "wave":      (26, 4.0),   # high wave — 大幅挥手, 招呼场景
    "clap":      (17, 3.0),   # 鼓掌
    "heart":     (20, 4.0),   # 比心
    "release":   (99, 2.5),   # 手动复位 (显式命令, 让 LLM 可主动收手)
}


# ── 模块级私有状态 ───────────────────────────────────────────────────
# 单例服务, 所有可变状态在模块级, _busy 用 asyncio.Lock 保护.

_running: bool = False
_busy_lock: Optional[asyncio.Lock] = None
_current_action: Optional[str] = None    # 忙时的动作名, 供 health / 报错


# ── 公开接口 ─────────────────────────────────────────────────────────

def start() -> None:
    """启动. 幂等. 不建线程 (RPC 走 asyncio.to_thread), 只标 running."""
    global _running
    if _running:
        return
    _running = True
    logger.info("arms: started (L2 命名调用形态)")


def stop_runtime(timeout: float = 2.0) -> None:
    """停止. 幂等. 无子线程, 只标 not-running.

    命名避开命令面 stop() (虽然 arms 没 stop 命令, 保持一致).
    """
    global _running
    if not _running:
        return
    _running = False
    # 兜底 release — 如果正忙, 让 arm 回到温柔状态. 不 await.
    try:
        get_arm_client().ExecuteAction(_ACTION_RELEASE)
    except Exception:
        logger.exception("arms.stop_runtime: release_arm 兜底调用异常 (isolated)")
    logger.info("arms: stopped")


def is_running() -> bool:
    return _running


def current_action() -> Optional[str]:
    """忙时返回动作名, 空闲返回 None. 供 channel debug / context."""
    return _current_action


def health() -> dict:
    return {
        "running": _running,
        "busy": _current_action is not None,
        "current_action": _current_action,
        "available_actions": list(_ACTION_MAP.keys()),
    }


async def _execute_action(name: str) -> str:
    """通用执行体. 所有命名命令收敛到这里.

    流程:
      1. 检查 _running
      2. 检查动作名合法
      3. 尝试拿 _busy_lock (非阻塞) — 拿不到说明其他动作在跑, 拒绝
      4. 记录 _current_action + ExecuteAction RPC (卸载线程池)
      5. sleep 估算时长
      6. 自动 release_arm (release 命令自己不再嵌套 release)
      7. 清 _current_action
    """
    global _busy_lock, _current_action

    if not _running:
        return "<observe>arms not started</observe>"

    if name not in _ACTION_MAP:
        return f"<observe>unknown action '{name}'. valid: {list(_ACTION_MAP.keys())}</observe>"

    # asyncio.Lock 延迟创建 — 绑当前 loop
    if _busy_lock is None:
        _busy_lock = asyncio.Lock()

    if _busy_lock.locked():
        return f"<observe>arm busy ({_current_action}), '{name}' ignored</observe>"

    action_id, duration = _ACTION_MAP[name]

    async with _busy_lock:
        _current_action = name
        started_at = time.monotonic()
        logger.info("arms: %s started (id=%d, est %.1fs)", name, action_id, duration)

        try:
            code = await asyncio.to_thread(
                get_arm_client().ExecuteAction, action_id,
            )
            if code != 0:
                _current_action = None
                logger.warning("arms: %s ExecuteAction code=%d", name, code)
                return f"<observe>{name} rejected by G1 (code={code}). arm 可能忙, 或 FSM 不允许</observe>"

            # 等待动作物理完成 (保守估时)
            await asyncio.sleep(duration)

            # 自动 release (release 命令自己不再嵌套)
            if name != "release":
                logger.info("arms: %s done, auto release_arm", name)
                await asyncio.to_thread(get_arm_client().ExecuteAction, _ACTION_RELEASE)
                await asyncio.sleep(2.0)  # 保守: release 恢复时长

            elapsed = time.monotonic() - started_at
            return f"<observe>{name} completed after {elapsed:.1f}s (含自动复位)</observe>"

        except Exception as e:
            logger.exception("arms: %s exception", name)
            return f"<observe>{name} failed: {e}</observe>"
        finally:
            _current_action = None


# ── 命名命令 (LLM 直接调) ───────────────────────────────────────────

async def wave() -> str:
    """挥手打招呼. 大幅高位挥手, ~4s + 自动复位.

    :return: Observe 文本 ("<observe>wave completed after ... </observe>").
    """
    return await _execute_action("wave")


async def clap() -> str:
    """鼓掌. ~3s + 自动复位."""
    return await _execute_action("clap")


async def heart() -> str:
    """比心. 双手在胸前比心, ~4s + 自动复位."""
    return await _execute_action("heart")


async def release() -> str:
    """手动温柔复位. arm 显式收回自然位.

    通常动作跑完自动 release, 这个命令是模型主动"收手"的入口.
    """
    return await _execute_action("release")
