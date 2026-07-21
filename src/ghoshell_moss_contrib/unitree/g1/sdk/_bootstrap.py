"""
G1 bootstrap — 初始化生命周期 + clients 单例 + 现场调试.

设计原则:
  - 显式 init, 不自动 init.
  - 所有 get_*_client() 在 bootstrap 未完成时 raise — 不返回 None.
  - 幂等(重复调不报错, 直接返回).
  - 失败任何子步骤 → raise + 不留半初始化状态.
  - dump_state() 现场调试.

上一版踩的坑(本版必须避免):
  1. _loco_client 等字段声明但 bootstrap() 没初始化 → 调用方拿到 None
     -> 本版: bootstrap 必须 init 全部 client. 缺一不可.
  2. main.py 自己 init 一次 client, bootstrap 也 init 一次 → 两份 client.
     -> 本版: clients 仅由 bootstrap 持有, main.py 通过 get_*_client 取.
  3. get_audio_client 内部隐式 bootstrap(); return client → "忘了 bootstrap"被掩盖.
     -> 本版: get_*_client 不自动 bootstrap, 未 bootstrap raise.

bootstrap 顺序:
  1. import SDK (上游 _sdk 已校验路径)
  2. ChannelFactoryInitialize(0, nic)
  3. 创建 AudioClient + Init() + SetTimeout
  4. 创建 LocoClient + Init() + SetTimeout
  5. 创建 G1ArmActionClient + Init() + SetTimeout
  6. start_monitor() — monitor 注册 cyclonedds callback
  7. 等首帧 LowState (wait_first_frame=True 时)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from . import _monitor
from . import state
from ._sdk import unitree_nic, load_unitree_g1_sdk

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级状态
# ═══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_bootstrapped: bool = False
_network_interface: str = ""

# Clients — bootstrap 完成后非 None.
_audio_client: Any = None
_loco_client: Any = None
_arm_client: Any = None


# ═══════════════════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════════════════


def bootstrap(
    nic: str | None = None,
    *,
    wait_first_frame: bool = True,
    timeout: float = 5.0,
) -> None:
    """初始化 G1 DDS 连接 + 三个 client + monitor.

    Args:
        nic: 网卡名. None 则走 env UNITREE_G1_NIC (默认 'eth0').
        wait_first_frame: 阻塞到 monitor 收到第一帧 LowState. 默认 True —
                          确保后续 motion()/remote() 等返回真值.
        timeout: wait_first_frame 的超时.

    Raises:
        ImportError: SDK 未安装.
        RuntimeError: bootstrap 任何步骤失败, 或 wait_first_frame 超时.
                      失败时清理半初始化状态.

    幂等. 重复调直接返回, 不抛.
    """
    global _bootstrapped, _network_interface
    global _audio_client, _loco_client, _arm_client

    with _lock:
        if _bootstrapped:
            return

        # 1. SDK 路径
        load_unitree_g1_sdk()  # raise ImportError if missing
        nic_to_use = nic if nic is not None else unitree_nic()
        logger.info("bootstrap: starting, nic=%s", nic_to_use)

        try:
            # 2. ChannelFactory
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, nic_to_use)
            logger.info("bootstrap: ChannelFactory initialized")

            # 3. AudioClient
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            audio = AudioClient()
            audio.SetTimeout(10.0)
            audio.Init()
            logger.info("bootstrap: AudioClient initialized")

            # 4. LocoClient
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            loco = LocoClient()
            loco.SetTimeout(10.0)
            loco.Init()
            logger.info("bootstrap: LocoClient initialized")

            # 5. ArmActionClient
            from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
            arm = G1ArmActionClient()
            arm.SetTimeout(10.0)
            arm.Init()
            logger.info("bootstrap: G1ArmActionClient initialized")

            # 6. monitor (这一步注册 cyclonedds callback, 不阻塞等数据)
            _monitor.start_monitor()
            logger.info("bootstrap: monitor started")

            # 提交到模块级
            _audio_client = audio
            _loco_client = loco
            _arm_client = arm
            _network_interface = nic_to_use
            _bootstrapped = True

        except Exception:
            # 回滚: 清理 monitor (如果起来了)
            try:
                _monitor.stop_monitor()
            except Exception:
                logger.exception("bootstrap: failed to clean monitor after error")
            _audio_client = None
            _loco_client = None
            _arm_client = None
            _network_interface = ""
            _bootstrapped = False
            logger.exception("bootstrap: failed")
            raise

        logger.info("bootstrap: clients + monitor ready")

    # 7. 等首帧 — 不持锁(避免阻塞其他线程检查状态)
    if wait_first_frame:
        _wait_first_lowstate(timeout)


def _wait_first_lowstate(timeout: float) -> None:
    """阻塞到 state.motion() 不再 raise. 超时 raise RuntimeError."""
    t_start = time.monotonic()
    while time.monotonic() - t_start < timeout:
        try:
            state.motion()
            elapsed = time.monotonic() - t_start
            logger.info("bootstrap: first LowState received (elapsed=%.2fs)", elapsed)
            return
        except RuntimeError:
            time.sleep(0.05)

    # 超时 — 这是个严重问题, raise. 但不撤销 bootstrap (monitor 继续跑, 用户可以重试).
    raise RuntimeError(
        f"bootstrap: timeout {timeout}s waiting for first LowState. "
        f"check: G1 powered, DDS reachable (ufw IP fragments?), nic={_network_interface}"
    )


def is_bootstrapped() -> bool:
    """不 raise, 用于现场调试."""
    return _bootstrapped


def get_audio_client() -> Any:
    """获取 AudioClient. 未 bootstrap raise."""
    if not _bootstrapped or _audio_client is None:
        raise RuntimeError("g1 not bootstrapped; call bootstrap() first")
    return _audio_client


def get_loco_client() -> Any:
    """获取 LocoClient. 未 bootstrap raise."""
    if not _bootstrapped or _loco_client is None:
        raise RuntimeError("g1 not bootstrapped; call bootstrap() first")
    return _loco_client


def get_arm_client() -> Any:
    """获取 G1ArmActionClient. 未 bootstrap raise."""
    if not _bootstrapped or _arm_client is None:
        raise RuntimeError("g1 not bootstrapped; call bootstrap() first")
    return _arm_client


def get_fsm_id() -> int:
    """当前运控模式 ID. RPC 7001.
    返回值: 0=ZeroTorque 1=Damp 3=Sit 4=Stand 500=Regular 801=WalkRun.
    """
    import json
    if not _bootstrapped or _loco_client is None:
        raise RuntimeError("g1 not bootstrapped; call bootstrap() first")
    code, data = _loco_client._Call(7001, "{}")
    if code != 0:
        raise RuntimeError(f"GetFsmId RPC failed: code={code}")
    return json.loads(data)["data"]


def get_fsm_mode() -> int:
    """运动状态. RPC 7002. 0=站立态(可切换模式) 1=移动态(不可切)."""
    import json
    if not _bootstrapped or _loco_client is None:
        raise RuntimeError("g1 not bootstrapped; call bootstrap() first")
    code, data = _loco_client._Call(7002, "{}")
    if code != 0:
        raise RuntimeError(f"GetFsmMode RPC failed: code={code}")
    return json.loads(data)["data"]


def get_network_interface() -> str:
    """已使用的网卡名. 未 bootstrap 时返回空字符串."""
    return _network_interface


def dump_state() -> dict:
    """现场调试: 一次性返回所有状态. PC2 上出问题时打印此字典看全貌."""
    return {
        'bootstrapped': _bootstrapped,
        'network_interface': _network_interface,
        'clients': {
            'audio': _audio_client is not None,
            'loco': _loco_client is not None,
            'arm': _arm_client is not None,
        },
        'state_module_started': state.is_started(),
        'state_last_update_age_sec': (
            time.monotonic() - state.last_update() if state.last_update() > 0 else None
        ),
        'monitor_health': _monitor.get_health(),
    }


def _reset_for_testing() -> None:
    """测试 hook: 把 bootstrap 状态归位. 不暴露到 __init__."""
    global _bootstrapped, _network_interface
    global _audio_client, _loco_client, _arm_client

    try:
        _monitor.stop_monitor()
    except Exception:
        pass
    state._reset_all_for_testing()

    _bootstrapped = False
    _network_interface = ""
    _audio_client = None
    _loco_client = None
    _arm_client = None
