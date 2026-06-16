"""
Unitree G1 全局状态管理 — module 级单例。

所有 SDK client 和 DDS 状态缓存在此模块中管理。
bootstrap() 是唯一的显式初始化入口，幂等且线程安全。

约定:
  - macOS 上无法 bootstrap (无 cyclonedds) — bootstrap() 会抛 ImportError
  - 模块级状态由后台 _monitor 线程 (20Hz) 维护, state.py 提供 O(1) 读取
  - SDK 路径由 UNITREE_G1_SDK_PATH 环境变量指定，import 时校验，无默认值
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# 模块级: SDK 路径 — import 时即校验，不依赖 bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

_sdk_path = os.environ.get("UNITREE_G1_SDK_PATH", "")
if not _sdk_path:
    raise RuntimeError(
        "UNITREE_G1_SDK_PATH 环境变量未设置。"
        "请设置为 unitree_sdk2_python 的本地路径，"
        "例如: export UNITREE_G1_SDK_PATH="
        "/home/moss/MOSShell/.moss_ws/apps/bodies/g1/src/unitree_sdk2_python"
    )
if not os.path.isdir(_sdk_path):
    raise RuntimeError(
        f"UNITREE_G1_SDK_PATH 指向的路径不存在: {_sdk_path}"
    )

sys.path.insert(0, _sdk_path)


def _check_sdk() -> None:
    """验证 SDK 关键模块可 import。sys.path 已在模块级设置。"""
    try:
        import unitree_sdk2py.core.channel  # noqa: F401
        import unitree_sdk2py.g1.audio.g1_audio_client  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"unitree_sdk2_python 无法导入 (路径: {_sdk_path})。"
            "请确认已 clone SDK 并安装依赖。"
            "详见 .moss_ws/apps/bodies/g1/README.md"
        ) from e


# -- 模块级状态 ----------------------------------------------------------------

_initialized: bool = False
_init_lock = threading.Lock()

# DDS clients (全局单例 — 构造即连接)
_audio_client = None       # g1.audio.g1_audio_client.AudioClient
_loco_client = None        # g1.loco.g1_loco_client.LocoClient
_arm_client = None         # g1.arm.g1_arm_action_client.G1ArmActionClient

_network_interface: str = ""


def bootstrap(nic: str) -> None:
    """初始化 DDS + AudioClient + 状态监控线程。幂等，线程安全。

    首次调用:
      1. 初始化 DDS domain (全局单例)
      2. 创建 AudioClient (音频播放用)
      3. 启动后台状态监控线程 (20Hz LowState + 2Hz 电池/主板)
    后续调用直接返回。
    """
    global _initialized, _network_interface, _audio_client

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        _check_sdk()

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        from ghoshell_moss_contrib.unitree.g1._monitor import start_monitor

        ChannelFactoryInitialize(0, nic)

        _audio_client = AudioClient()
        _audio_client.SetTimeout(10.0)
        _audio_client.Init()

        # 启动 DDS 状态监控 — motion/joints/imu/remote 20Hz, battery/health 2Hz
        start_monitor(nic)

        _network_interface = nic
        _initialized = True


def is_initialized() -> bool:
    return _initialized


def get_audio_client():
    """获取全局 AudioClient 单例。未 bootstrap 时返回 None。"""
    return _audio_client


def get_network_interface() -> str:
    return _network_interface