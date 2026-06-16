"""
Unitree G1 全局状态管理 — module 级单例。

所有 SDK client 和 DDS 状态缓存在此模块中管理。
bootstrap() 是唯一的显式初始化入口，幂等且线程安全。

约定:
  - macOS 上无法 bootstrap (无 cyclonedds) — bootstrap() 会抛 ImportError
  - 模块级状态由后台 _monitor 线程 (20Hz) 维护, state.py 提供 O(1) 读取
  - SDK 路径: 各模块在 import SDK 前显式调用 _sdk.setup_sdk_path() (幂等)
"""

from __future__ import annotations
from ghoshell_moss_contrib.unitree.g1._sdk import load_unitree_g1_sdk, unitree_nic

load_unitree_g1_sdk()

import threading

# -- 模块级状态 ----------------------------------------------------------------

_initialized: bool = False
_init_lock = threading.Lock()

# DDS clients (全局单例 — 构造即连接)
_audio_client = None  # g1.audio.g1_audio_client.AudioClient
_loco_client = None  # g1.loco.g1_loco_client.LocoClient
_arm_client = None  # g1.arm.g1_arm_action_client.G1ArmActionClient

_network_interface: str = ""


def bootstrap() -> None:
    """初始化 DDS domain + AudioClient 单例。幂等，线程安全。

    调用方负责先执行 _sdk.setup_sdk_path() 确保 SDK 在 sys.path 上。
    不启动状态监控线程 — 由调用方按需调用 _monitor.start_monitor()。
    """
    global _initialized, _network_interface, _audio_client
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        nic = unitree_nic()

        ChannelFactoryInitialize(0, nic)

        _audio_client = AudioClient()
        _audio_client.SetTimeout(10.0)
        _audio_client.Init()

        _network_interface = nic
        _initialized = True


def is_initialized() -> bool:
    bootstrap()
    return _initialized


def get_audio_client():
    """获取全局 AudioClient 单例。未 bootstrap 时返回 None。"""
    bootstrap()
    return _audio_client


def get_network_interface() -> str:
    bootstrap()
    return _network_interface
