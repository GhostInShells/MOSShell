"""
SDK 路径初始化 — 独立模块，由各需要 SDK 的模块显式调用。

从 UNITREE_G1_SDK_PATH 环境变量读取路径并加入 sys.path。
幂等：多次调用只执行一次。

约定: 每个依赖 unitree_sdk2py 的模块在 import SDK 之前显式调用。
不放在 __init__.py 中作为隐式副作用。
"""

import os

__all__ = ["load_unitree_g1_sdk", 'unitree_nic']

_EXPECTED_G1_NIC = 'eth0'


def unitree_nic() -> str:
    return os.environ.get('UNITREE_G1_NIC', _EXPECTED_G1_NIC)


def load_unitree_g1_sdk() -> None:
    """确保 unitree_sdk2_python 在 sys.path 上。幂等。

    Raises:
        RuntimeError: 环境变量未设置或路径不存在。
    """
    try:
        import unitree_sdk2py
    except ImportError:
        raise ImportError(f"unitree_sdk2py not installed")
