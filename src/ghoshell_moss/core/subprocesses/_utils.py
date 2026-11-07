"""ProcessManager 内部工具函数 — 已迁到 :mod:`ghoshell_processes.utils`.

保留本路径只为兼容既有 import；实现与 ``ghoshell_processes.utils.killpg`` 是同一个函数
（迁移前两份实现逐字节相同，这里不再保留第二份）。
"""

from ghoshell_processes.utils import killpg

__all__ = ["killpg"]
