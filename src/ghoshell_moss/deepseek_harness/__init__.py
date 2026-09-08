"""deepseek_harness — dsh 连接的协议层与进程层.

供 dsh 融合的两条落点路径 (gui 管理的 agent / dolores ghost) 消费.
对外导出的最小面在此, 具体类型从子模块再导出见各模块 __all__.
"""

from ghoshell_moss.deepseek_harness.launcher import (
    DshConnection,
    DshConnectionConfig,
    DshExit,
    DshLauncher,
    DshLauncherConfig,
)

__all__ = [
    "DshConnection",
    "DshConnectionConfig",
    "DshLauncher",
    "DshLauncherConfig",
    "DshExit",
]
