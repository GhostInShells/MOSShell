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

# 本自研协议客户端对齐的 dsh 版本. dsh 是开发者预览, serverInfo.version 恒 0.0.1
# (不承诺接口稳定); 追版本时改此常量 + diff interface 变动 (见 research_2026-08-30).
DSH_VERSION = "0.1.1-rc.2"

__all__ = [
    "DshConnection",
    "DshConnectionConfig",
    "DshLauncher",
    "DshLauncherConfig",
    "DshExit",
    "DSH_VERSION",
]
