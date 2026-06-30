"""
G1 Body App — 进程入口。

由 Circus 管理生命周期。构造 DDS 连接 → 创建 SDK clients → 构建 channel → Matrix 注册。
连接失败抛异常 → 进程退出 → Circus 重启。
"""


print("todo: not implemented")