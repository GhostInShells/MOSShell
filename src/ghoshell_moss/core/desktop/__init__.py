"""Desktop — Ghost 的文件系统认知桌面 (concrete 层).

**WIP (2026-07-12)**: 契约层已按 2026-07 重绘 (Grounds/Ground/Pin/
GroundConvention/UpdateResult), concrete 层在渐进重写. 完成后本 __init__
会 re-export 契约 + DefaultGrounds/DefaultGround.

当前进度: _addr.py (地址解析) + _hash.py (对账观察).

历史资产: ``desktop.py``, ``models.py`` 是 2026-06-29 Stage 1 的 12+1 原语
实现, 已与新契约不兼容, K26 迁移完成后清理. 期间它们保留在磁盘上但不再从
本 __init__ 导入.
"""
