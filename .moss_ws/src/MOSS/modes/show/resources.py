# 资源存储声明 — 声明可寻址的资源数据集。
# 在此文件中定义 ResourceStorageMeta 实例，Matrix 启动时自动发现。
#
# 显式继承全局 resources，然后追加 mode 专属的：
from MOSS.manifests.resources import *  # noqa: F403
