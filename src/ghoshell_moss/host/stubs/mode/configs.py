# 配置模型声明 — 声明配置的 schema 和默认值。
# 在此文件中定义 ConfigType 子类，Matrix 启动时自动发现。
#
# 显式继承全局 configs，然后追加 mode 专属的：
from MOSS.manifests.configs import *  # noqa: F403
