# IoC Provider 声明 — 声明"这个接口由这个工厂生产"。
# 在此文件中定义 Provider 实例，Matrix 启动时自动发现并注入 IoC 容器。
#
# 显式继承全局 providers，然后追加 mode 专属的：
from MOSS.manifests.providers import *  # noqa: F403
# mode 专属追加:
# my_provider = MyProvider()
#
# 参考: moss manifests providers
