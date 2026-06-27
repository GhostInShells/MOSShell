# 感知核声明 — Mindflow 输入信号源的声明。
# 在此文件中定义 NucleusMeta 实例，Matrix 启动时自动发现。
#
# 显式继承全局 nuclei，然后追加 mode 专属的：
from MOSS.manifests.nuclei import *  # noqa: F403

from ghoshell_moss.core.mindflow.audio_nucleus import AudioNucleusMeta

audio_nucleus_factory = AudioNucleusMeta()
