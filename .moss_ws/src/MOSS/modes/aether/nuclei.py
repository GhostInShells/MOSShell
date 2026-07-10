# 感知核声明 — 继承全局 + audio_nucleus（处理 listener 的 SPEECH_FINAL 信号）。
from MOSS.manifests.nuclei import *  # noqa: F403

from ghoshell_moss.core.mindflow.audio_nucleus import AudioNucleusMeta

# aEther 是全双工语音模式：普通 ASR final 表示“用户说完一句话”，不等同于
# “立刻清空 shell/TTS”。真正的急停由 listener 的 wake word 走
# InterruptNucleus 和 AudioRuntimeTopic(device_name="interrupt") 两条路径。
audio_nucleus_factory = AudioNucleusMeta(interrupt_on_complete=False)
