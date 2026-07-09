# 感知核声明 — 继承全局 + audio_nucleus（处理 listener 的 SPEECH_FINAL 信号）。
from MOSS.manifests.nuclei import *  # noqa: F403

from ghoshell_moss.core.mindflow.audio_nucleus import AudioNucleusMeta

audio_nucleus_factory = AudioNucleusMeta()
