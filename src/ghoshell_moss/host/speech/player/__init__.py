# player 包的轻量 re-export — 只导出不拉重依赖的成员.
# 重量实现 (MiniAudioStreamPlayer) 走显式子模块 import, 保证 import 本包不依赖音频库.
from ghoshell_moss.core.speech.base_player import BaseAudioStreamPlayer
from ghoshell_moss.core.speech.virtual_player import VirtualStreamPlayer

__all__ = ["BaseAudioStreamPlayer", "VirtualStreamPlayer"]
