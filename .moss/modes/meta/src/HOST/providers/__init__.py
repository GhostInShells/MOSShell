# Mode Provider manifest — mode-specific IoC provider declarations.
#
# Define Provider instances here to extend or override global providers.
# Matrix scans both global (MOSS.manifests.providers) and mode (HOST.providers),
# combining them at runtime.
#
# --
# Mode Provider 清单 — mode 专属 IoC 声明。
# 在此定义 Provider 实例以扩展或覆盖全局 providers。
#
# 音频/语音基线能力 (tts/speech/player/capture) 已迁移到 project 级
# MOSS.manifests.providers — mode 无需重复声明, 继承即可.
# 若需要 mode 专属覆盖, 在此定义同名 provider 实例 (后注册覆盖 baseline).
