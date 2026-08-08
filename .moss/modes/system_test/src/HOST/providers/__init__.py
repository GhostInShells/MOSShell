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
# 音频/语音 provider (tts/speech/player) 已迁到 project 级
# MOSS.manifests.providers (基线能力, CLI 经 Matrix.new 可见).
# 此处仅保留 mode 专属/待迁移项: audio_capture (下轮迁 project 级).

from ghoshell_moss.host.providers.audio_capture_provider import AudioCaptureProvider

# audio capture
audio_capture_provider = AudioCaptureProvider()
