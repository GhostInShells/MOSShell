# Mode Provider manifest — mode-specific IoC provider declarations.
#
# Define Provider instances here to extend or override global providers.
# Matrix scans both global (MOSS.manifests.providers) and mode (HOST.providers),
# combining them at runtime.
#
# --
# Mode Provider 清单 — mode 专属 IoC 声明。
# 在此定义 Provider 实例以扩展或覆盖全局 providers。

from ghoshell_moss.host.providers.tts_service_provider import TTSServiceProvider
from ghoshell_moss.host.providers.speech_service_provider import TTSSpeechServiceProvider
from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerProvider
from ghoshell_moss.host.providers.audio_capture_provider import AudioCaptureProvider

# audio capture
audio_capture_provider = AudioCaptureProvider()

# audio player
player_service_provider = AudioPlayerProvider()

# text-to-speech
tts_service_provider = TTSServiceProvider()

# speech service
speech_service_provider = TTSSpeechServiceProvider()
