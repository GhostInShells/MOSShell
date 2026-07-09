# Aether mode 的 main channel — 复用 default 的 speech + apps + terminal。
from ghoshell_moss import new_default_shell_main_channel
from ghoshell_moss.channels.app_store_channel import AppStoreChannel
from ghoshell_moss.channels.terminal_channel import new_terminal_channel
from ghoshell_moss.core.speech import SpeechChannelModule

main = new_default_shell_main_channel()

# Speech channel（TTS 能力）—— voice demo 要把普通短句也送进 TTS。
# 不依赖模型稳定生成 <say>，否则 prompt 收紧为 plain text 后会只显示不播放。
main.with_module(SpeechChannelModule(register_content=True))

# app store + terminal
main.import_channels(
    AppStoreChannel(name='apps'),
    new_terminal_channel(name='bash'),
)
