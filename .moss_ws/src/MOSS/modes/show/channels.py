# Mode 的 main channel。
# Mode 的 __main__ channel 是唯一的生效 main channel。
#
# 两种构建模式：
#
# 1. 从零构建（推荐 — 独立 mode）：
#    from ghoshell_moss import new_default_shell_main_channel
#    main = new_default_shell_main_channel(description="...")
#    # 在 main 上追加 command 或 compose sub-channel...
#
# 2. 复用全局 main（mode 作为全局的增量改造）：
#    from MOSS.manifests.channels import main
#    # 在 main 上追加改造...

from ghoshell_moss import new_default_shell_main_channel
from ghoshell_moss.channels.app_store_channel import AppStoreChannel
from ghoshell_moss.channels.mac_channel import new_mac_control_channel
from ghoshell_moss.channels.terminal_channel import new_terminal_channel
from ghoshell_moss.channels.mermaid_draw import new_mermaid_channel
from ghoshell_moss.channels.web_bookmark import new_web_bookmark_channel
from ghoshell_moss.core import SpeechChannelModule

main = new_default_shell_main_channel()

main.with_module(SpeechChannelModule())

main.import_channels(
    AppStoreChannel(name='apps'),
    new_mac_control_channel(name='mac'),
    new_mermaid_channel(name='mermaid'),
    # new_terminal_channel(name='bash'),
    new_web_bookmark_channel(name='web_bookmark'),
)
