# Default mode 的 main channel — 从零构建标准 shell main channel。
# 如需复用全局 main，改为 from MOSS.manifests.channels import main。

from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.core.ctml.shell.ctml_main import inject_system_primitives

from ghoshell_moss.channels.app_store_channel import AppStoreChannel

from ghoshell_moss.core.speech import SpeechChannelModule

main = new_shell_main_channel()

# -- 系统原语 --------------------------------------------------
inject_system_primitives(main, extended=True)

# -- Speech --------------------------------------------------
main.with_module(SpeechChannelModule())

# -- app store ---------------------------------------------
main.import_channels(AppStoreChannel())
