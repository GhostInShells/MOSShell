# MOSS Main Channel manifest.
#
# FastAPI-like entry point: 创建 CTML shell 的主 channel (__main__)。
# 在这里可以 import_channels（树挂载）、with_module（能力叠加）、with_state（排他切换）。
# 所有组合显式可见，零隐式逻辑。
#
# 如果这个文件没有定义 name == "__main__" 的 channel，MossRuntime 会使用默认空白 main。
#
# Mode 复用全局 main 的最简方式：
#     from MOSS.manifests.channels import main
#     然后继续 inject_system_primitives(main, extended=True) 或 import_channels 等。

from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.core.ctml.shell.ctml_main import inject_system_primitives
from ghoshell_moss.core.speech import SpeechChannelModule
from ghoshell_moss.channels.app_store_channel import AppStoreChannel

main = new_shell_main_channel(description="Default MOSS main channel with app store")

# -- 系统原语 --------------------------------------------------
inject_system_primitives(main)

# -- App Store ---------------------------------------------------
main.import_channels(AppStoreChannel(name='apps'))

# -- Speech --------------------------------------------------
main.with_module(SpeechChannelModule())