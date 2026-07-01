# Mode 的 main channel。
# Mode 的 __main__ channel 是唯一的生效 main channel。

from ghoshell_moss import new_default_shell_main_channel

# ── G1 SDK bootstrap ────────────────────────────────────────────────────────
from ghoshell_moss_contrib.unitree.g1 import sdk

sdk.bootstrap()

main = new_default_shell_main_channel()

# ── G1 body channels 拓扑组装 ─────────────────────────────────────────────
from ghoshell_moss_contrib.unitree.g1.channels.g1_root import g1_root
from ghoshell_moss_contrib.unitree.g1.channels.fsm import g1_fsm
from ghoshell_moss_contrib.unitree.g1.channels.face_led import face_led
from ghoshell_moss_contrib.unitree.g1.channels.listener import listener_channel
from ghoshell_moss_contrib.unitree.g1.channels.asr import g1_asr
from ghoshell_moss_contrib.unitree.g1.channels.locomotion import locomotion_channel

g1_root.import_channels(g1_fsm)
g1_root.import_channels(face_led)
g1_root.import_channels(listener_channel)
g1_root.import_channels(g1_asr)
g1_root.import_channels(locomotion_channel)
main.import_channels(g1_root)
