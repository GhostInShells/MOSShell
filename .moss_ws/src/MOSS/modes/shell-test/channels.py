# Shell channel verification mode — minimal test mode.

from ghoshell_moss import new_default_shell_main_channel
from ghoshell_moss.channels.shell_channel import new_shell_channel

main = new_default_shell_main_channel()
main.import_channels(new_shell_channel())
