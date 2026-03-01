from ghoshell_moss.core.ctml.elements import *
from ghoshell_moss.core.ctml.interpreter import *
from ghoshell_moss.core.ctml.prompt import get_moss_meta_prompt
from ghoshell_moss.core.ctml.shell import create_ctml_main_chan, new_ctml_shell, CTMLShell

system_prompt = get_moss_meta_prompt()
