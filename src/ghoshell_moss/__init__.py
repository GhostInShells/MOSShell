"""MOSShell — Model-Oriented Operating System Shell.

The facade doubles as a code-as-prompt architectural index.
Each section maps to a concept area in the topology.

Entry-point pattern (see CLI for working examples)::

    env = Environment.discover()
    host = MossHost(env=env)
    # then access host.runtime, host.ghost_runtime(name), etc.
"""

# -- IoC Container -----------------------------------------------------------
from ghoshell_container import (
    Provider,
    Bootstrapper,
    IoCContainer,
    provide,
)

# -- Core Concepts -----------------------------------------------------------
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.command import Command, Observe, ObserveError
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.core.concepts.shell import MOSShell

# -- Message Types -----------------------------------------------------------
from ghoshell_moss.message.contents.images import Base64Image
from ghoshell_moss.message.contents.text import Text
from ghoshell_moss.message.message import Addition, Message

# -- Channel Builders --------------------------------------------------------
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.states_channel import (
    # 创建 main channel
    new_shell_main_channel,
    new_default_shell_main_channel,

    new_channel_from_state,
    new_prime_channel,
    new_channel_state,
    new_stateful_channel,
)
from ghoshell_moss.core.py_channel import PyChannel

# -- CTML Shell & Interpreter ------------------------------------------------
from ghoshell_moss.core.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.core.ctml.shell.ctml_main import create_ctml_main_chan, inject_system_primitives
from ghoshell_moss.core.ctml.shell.ctml_shell import CTMLShell, new_ctml_shell
