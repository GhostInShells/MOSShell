from ghoshell_moss.core.blueprint.channel_builder import new_command
from ghoshell_moss.core.ctml.shell.primitives import (
    loop,
    sample,
    branch,
)

loop_primitive = new_command(loop)
sample_primitive = new_command(sample)
branch_primitive = new_command(branch)
