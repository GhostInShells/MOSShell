"""MOSS node entry point.

Run:    moss nodes run <path-to-this-dir>    # foreground, this CLI owns the node
Debug:  python main.py                       # ad-hoc launch (from_proc identity)
"""

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    # A node with no channel does not exist in the Ghost's world: the channel IS
    # its capability surface, and the Python signatures are the prompt.
    #
    #   from ghoshell_moss.core.blueprint.channel_builder import new_channel
    #
    #   channel = new_channel(name="my_node", description="what this node does")
    #
    #   @channel.build.command()
    #   async def ping() -> str:
    #       """One-line description shown to the Ghost."""
    #       return "pong"
    #
    #   await matrix.provide_channel(channel)   # blocks until the membrane closes
    #
    # Read the channel builder and Matrix for how a channel is built and provided;
    # see Matrix channel for how it mounts into the Ghost's shell.
    pass


if __name__ == "__main__":
    Matrix.discover().run(main)
