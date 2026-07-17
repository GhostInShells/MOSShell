"""MOSS node cell entry point.

Start:  moss nodes run <path-to-this-dir>    # via CLI (foreground, CLI is owner)
Debug:  python main.py                        # ad-hoc launch (from_proc identity)

Explore:
    moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest
    moss codex blueprint channel_builder
    moss codex blueprint matrix
    moss ctml read
"""

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    # A node without a membrane (channel) does not exist in the model's world.
    # Build the channel and provide it — the Python signatures ARE the prompt.
    #
    #   from ghoshell_moss.core.blueprint.channel_builder import new_channel
    #
    #   channel = new_channel(name="my_node", description="what this node does")
    #
    #   @channel.build.command()
    #   async def ping() -> str:
    #       """One-line description shown to the model."""
    #       return "pong"
    #
    #   await matrix.provide_channel(channel)   # blocks until membrane closes
    pass


if __name__ == "__main__":
    Matrix.discover().run(main)
