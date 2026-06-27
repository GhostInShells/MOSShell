"""MOSS Cell entry point.

Start:  moss cells run {name}       # name mode (once CELL.md name is filled)
Debug:  python main.py                      # direct launch

For full context:  moss cells specification
"""

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    # Build your channel here and provide it to the Matrix:
    #
    #   from ghoshell_moss.core.blueprint.channel_builder import new_channel
    #   channel = new_channel(name="my_cell", description="...")
    #
    #   @channel.build.command()
    #   async def ping() -> str:
    #       return "pong"
    #
    #   await matrix.provide_channel(channel)

    pass  # replace with channel setup


if __name__ == "__main__":
    Matrix.discover().run(main)
