"""MOSS App entry point.

Quick start:
    1. Define commands on the channel builder below
    2. Test: moss apps test <group>/<name>
    3. Use in CTML: <apps.<group>_<name>:<command> />

For full context, read the generated CLAUDE.md in this directory.
"""
import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel


async def main(matrix: Matrix):
    channel = new_channel(
        name="genkits_video",
        description="AI video generation — text-to-video and image-to-video",
    )

    @channel.build.command()
    async def ping() -> str:
        """Example command — replace with your own."""
        return "pong"

    # More command patterns:
    #
    # @channel.build.command(always_observe=True)
    # async def observe_result(arg: str) -> str:
    #     """Command whose result flows into the observe stream."""
    #     return f"result: {arg}"
    #
    # @channel.build.command(blocking=False)
    # async def fire_and_forget() -> None:
    #     """Non-blocking command — does not occupy the channel."""
    #     ...

    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
