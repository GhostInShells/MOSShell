from ghoshell_moss.core.blueprint.channel_builder import ObserveError
from ghoshell_moss.core.concepts.channel import Channel

__all__ = [
    'is_g1_available',
    'check_g1_available',
    'check_channel_information',
]


def is_g1_available() -> bool:
    # todo
    return True


def check_g1_available() -> None:
    if not is_g1_available():
        raise ObserveError("g1 is not available yet")


def check_channel_information(channel: Channel) -> None:
    from ghoshell_moss.core.ctml import new_ctml_shell
    import asyncio

    async def _main() -> None:
        shell = new_ctml_shell(primitives=[])
        shell.main_channel.import_channels(channel)
        async with shell:
            print("---- static ----")
            print(shell.static_messages())
            print("---- dynamic ----")
            for msg in shell.dynamic_messages():
                print(msg.to_content_string())

    asyncio.run(_main())
