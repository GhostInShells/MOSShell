from ghoshell_moss import PyChannel

channel = PyChannel(
    name="explicit_chan",
    description="A demo explicit channel loaded from __channel__.",
)


@channel.build.command()
async def ping() -> str:
    return "pong"


@channel.build.command()
async def echo(text: str) -> str:
    return text


__channel__ = channel
