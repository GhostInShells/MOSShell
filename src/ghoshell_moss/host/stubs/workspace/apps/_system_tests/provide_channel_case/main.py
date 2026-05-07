from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel

channel = new_channel(name="test_provider", description="test provider")


@channel.build.command()
async def foo(a: int, b: int) -> int:
    return a + b


@channel.build.context_messages
async def get_content() -> list[str]:
    return ['hello world']


async def main(_matrix: Matrix) -> None:
    await _matrix.provide_channel(channel)


if __name__ == '__main__':
    matrix = Matrix.discover()
    matrix.run(main)
