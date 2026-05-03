from ghoshell_moss.core.blueprint.matrix import Matrix
import asyncio


async def main(_matrix: Matrix) -> None:
    proxy = _matrix.channel_proxy(
        "apps/system_tests/provide_channel_case",
        name="test", only_allowed_in_main_cell=False,
    )
    async with proxy.bootstrap(_matrix.container) as runtime:
        await runtime.wait_connected()
        print("-- connected")
        print("-- metas", runtime.metas())
        await runtime.refresh_metas()
        print("-- refreshed metas", runtime.metas())
        foo = runtime.get_own_command('foo')
        result = await foo(3, 5)
        print("expect foo(3, 5) result is 8, %s given", result)

        while True:
            await runtime.refresh_metas()
            print("-- refreshed metas", runtime.metas())
            await asyncio.sleep(1)


if __name__ == '__main__':
    matrix = Matrix.discover()
    matrix.run(main)
