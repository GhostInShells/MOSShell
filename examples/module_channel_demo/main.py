import asyncio
import sys
from pathlib import Path

from ghoshell_moss.channels.module_channel import ModuleChannel

# Allow running directly via: python examples/module_channel_demo/main.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def demo_auto_wrap() -> None:
    chan = ModuleChannel(
        name="auto",
        description="auto wrap module functions",
        module_name="tests.module_fixtures.simple_module",
    )
    async with chan.bootstrap() as runtime:
        add = runtime.get_command("add")
        foo = runtime.get_command("foo")
        assert add is not None
        assert foo is not None
        print("[auto] add(1,2) =", await add(1, 2))
        print("[auto] foo() =", await foo())


async def demo_script_path() -> None:
    # `module_name` can also be a concrete python file path.
    # When `reload_on_bootstrap=False` (default), the script module instance is reused.
    # When `reload_on_bootstrap=True`, the script module is reloaded on every bootstrap.

    path = "tests/module_fixtures/script_module.py"

    chan = ModuleChannel(
        name="script",
        description="load from python script path",
        module_name=path,
    )
    async with chan.bootstrap() as runtime:
        inc = runtime.get_command("inc")
        assert await inc() == 1
    async with chan.bootstrap() as runtime:
        inc = runtime.get_command("inc")
        assert await inc() == 2

    chan_reload = ModuleChannel(
        name="script_reload",
        description="load from python script path (reload_on_bootstrap)",
        module_name=path,
        reload_on_bootstrap=True,
    )
    async with chan_reload.bootstrap() as runtime:
        inc = runtime.get_command("inc")
        assert await inc() == 1
    async with chan_reload.bootstrap() as runtime:
        inc = runtime.get_command("inc")
        assert await inc() == 1


async def demo_channel_py_priority() -> None:
    # This module lives in a package directory containing __channel__.py,
    # so ModuleChannel will prefer that explicit channel.
    chan = ModuleChannel(
        name="with_channel",
        description="prefer __channel__.py",
        module_name="tests.module_fixtures.with_channel.mod",
    )
    async with chan.bootstrap() as runtime:
        print("[__channel__.py] runtime.name=", runtime.name)
        hello = runtime.get_command("hello")
        assert hello is not None
        print("[__channel__.py] hello() =", await hello())

        should_not_be_used = runtime.get_command("should_not_be_used")
        assert should_not_be_used is None


async def demo_channel_py_file() -> None:
    # Explicitly load a file that defines __channel__.
    chan = ModuleChannel(
        name="file",
        description="load explicit channel from file",
        module_name="tests.module_fixtures.simple_module",
        channel_file="tests/module_fixtures/channel_val_module.py",
    )
    async with chan.bootstrap() as runtime:
        print("[channel_file] runtime.name=", runtime.name)
        hello = runtime.get_command("hello")
        assert hello is not None
        print("[channel_file] hello() =", await hello())


async def demo_stdlib_math() -> None:
    # Wrap a standard library module into a channel.
    # Note: many functions in `math` are C-implemented builtins.
    chan = ModuleChannel(
        name="math",
        description="wrap stdlib math as a channel",
        module_name="math",
    )
    async with chan.bootstrap() as runtime:
        sqrt = runtime.get_command("sqrt")
        isfinite = runtime.get_command("isfinite")
        assert sqrt is not None
        assert isfinite is not None
        print("[stdlib math] sqrt(9) =", await sqrt(9))
        print("[stdlib math] isfinite(1.0) =", await isfinite(1.0))


async def main() -> None:
    await demo_auto_wrap()
    await demo_script_path()
    await demo_channel_py_priority()
    await demo_channel_py_file()
    await demo_stdlib_math()


if __name__ == "__main__":
    asyncio.run(main())
