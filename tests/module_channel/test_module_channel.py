import pytest

from ghoshell_moss.channels.module_channel import ModuleChannel


@pytest.mark.asyncio
async def test_module_channel_wraps_module_functions() -> None:
    chan = ModuleChannel(
        name="m",
        description="test ModuleChannel",
        module_name="tests.module_fixtures.simple_module",
    )
    async with chan.bootstrap() as runtime:
        add_cmd = runtime.get_command("add")
        assert add_cmd is not None
        assert await add_cmd(1, 2) == 3

        foo_cmd = runtime.get_command("foo")
        assert foo_cmd is not None
        assert await foo_cmd() == 9527

        # private symbol should be ignored
        assert runtime.get_command("_private_value") is None


@pytest.mark.asyncio
async def test_module_channel_respects___all__() -> None:
    chan = ModuleChannel(
        name="m",
        description="test ModuleChannel",
        module_name="tests.module_fixtures.all_module",
    )
    async with chan.bootstrap() as runtime:
        only_this = runtime.get_command("only_this")
        assert only_this is not None
        assert await only_this() == "ok"

        not_exported = runtime.get_command("not_exported")
        assert not_exported is None


@pytest.mark.asyncio
async def test_module_channel_uses_explicit_channel_value() -> None:
    chan = ModuleChannel(
        name="m",
        description="",
        module_name="tests.module_fixtures.with_channel.mod",
    )
    async with chan.bootstrap() as runtime:
        # Prefer package-level __channel__.py.
        assert runtime.name == "pkg_explicit"
        hello = runtime.get_command("hello")
        assert hello is not None
        assert await hello() == "world"


@pytest.mark.asyncio
async def test_module_channel_can_load_explicit_channel_from_given_file() -> None:
    chan = ModuleChannel(
        name="m",
        description="",
        module_name="tests.module_fixtures.channel_val_module",
        channel_file="tests/module_fixtures/channel_val_module.py",
    )
    async with chan.bootstrap() as runtime:
        assert runtime.name == "explicit"
        hello = runtime.get_command("hello")
        assert hello is not None
        assert await hello() == "world"


@pytest.mark.asyncio
async def test_module_channel_can_load_from_python_script_path() -> None:
    chan = ModuleChannel(
        name="m",
        description="test ModuleChannel",
        module_name="tests/module_fixtures/script_module.py",
    )
    async with chan.bootstrap() as runtime:
        add_cmd = runtime.get_command("add")
        assert add_cmd is not None
        assert await add_cmd(1, 2) == 3


@pytest.mark.asyncio
async def test_module_channel_script_path_reuses_module_by_default() -> None:
    chan = ModuleChannel(
        name="m",
        description="test ModuleChannel",
        module_name="tests/module_fixtures/script_module.py",
    )

    async with chan.bootstrap() as runtime:
        inc_cmd = runtime.get_command("inc")
        assert inc_cmd is not None
        assert await inc_cmd() == 1

    async with chan.bootstrap() as runtime:
        inc_cmd = runtime.get_command("inc")
        assert inc_cmd is not None
        assert await inc_cmd() == 2


@pytest.mark.asyncio
async def test_module_channel_script_path_can_reload_on_bootstrap() -> None:
    chan = ModuleChannel(
        name="m",
        description="test ModuleChannel",
        module_name="tests/module_fixtures/script_module.py",
        reload_on_bootstrap=True,
    )

    async with chan.bootstrap() as runtime:
        inc_cmd = runtime.get_command("inc")
        assert inc_cmd is not None
        assert await inc_cmd() == 1

    async with chan.bootstrap() as runtime:
        inc_cmd = runtime.get_command("inc")
        assert inc_cmd is not None
        assert await inc_cmd() == 1


@pytest.mark.asyncio
async def test_module_channel_can_wrap_stdlib_math_positional_only_builtins() -> None:
    # Reference: examples/module_channel_demo/main.py::demo_stdlib_math
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

        # `math.sqrt(x, /)` and `math.isfinite(x, /)` are positional-only.
        # ModuleChannel should generate wrappers so MOSS can call them via kwargs.
        assert await sqrt(9) == 3
        assert await sqrt(x=9) == 3
        assert await isfinite(1.0) is True
        assert await isfinite(x=1.0) is True
        assert await isfinite(float("inf")) is False
        assert await isfinite(x=float("inf")) is False
