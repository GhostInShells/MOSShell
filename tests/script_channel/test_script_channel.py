from pathlib import Path

import pytest

from ghoshell_moss.transports.script_channel.script_channel import ScriptChannelProxy


def _write_provider_launcher(tmp_path: Path) -> str:
    launcher = tmp_path / "provider_launcher.py"
    launcher.write_text(
        """\
import sys

from ghoshell_moss.transports.script_channel.provider_main import main


if __name__ == '__main__':
    main(sys.argv[1:])
""",
        encoding="utf-8",
    )
    return str(launcher)


@pytest.mark.asyncio
async def test_script_channel_baseline(tmp_path: Path):
    launcher = _write_provider_launcher(tmp_path)

    # Target script to be loaded as a channel (ModuleChannel path-loading).
    target = str((Path(__file__).resolve().parents[1] / "module_fixtures" / "script_module.py").resolve())

    proxy = ScriptChannelProxy(
        name="script_proxy",
        description="script-backed channel",
        provider_launcher=launcher,
        provider_target=target,
        channel_autostart=True,
    )

    async with proxy.bootstrap() as runtime:
        await runtime.wait_connected()

        cmd = runtime.get_command("add")
        assert cmd is not None
        assert await cmd(1, 2) == 3

        inc = runtime.get_command("inc")
        assert inc is not None
        assert await inc() == 1
        assert await inc() == 2


@pytest.mark.asyncio
async def test_script_channel_close_releases_process(tmp_path: Path):
    launcher = _write_provider_launcher(tmp_path)
    target = str((Path(__file__).resolve().parents[1] / "module_fixtures" / "script_module.py").resolve())

    proxy = ScriptChannelProxy(
        name="script_proxy",
        description="script-backed channel",
        provider_launcher=launcher,
        provider_target=target,
        channel_autostart=True,
    )

    runtime = proxy.bootstrap()
    async with runtime:
        await runtime.wait_connected()
        assert runtime.is_running()

    # After closing runtime, proxy connection should be closed.
    assert proxy._provider_connection.is_closed()


@pytest.mark.asyncio
async def test_script_channel_ignores_stdout_noise(tmp_path: Path):
    launcher = _write_provider_launcher(tmp_path)

    # Target script that prints to stdout.
    target = str((Path(__file__).resolve().parents[1] / "module_fixtures" / "script_module_with_stdout.py").resolve())

    proxy = ScriptChannelProxy(
        name="script_proxy",
        description="script-backed channel",
        provider_launcher=launcher,
        provider_target=target,
        channel_autostart=True,
    )

    async with proxy.bootstrap() as runtime:
        await runtime.wait_connected()
        cmd = runtime.get_command("add")
        assert cmd is not None
        assert await cmd(1, 2) == 3


@pytest.mark.asyncio
async def test_script_channel_stderr_inherit_does_not_break(tmp_path: Path):
    launcher = _write_provider_launcher(tmp_path)

    target = str((Path(__file__).resolve().parents[1] / "module_fixtures" / "script_module_with_stderr.py").resolve())

    proxy = ScriptChannelProxy(
        name="script_proxy",
        description="script-backed channel",
        provider_launcher=launcher,
        provider_target=target,
        channel_autostart=True,
        stderr="inherit",
    )

    async with proxy.bootstrap() as runtime:
        await runtime.wait_connected()
        cmd = runtime.get_command("add")
        assert cmd is not None
        assert await cmd(1, 2) == 3
