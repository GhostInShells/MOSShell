from __future__ import annotations

from pathlib import Path

import pytest

from ghoshell_moss.channels import AppStoreChannel, AppStoreConfig


def _write_app_manifest(
    app_dir: Path,
    *,
    name: str,
    description: str,
    main: str,
    mode: str,
    body: str = "",
) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "CHANNEL.md").write_text(
        "\n".join(
            [
                "---",
                f'name: "{name}"',
                f'description: "{description}"',
                f'main: "{main}"',
                f'mode: "{mode}"',
                "---",
                "",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_script_module(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "main.py").write_text(
        """\
def add(a: int, b: int) -> int:
    return a + b
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_app_store_scans_and_reads(tmp_path: Path) -> None:
    apps_dir = tmp_path / "apps"
    app1 = apps_dir / "app1"
    _write_script_module(app1)
    _write_app_manifest(
        app1,
        name="adder",
        description="simple adder",
        main="main.py",
        mode="python",
        body="Use `add(a, b)`.",
    )

    store = AppStoreChannel(AppStoreConfig(root_dir=str(apps_dir)))
    async with store.bootstrap() as runtime:
        await runtime.wait_connected()

        meta = runtime.own_meta()
        assert meta is not None
        assert "adder" in meta.children
        assert meta.context
        context_text = meta.context[0].contents[0]["data"]["text"]
        assert "Available apps" in context_text
        assert "adder" in context_text
        assert "stopped" in context_text

        read = runtime.get_command("read")
        assert read is not None
        content = await read(name="adder")
        assert "# adder" in content
        assert "simple adder" in content
        assert "Use `add(a, b)`" in content


@pytest.mark.asyncio
async def test_app_store_open_close(tmp_path: Path) -> None:
    apps_dir = tmp_path / "apps"
    app1 = apps_dir / "app1"
    _write_script_module(app1)
    _write_app_manifest(
        app1,
        name="adder",
        description="simple adder",
        main="main.py",
        mode="python",
    )

    store = AppStoreChannel(AppStoreConfig(root_dir=str(apps_dir)))
    async with store.bootstrap() as runtime:
        await runtime.wait_connected()

        open_cmd = runtime.get_command("open")
        assert open_cmd is not None
        await open_cmd(name="adder", timeout=10.0)

        child = await runtime.fetch_sub_runtime("adder")
        assert child is not None
        add = child.get_command("add")
        assert add is not None
        assert await add(a=1, b=2) == 3

        close_cmd = runtime.get_command("close")
        assert close_cmd is not None
        await close_cmd(name="adder", timeout=10.0)

        # meta should reflect stopped state after refresh
        await runtime.refresh_metas()
        meta = runtime.own_meta()
        assert meta is not None
        context_text = meta.context[0].contents[0]["data"]["text"]
        assert "stopped" in context_text
