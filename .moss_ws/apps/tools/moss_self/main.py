"""Moss Self Channel — expose moss CLI as CTML commands for ghost self-bootstrapping."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel


# --- runtime: project root from MOSS_WORKSPACE env (injected by HostAppStore) ---
def _get_project_root() -> Path | None:
    workspace = os.environ.get("MOSS_WORKSPACE")
    if workspace:
        return Path(workspace).parent
    return None


_PROJECT_ROOT = _get_project_root()

chan = new_channel(
    name="moss_self",
    description="Moss CLI self-control channel. Execute moss commands via CTML.",
)

# --- build-time: subprocess reflection, cached ---
_INSTRUCTION: str | None = None


def _get_instruction() -> str:
    global _INSTRUCTION
    if _INSTRUCTION is not None:
        return _INSTRUCTION

    reflect_script = Path(__file__).parent / "reflect_cli.py"
    proc = subprocess.run(
        [sys.executable, str(reflect_script)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        _INSTRUCTION = f"Error reflecting moss CLI:\n{proc.stderr}"
    else:
        _INSTRUCTION = proc.stdout
    return _INSTRUCTION


@chan.build.instruction
def instruction():
    return _get_instruction()


# --- runtime: subprocess execution ---
@chan.build.command(
    name="exec",
    doc="Execute a moss CLI command. Pass the full command string after 'moss --ai'.",
    always_observe=True,
)
async def exec_command(text__: str) -> str:
    """
    :param text__: The moss command string.
                   e.g. 'codex get-interface ghoshell_moss.channels.typer_channel'
    """
    kwargs = {}
    if _PROJECT_ROOT is not None:
        kwargs["cwd"] = str(_PROJECT_ROOT)

    proc = await asyncio.create_subprocess_exec(
        "moss", "--ai", *text__.split(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    stdout, stderr = await proc.communicate()
    return stdout.decode() + stderr.decode()


async def main(matrix: Matrix):
    await matrix.provide_channel(chan)


if __name__ == "__main__":
    Matrix.discover().run(main)
