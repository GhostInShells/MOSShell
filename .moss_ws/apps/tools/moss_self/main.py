"""Moss Self Channel — expose moss CLI as CTML commands for ghost self-bootstrapping."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from ghoshell_moss import Observe
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel, CommandUtil


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

    kwargs = {}
    if _PROJECT_ROOT is not None:
        kwargs["cwd"] = str(_PROJECT_ROOT)

    # moss start — cognitive entry point
    start_proc = subprocess.run(
        ["moss", "--ai", "start"],
        capture_output=True,
        text=True,
        **kwargs,
    )

    # reflected command tree
    reflect_script = Path(__file__).parent / "reflect_cli.py"
    reflect_proc = subprocess.run(
        [sys.executable, str(reflect_script)],
        capture_output=True,
        text=True,
    )
    if reflect_proc.returncode != 0:
        _INSTRUCTION = f"Error reflecting moss CLI:\n{reflect_proc.stderr}"
    else:
        start_text = start_proc.stdout if start_proc.returncode == 0 else ""
        _INSTRUCTION = start_text + "\n---\n\n" + reflect_proc.stdout
    return _INSTRUCTION


@chan.build.instruction
def instruction():
    return _get_instruction()


# --- runtime: subprocess execution ---
@chan.build.command(
    name="exec",
    # always_observe=True,
)
async def exec_command(text__: str) -> Observe:
    """
    Execute a moss CLI command. 'moss --ai' is prepended automatically —
    pass ONLY the subcommand and its arguments, nothing else.
    Example: to run 'moss --ai codex concepts', pass 'codex concepts'.

    :param text__: subcommand + arguments. NEVER include 'moss' or '--ai'.
                   Correct: 'codex get-interface ghoshell_moss.channels.typer_channel'
                   Wrong:   'moss --ai codex get-interface ...'
    """
    args = text__.split()
    # Strip accidental 'moss' / '--ai' prefix — model may include them despite prompt.
    if args and args[0] == "moss":
        args.pop(0)
    if args and args[0] == "--ai":
        args.pop(0)

    kwargs = {}
    if _PROJECT_ROOT is not None:
        kwargs["cwd"] = str(_PROJECT_ROOT)

    proc = await asyncio.create_subprocess_exec(
        "moss", "--ai", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    stdout, stderr = await proc.communicate()
    return CommandUtil.observe(stdout.decode() + stderr.decode())


async def main(matrix: Matrix):
    await matrix.provide_channel(chan)


if __name__ == "__main__":
    Matrix.discover().run(main)
