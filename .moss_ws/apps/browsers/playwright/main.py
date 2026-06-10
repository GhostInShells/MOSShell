"""Playwright Browser App — subprocess Sandbox eval via ModuleEval.

Spawns eval_server.py as child (Playwright runs in sync, no asyncio),
communicates via stdin/stdout JSON-line.  ModuleEval handles the spawn,
Sandbox provides builtins safety + persistent namespace.
"""

from pathlib import Path

from ghoshell_moss.channels.module_eval_channel import new_module_eval_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

_DOMAIN = str(Path(__file__).parent / "playwright_domain.py")


async def main(matrix: Matrix):
    chan = new_module_eval_channel(
        _DOMAIN,
        matrix=matrix,
        channel_name="playwright",
        description=(
            "Playwright browser control — subprocess Sandbox eval server. "
            "exec(code) for arbitrary Python in the browser namespace, "
            "vars() to list available objects, api(name) for detail."
        ),
    )
    await matrix.provide_channel(chan)


if __name__ == "__main__":
    Matrix.discover().run(main)
