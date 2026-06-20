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
    try:
        await matrix.provide_channel(chan)
    except RuntimeError:
        import sys
        err = sys.exc_info()[1]
        if "module compilation failed" in str(err):
            B = "\033[1;33m"
            R = "\033[0m"
            raise RuntimeError(
                "Module compilation failed — likely missing Playwright browser.\n"
                f"\n"
                f"  {B}Run: playwright install chromium{R}\n"
            ) from err
        raise


if __name__ == "__main__":
    Matrix.discover().run(main)
