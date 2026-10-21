
"""Memento hooks — lifecycle slots as files, never as a registration API.

Three slots are agreed by name, and the name is the whole contract:

- ``on_commit``    right after an anchor lands; side effects only.
- ``commit_view``  one commit is read in detail; stdout is appended under the note.
- ``branch_view``  a branch view is generated; stdout is appended under the description.

A slot is a file: ``<root>/hooks/<slot>.py``. A branch's own ``hooks/`` sits beside the
ego-root one, and both are looked up — branch first. No file, no run; deleting the file is
turning the slot off.

A ``.py`` file is executed by the interpreter the memento process already runs on, so
writing the file is the whole of turning it on: no shebang, no ``chmod``, no manifest.

Hooks are a bypass, not a step: they run beside the commit, never in front of it. A
failure is recorded and left alone — no retry, no rollback, and it never blocks the main
path. That is why everything here returns results instead of raising.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

__all__ = [
    "HOOK_SLOTS",
    "HOOK_DIR_NAME",
    "DEFAULT_TIMEOUT",
    "HookResult",
    "hook_paths",
    "hook_env",
    "run_slot",
    "render_view_text",
    "log_results",
]

HOOK_SLOTS: tuple[str, ...] = ("on_commit", "commit_view", "branch_view")
"""The slots themselves. The file name decides when it runs."""

HOOK_DIR_NAME = "hooks"
"""Directory holding the slots, under a branch directory or the memento root."""

LOG_FILE_NAME = ".log.jsonl"
"""Where verdicts go: append-only, one row per run, readable with ``hook log``."""

DEFAULT_TIMEOUT = 20.0
"""Seconds a single hook may run. A view hook has a person waiting on it."""


class HookResult(BaseModel):
    """One hook's verdict — what ran, whether it landed, and what it said."""

    slot: str = Field(description="Which slot this ran for.")
    path: str = Field(description="The hook file, absolute.")
    ok: bool = Field(default=False, description="Ran, exited zero, produced no error.")
    exit_code: int | None = Field(default=None, description="Process exit code; None = never ran.")
    duration: float = Field(default=0.0, description="Seconds it took, including spawn.")
    stdout: str = Field(default="", description="What it printed; view hooks return this.")
    stderr: str = Field(default="", description="What it complained with.")
    error: str = Field(default="", description="Why it did not land: spawn failure, timeout, non-zero exit.")

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{self.slot}: ok ({self.duration:.2f}s)"
        return f"{self.slot}: {self.error or f'exit {self.exit_code}'}"


def hook_paths(roots: Iterable[Path | str | None], slot: str) -> list[Path]:
    """Every file that would run for ``slot`` — branch directory first, then the ego root.

    Returning the files rather than a single one is what lets a branch override the
    default without replacing it: both are mounted, both run.
    """
    found: list[Path] = []
    for root in roots:
        if root is None:
            continue
        path = Path(root) / HOOK_DIR_NAME / f"{slot}.py"
        if path.is_file():
            found.append(path)
    return found


def hook_env(
    slot: str,
    *,
    root: Path | str,
    branch: str = "",
    branch_dir: Path | str | None = None,
    coord: str = "",
    commit_dir: Path | str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The protocol: what a hook is promised it will always be handed.

    These names are the contract, not a convenience — a hook that has to guess where the
    commit landed is a hook that will be rewritten for every reader.
    """
    env = {
        "MEMENTO_HOOK_SLOT": slot,
        "MEMENTO_ROOT": str(Path(root).resolve()),
        "MEMENTO_BRANCH": branch,
        "MEMENTO_BRANCH_DIR": "" if branch_dir is None else str(Path(branch_dir).resolve()),
        "MEMENTO_COORD": coord,
        "MEMENTO_COMMIT_DIR": "" if commit_dir is None else str(Path(commit_dir).resolve()),
    }
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


async def _run_one(path: Path, env: Mapping[str, str], timeout: float) -> HookResult:
    """Run one hook file to completion, or to its deadline. Never raises."""
    result = HookResult(slot=env.get("MEMENTO_HOOK_SLOT", ""), path=str(path))
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(path),
            env={**os.environ, **env},
            cwd=str(path.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        result.duration = time.monotonic() - started
        result.error = f"spawn failed: {error}"
        return result

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        result.duration = time.monotonic() - started
        result.error = f"timeout after {timeout:g}s"
        return result

    result.duration = time.monotonic() - started
    result.exit_code = proc.returncode
    result.stdout = out.decode("utf-8", "replace")
    result.stderr = err.decode("utf-8", "replace")
    result.ok = proc.returncode == 0
    if not result.ok:
        result.error = f"exit {proc.returncode}"
    return result


async def run_slot(
    slot: str,
    roots: Sequence[Path | str | None],
    *,
    env_base: Mapping[str, str],
    timeout: float = DEFAULT_TIMEOUT,
    log_root: Path | str | None = None,
) -> list[HookResult]:
    """Run every file mounted on ``slot`` — concurrently, each on its own clock.

    Several hooks on one slot never wait for each other, and one failing cannot stop the
    others. ``hook_paths`` decides what is mounted; this decides what happens to it.
    """
    paths = hook_paths(roots, slot)
    if not paths:
        return []
    env = {**env_base, "MEMENTO_HOOK_SLOT": slot}
    results = await asyncio.gather(*[_run_one(path, env, timeout) for path in paths])
    log_results(results, log_root if log_root is not None else roots[0])
    return list(results)


def log_results(results: Iterable[HookResult], log_root: Path | str | None) -> None:
    """Append the verdicts to the root's hook log. A read-only disk must not break a view."""
    if log_root is None:
        return
    directory = Path(log_root) / HOOK_DIR_NAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / LOG_FILE_NAME).open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(
                    json.dumps(
                        {
                            "at": time.time(),
                            "slot": result.slot,
                            "path": result.path,
                            "ok": result.ok,
                            "exit_code": result.exit_code,
                            "duration": round(result.duration, 3),
                            "error": result.error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    except OSError:
        return


def render_view_text(results: Iterable[HookResult]) -> str:
    """What a view hook contributes to the page: the stdout of the ones that landed.

    A failed hook contributes nothing to the reader — its failure belongs in the log, not
    in the middle of a note. Silence and success must not look alike, though, so a broken
    slot is announced on one line instead of vanishing.
    """
    parts: list[str] = []
    broken: list[str] = []
    for result in results:
        if result.ok:
            text = result.stdout.strip()
            if text:
                parts.append(text)
        else:
            broken.append(result.summary)
    if broken:
        parts.append("[hook] " + "; ".join(broken))
    return "\n\n".join(parts)
