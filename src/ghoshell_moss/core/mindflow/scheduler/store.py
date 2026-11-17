"""Timer storage — one file per plan, the filesystem as the source of truth.

Layout under ``root``::

    .gitignore          # *, !.gitignore, !README.md  (seeded by the workspace stub)
    state.yml           # {version, next_id}
    tasks/sch-000017.yml
    fired/2026/11/sch-000009.yml

Why one file per plan:

- **Isolation.** A corrupt record costs that record, never the set; the loader reports it
  and keeps going.
- **Atomicity.** A rewrite is ``tmp`` → ``fsync`` → ``os.replace``: a reader sees the old
  file or the new one, never half of either.
- **Legibility.** The model can read the file that will fire at 19:30 and see exactly what
  will happen, without asking a service.

Why a flat ``tasks/`` instead of time buckets: a recurring record's due instant changes
every fire, so bucketing would move the file (and break id → path) on every tick. The set
is small enough that the tick loop keeps it in memory and sorts there.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

from .models import TimerTask

__all__ = ["FilesystemTimerStore", "TimerStore", "TimerStoreError"]

_STATE_VERSION = 1
_TMP_SUFFIX = ".tmp"


class TimerStoreError(RuntimeError):
    """The store cannot honour the request (bad file, unwritable root, ...)."""


class TimerStore(ABC):
    """Where plans live between two lives of the process."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """The directory this store owns — printed to the model, read by a node."""

    @abstractmethod
    def load(self) -> list[TimerTask]:
        """Every armed record, oldest id first. Broken files are reported, not fatal."""

    @abstractmethod
    def allocate_id(self) -> str:
        """A fresh id, monotonic and never reused."""

    @abstractmethod
    def save(self, task: TimerTask) -> None:
        """Write (or overwrite) one record."""

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """Remove one record; False when it was not there."""

    @abstractmethod
    def archive(self, task: TimerTask) -> Path | None:
        """Move a finished record out of the armed set; the returned path is its grave."""

    @abstractmethod
    def load_archived(self, *, limit: int = 100) -> list[TimerTask]:
        """Recently finished records, newest first — enough to answer "did this ever fire"."""


class FilesystemTimerStore(TimerStore):
    """The default store: one YAML file per record under ``<root>/tasks``."""

    STATE_FILE = "state.yml"

    def __init__(self, root: Path, *, logger: LoggerItf | None = None) -> None:
        self._root = Path(root)
        self._tasks_dir = self._root / "tasks"
        self._fired_dir = self._root / "fired"
        self._logger = logger or get_moss_logger()
        #: None until the first load; healed from both the state file and the existing ids.
        self._next_seq: int | None = None

    @property
    def root(self) -> Path:
        return self._root

    # ── reading ──────────────────────────────────────────────────

    def load(self) -> list[TimerTask]:
        self._ensure()
        tasks: list[TimerTask] = []
        for path in sorted(self._tasks_dir.glob("*.yml")):
            task = self._read(path)
            if task is None:
                continue
            if task.kind == "once" and task.last_fire_at is not None \
                    and task.last_fire_at >= task.final_instant():
                # A crash between "watermark written" and "archived": the plan is spent.
                # Heal it here so it does not sit in the armed set forever.
                self.archive(task)
                continue
            tasks.append(task)
        self._heal_next_seq(tasks)
        return tasks

    def load_archived(self, *, limit: int = 100) -> list[TimerTask]:
        self._ensure()
        paths = sorted(self._fired_dir.rglob("*.yml"), reverse=True)[: max(0, limit)]
        tasks = [task for task in (self._read(path) for path in paths) if task is not None]
        return tasks

    # ── writing ──────────────────────────────────────────────────

    def allocate_id(self) -> str:
        self._ensure()
        if self._next_seq is None:
            self._heal_next_seq(self.load())
        assert self._next_seq is not None
        seq = self._next_seq
        self._next_seq = seq + 1
        self._write_state(seq + 1)
        return f"sch-{seq:06d}"

    def save(self, task: TimerTask) -> None:
        self._ensure()
        self._atomic_write(self._path_of(task.id), self._dump(task))

    def delete(self, task_id: str) -> bool:
        path = self._path_of(task_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def archive(self, task: TimerTask) -> Path | None:
        self._ensure()
        source = self._path_of(task.id)
        created = task.due or task.created
        stamp = created.astimezone().strftime("%Y/%m")
        target = self._fired_dir / stamp / f"{task.id}.yml"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            os.replace(source, target)
        else:
            self.save(task)
            os.replace(self._path_of(task.id), target)
        return target

    # ── internals ────────────────────────────────────────────────

    def _ensure(self) -> None:
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._fired_dir.mkdir(parents=True, exist_ok=True)
        # A crash mid-write leaves a `.tmp`; it was never authoritative, so sweep it.
        for stale in self._tasks_dir.glob(f"*{_TMP_SUFFIX}"):
            with_quiet_unlink(stale)
        for stale in self._fired_dir.rglob(f"*{_TMP_SUFFIX}"):
            with_quiet_unlink(stale)

    def _path_of(self, task_id: str) -> Path:
        return self._tasks_dir / f"{task_id}.yml"

    def _read(self, path: Path) -> TimerTask | None:
        try:
            data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            self._logger.exception("scheduler: cannot read %s", path)
            return None
        if not isinstance(data, dict):
            self._logger.error("scheduler: %s is not a mapping — skipped", path)
            return None
        data.setdefault("id", path.stem)
        try:
            return TimerTask.model_validate(data)
        except Exception:
            self._logger.exception("scheduler: %s does not validate", path)
            return None

    def _dump(self, task: TimerTask) -> str:
        return yaml.safe_dump(
            task.to_mapping(),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(path.parent),
                prefix=f".{path.name}.", suffix=_TMP_SUFFIX, delete=False,
        ) as handle:
            tmp = Path(handle.name)
            try:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                with_quiet_unlink(tmp)
                raise
        # Rename only after the handle is closed: a reader must never see a half file.
        os.replace(tmp, path)

    def _heal_next_seq(self, tasks: list[TimerTask]) -> None:
        """next_id = max(state, highest existing id + 1) — self-heals a lost counter write."""
        stored = self._read_state().get("next_id", 1)
        highest = 0
        for task in tasks:
            try:
                highest = max(highest, int(task.id.rsplit("-", 1)[-1]))
            except ValueError:
                continue
        self._next_seq = max(int(stored or 1), highest + 1, 1)

    def _read_state(self) -> dict[str, Any]:
        path = self._root / self.STATE_FILE
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            self._logger.exception("scheduler: cannot read %s", path)
            return {}
        return data if isinstance(data, dict) else {}

    def _write_state(self, next_id: int) -> None:
        payload = {"version": _STATE_VERSION, "next_id": next_id}
        self._atomic_write(
            self._root / self.STATE_FILE,
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        )


def with_quiet_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
