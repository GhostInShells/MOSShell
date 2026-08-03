"""
FsMemento — Memento FORMAT v3 filesystem reference implementation.

Storage/memory separation: this module uses ``_storage`` row types for all
jsonl I/O and projects them to ``abc`` API models at the public interface
boundary. Consumers (CLI, agent, ghost) only see ``abc`` types.

Discipline:
- No magic values. All type discriminators, status labels, and kind constants
  are imported from ``_storage``.
- No silent failures. Write-path errors either raise a typed ``MementoError``
  subclass or log at warning/error level.
- Logger is accepted via constructor (``LoggerItf``), defaulting to
  ``"moss.memento.fs"``. Key lifecycle events (commit, line creation, crash
  recovery) are logged at INFO; data anomalies at WARNING.
- Field names use the same short forms as the JSON contract (``id``, ``type``,
  ``t``) — these are the established convention throughout the codebase and
  match FORMAT.md key names exactly.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ulid import ULID

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.memento import _storage as store
from ghoshell_moss.memento.abc import (
    BranchMeta,
    BranchNotFoundError,
    BranchRef,
    BranchWindow,
    CheckoutRecord,
    Commit,
    CommitDetail,
    CommitNote,
    CommitNotFoundError,
    CommitRef,
    CommitView,
    ConfluentRecord,
    EmptyStagingError,
    LineNotFoundError,
    Memento as MementoABC,
    MementoError,
    MementoHooks,
    MomentFrozenError,
    MomentNotInCommitError,
    MomentRecord,
    NullHooks,
    ReadonlyLineError,
    join_trailers,
    new_branch_id,
    new_commit_id,
    new_moment_id,
)


# ── Filesystem helpers ─────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _y_m(commit_id: str) -> str:
    """ULID → Y-m path segment (UTC, pure function, O(1))."""
    ts = ULID().from_str(commit_id[len(store.COMMIT_ID_PREFIX):]).timestamp
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def _try_y_m(commit_id: str) -> str | None:
    """Best-effort ULID → Y-m. Returns None if the id can't be decoded."""
    try:
        return _y_m(commit_id)
    except (ValueError, IndexError):
        return None


def _read_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    """Read a jsonl file, skipping the torn last line (append-crash residue).

    :raise MementoError: if a non-last line fails to parse as JSON.
    """
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    lines = raw.rstrip("\n").split("\n")
    result: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            result.append(json.loads(stripped))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                # Torn last line — legal append-crash residue, skip silently.
                continue
            raise MementoError(
                f"jsonl parse error at line {i + 1} in {path}: {stripped[:200]}"
            ) from None
    return result


def _append_jsonl_lines(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows to a jsonl file. Creates parent dirs if needed.

    Each row is written as a compact single-line JSON object.
    POSIX O_APPEND guarantees atomic writes for lines < PIPE_BUF (4096B).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _write_atomic(path: Path, content: str) -> None:
    """Write content to a tmp file in the same directory, fsync, then atomically rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        os.write(tmp_fd, content.encode("utf-8"))
        os.fsync(tmp_fd)
    finally:
        os.close(tmp_fd)
    os.rename(tmp_name, str(path))


def _read_text_file(path: Path) -> str | None:
    """Read a single-line text file. Returns None if the file does not exist."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _resolve_staging(staging_path: Path) -> tuple[list[str], dict[str, store.StagingRow]]:
    """Read staging.jsonl and resolve last-wins by id.

    Returns (order_list, id_to_last_wins_row).
    """
    id_order: list[str] = []
    records: dict[str, store.StagingRow] = {}
    for obj in _read_jsonl_lines(staging_path):
        if obj.get("t") != store.ROW_TYPE_MOMENT:
            continue
        row = store.StagingRow(**obj)
        if row.id not in records:
            id_order.append(row.id)
        records[row.id] = row  # last-wins
    return id_order, records


# ── FsLine ─────────────────────────────────────────────────────────────────────


class FsLine:
    """Branch handle implementing the Line protocol.

    Bound to a branch uid. Delegates storage I/O to the owning FsMemento;
    this class is the public-facing handle that consumers interact with.
    """

    def __init__(self, memento: FsMemento, branch_uid: str, readonly: bool = False):
        self._memento = memento
        self._uid = branch_uid
        self._readonly = readonly

    @property
    def branch_identifier(self) -> str:
        return self._uid

    @property
    def name(self) -> str:
        # Resolve current head name from heads/ pointer files.
        return self._memento._head_name_for(self._uid) or self._uid

    @property
    def ref(self) -> BranchRef | None:
        return self._memento._read_ref(self._uid)

    @property
    def readonly(self) -> bool:
        return self._readonly

    def __repr__(self) -> str:
        return f"FsLine(uid={self._uid!r}, readonly={self._readonly})"

    # ── Write ──

    def record(self, record: MomentRecord) -> None:
        if self._readonly:
            raise ReadonlyLineError(f"line {self._uid!r} is readonly")
        self._memento._record(self._uid, record)

    def commit(
        self,
        text: str = "",
        *,
        kind: str = store.COMMIT_KIND_SEMANTIC,
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        boundary_moment_id: str | None = None,
        by: str = "",
    ) -> CommitView:
        if self._readonly:
            raise ReadonlyLineError(f"line {self._uid!r} is readonly")
        return self._memento._commit(
            self._uid,
            text=text,
            kind=kind,
            threads=threads,
            resumes=resumes,
            suspends=suspends,
            extra_trailers=extra_trailers,
            boundary_moment_id=boundary_moment_id,
            by=by,
        )

    # ── Read ──

    def staging(self) -> list[MomentRecord]:
        _, rows = self._memento._resolve_staging_of(self._uid)
        return [_row_to_moment(r) for r in rows.values()]

    def log(self) -> list[CommitView]:
        return self._memento._line_log(self._uid)

    def window(self, *, detail_n: int = 10, summary_m: int = -1) -> BranchWindow:
        return self._memento._line_window(self._uid, detail_n=detail_n, summary_m=summary_m)


# ── Row → API model projection ─────────────────────────────────────────────────


def _row_to_moment(row: store.StagingRow | store.FrozenMomentRow) -> MomentRecord:
    """Project a storage row to the API MomentRecord."""
    return MomentRecord(
        id=row.id,
        created=row.created,
        type=row.type,
        content=row.content,
        payload=row.payload,
        threads=row.threads,
    )


def _ref_fields_to_branch_ref(rf: store.BranchRefFields) -> BranchRef:
    return BranchRef(origin=rf.origin, commit_id=rf.commit_id, moment_id=rf.moment_id)


def _branch_ref_to_fields(ref: BranchRef) -> store.BranchRefFields:
    return store.BranchRefFields(origin=ref.origin, commit_id=ref.commit_id, moment_id=ref.moment_id)


# ── FsMemento ──────────────────────────────────────────────────────────────────


class FsMemento(MementoABC):
    """Memento FORMAT v3 filesystem reference implementation.

    Constructor accepts a logger; key lifecycle events are logged at INFO,
    data anomalies at WARNING.
    """

    def __init__(
        self,
        root: str | Path,
        owner: str,
        hooks: MementoHooks | None = None,
        logger: LoggerItf | None = None,
    ):
        self._root = Path(root)
        self._owner = owner
        self._hooks = hooks or NullHooks()
        self._logger = logger or logging.getLogger("moss.memento.fs")

    @property
    def owner(self) -> str:
        return self._owner

    # ── Path resolution (internal) ──────────────────────────────────────────

    def _owner_dir(self) -> Path:
        return self._root / self._owner

    def _heads_dir(self) -> Path:
        return self._owner_dir() / "heads"

    def _head_path(self, name: str) -> Path:
        return self._heads_dir() / name

    def _workspace_dir(self, branch_uid: str) -> Path:
        return self._owner_dir() / "ws" / branch_uid

    def _ref_path(self, branch_uid: str) -> Path:
        return self._workspace_dir(branch_uid) / "ref"

    def _staging_path(self, branch_uid: str) -> Path:
        return self._workspace_dir(branch_uid) / "staging.jsonl"

    def _status_path(self, branch_uid: str) -> Path:
        return self._workspace_dir(branch_uid) / "status.json"

    def _commits_dir(self) -> Path:
        return self._owner_dir() / "commits"

    def _commit_dir_for(self, origin: str, commit_id: str) -> Path:
        if not commit_id.startswith(store.COMMIT_ID_PREFIX):
            raise CommitNotFoundError(f"invalid commit id prefix: {commit_id!r}")
        ym = _try_y_m(commit_id)
        if ym is None:
            raise CommitNotFoundError(f"unable to decode ULID from: {commit_id!r}")
        return self._root / origin / "commits" / ym / commit_id

    def _commit_dir(self, commit_id: str) -> Path:
        return self._commit_dir_for(self._owner, commit_id)

    def _commits_jsonl_path(self) -> Path:
        return self._owner_dir() / "commits.jsonl"

    def _branches_jsonl_path(self) -> Path:
        return self._owner_dir() / "branches.jsonl"

    def _checkouts_jsonl_path(self) -> Path:
        return self._owner_dir() / "checkouts.jsonl"

    def _confluents_jsonl_path(self) -> Path:
        return self._owner_dir() / "confluents.jsonl"

    def _meta_json_path(self) -> Path:
        return self._owner_dir() / "meta.json"

    # ── Head name resolution ────────────────────────────────────────────────

    def _resolve_uid(self, identifier: str) -> str:
        """Resolve an identifier to a branch uid.

        If the identifier starts with brn_, treat it as a uid directly.
        Otherwise, look it up as a head name via heads/{name}.
        """
        if identifier.startswith(store.BRANCH_ID_PREFIX):
            ws = self._workspace_dir(identifier)
            if ws.exists():
                return identifier
            raise BranchNotFoundError(
                f"branch uid {identifier!r} not found for owner {self._owner!r}"
            )
        # Name lookup
        head_path = self._head_path(identifier)
        if not head_path.exists():
            raise LineNotFoundError(
                f"line {identifier!r} not found for owner {self._owner!r}"
            )
        uid = _read_text_file(head_path)
        if not uid:
            raise LineNotFoundError(
                f"head file for {identifier!r} is empty"
            )
        return uid

    def _head_name_for(self, branch_uid: str) -> str | None:
        """Find the current head name for a branch uid by scanning heads/."""
        heads_dir = self._heads_dir()
        if not heads_dir.exists():
            return None
        for entry in heads_dir.iterdir():
            if entry.is_file():
                content = _read_text_file(entry)
                if content == branch_uid:
                    return entry.name
        return None

    def _write_head(self, name: str, branch_uid: str) -> None:
        """Atomically write a head pointer file."""
        self._heads_dir().mkdir(parents=True, exist_ok=True)
        _write_atomic(self._head_path(name), f"{branch_uid}\n")

    # ── Ref I/O ─────────────────────────────────────────────────────────────

    def _read_ref(self, branch_uid: str) -> BranchRef | None:
        path = self._ref_path(branch_uid)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            return None
        return BranchRef(**data)

    def _write_ref(self, branch_uid: str, ref: BranchRef) -> None:
        self._workspace_dir(branch_uid).mkdir(parents=True, exist_ok=True)
        _write_atomic(self._ref_path(branch_uid),
                      json.dumps(ref.model_dump(mode="json"), ensure_ascii=False))

    # ── Staging ─────────────────────────────────────────────────────────────

    def _resolve_staging_of(
        self, branch_uid: str
    ) -> tuple[list[str], dict[str, store.StagingRow]]:
        return _resolve_staging(self._staging_path(branch_uid))

    def _frozen_ids(self, branch_uid: str) -> set[str]:
        """Walk parent chain and collect all frozen moment ids for this branch."""
        frozen: set[str] = set()
        ref = self._read_ref(branch_uid)
        cid = ref.commit_id if ref else None
        while cid:
            moments_path = self._commit_dir(cid) / "moments.jsonl"
            if moments_path.exists():
                for obj in _read_jsonl_lines(moments_path):
                    if obj.get("t") == store.ROW_TYPE_MOMENT:
                        frozen.add(obj["id"])
            meta = self._load_commit_meta(cid)
            parent = meta.get("parent") if meta else None
            cid = parent["commit_id"] if parent else None
        return frozen

    def _record(self, branch_uid: str, record: MomentRecord) -> None:
        staging_path = self._staging_path(branch_uid)
        if not staging_path.parent.exists():
            raise BranchNotFoundError(
                f"branch workspace for {branch_uid!r} not found — create line first"
            )
        if record.id in self._frozen_ids(branch_uid):
            raise MomentFrozenError(f"moment {record.id!r} is frozen")
        _append_jsonl_lines(staging_path, [
            {"t": store.ROW_TYPE_MOMENT, **record.model_dump(mode="json")}
        ])
        self._hooks.on_record_staged(branch_uid, record)

    # ── Commit ──────────────────────────────────────────────────────────────

    def _commit(
        self,
        branch_uid: str,
        *,
        text: str = "",
        kind: str = store.COMMIT_KIND_SEMANTIC,
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        boundary_moment_id: str | None = None,
        by: str = "",
    ) -> CommitView:
        id_order, last_wins = self._resolve_staging_of(branch_uid)
        if not last_wins:
            raise EmptyStagingError(
                f"staging is empty for branch {branch_uid!r}; nothing to commit"
            )

        # Apply boundary_moment_id slice: freeze prefix up to and including it.
        if boundary_moment_id is not None:
            if boundary_moment_id not in last_wins:
                raise MomentNotInCommitError(
                    f"boundary moment {boundary_moment_id!r} not in staging"
                )
            cutoff = id_order.index(boundary_moment_id) + 1
            freeze_order = id_order[:cutoff]
            remain_order = id_order[cutoff:]
        else:
            freeze_order = id_order
            remain_order = []

        freeze_moments = [last_wins[mid] for mid in freeze_order]

        # Build trailers
        trailers: list[tuple[str, str]] = []
        if kind == store.COMMIT_KIND_SEMANTIC and text:
            pass  # text is the body; trailers appended below
        elif kind == store.COMMIT_KIND_MECHANICAL:
            trailers.append(("Kind", kind))
        for thread in threads:
            trailers.append(("Thread", thread))
        for resume_id in resumes:
            trailers.append(("Resumes", resume_id))
        for suspend_name in suspends:
            trailers.append(("Suspends", suspend_name))
        for k, v in extra_trailers:
            trailers.append((k, v))

        if text and trailers:
            body = join_trailers(text, trailers)
        elif trailers:
            body = join_trailers("", trailers)
        else:
            body = text

        ref = self._read_ref(branch_uid)
        parent = _branch_ref_to_fields(ref) if ref else None

        commit_id = new_commit_id()
        commit_dir = self._commit_dir(commit_id)

        # Read current status for the CommiRef
        status_data = self._read_status(branch_uid)

        # 1. Write commit directory under a tmp name, then atomically rename.
        commit_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=str(commit_dir.parent), prefix=".tmp_cmt_"))
        try:
            # meta.json
            meta_row = store.CommitMetaRow(
                commit_id=commit_id,
                parent=parent,
                kind=kind,
                created=_now_utc(),
            )
            (tmp_dir / "meta.json").write_text(
                json.dumps(meta_row.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # moments.jsonl — frozen last-wins rows from staging
            moment_rows: list[dict[str, Any]] = []
            for row in freeze_moments:
                moment_rows.append({"t": store.ROW_TYPE_MOMENT, **row.model_dump(mode="json")})
            (tmp_dir / "moments.jsonl").write_text(
                "\n".join(
                    json.dumps(r, ensure_ascii=False, separators=(",", ":"))
                    for r in moment_rows
                ) + "\n",
                encoding="utf-8",
            )

            # notes.jsonl — initial commit_note
            note_row = store.CommitNoteRow(
                ref=commit_id, title=text.split("\n")[0] if text else "", body=body, by=by,
            )
            (tmp_dir / "notes.jsonl").write_text(
                json.dumps({"t": store.ROW_TYPE_COMMIT_NOTE, **note_row.model_dump(mode="json")},
                          ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            # fsync tmp dir
            for f in tmp_dir.iterdir():
                _fsync_path(f)
            _fsync_path(tmp_dir)

            # 2. Atomic rename
            commit_dir.parent.mkdir(parents=True, exist_ok=True)
            os.rename(str(tmp_dir), str(commit_dir))
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise

        # 3. Append commits.jsonl
        commit_ref_row = store.CommitRefRow(
            commit_id=commit_id,
            branch_uid=branch_uid,
            parent=parent,
            kind=kind,
        )
        _append_jsonl_lines(self._commits_jsonl_path(), [
            {"t": store.ROW_TYPE_COMMIT_REF, **commit_ref_row.model_dump(mode="json")}
        ])

        # 4. Truncate staging
        if remain_order:
            # Keep remaining moments: rewrite staging with only the unfrozen ones
            remain_rows = [last_wins[mid] for mid in remain_order]
            staging_path = self._staging_path(branch_uid)
            staging_path.write_text(
                "\n".join(
                    json.dumps({"t": store.ROW_TYPE_MOMENT, **r.model_dump(mode="json")},
                              ensure_ascii=False, separators=(",", ":"))
                    for r in remain_rows
                ) + "\n",
                encoding="utf-8",
            )
        else:
            staging_path = self._staging_path(branch_uid)
            if staging_path.exists():
                staging_path.write_text("", encoding="utf-8")

        # 5. Update ref to point to the new commit
        new_ref = BranchRef(origin="", commit_id=commit_id)
        self._write_ref(branch_uid, new_ref)

        # Build CommitView for return
        commit = Commit(id=commit_id, created=meta_row.created)
        commit_note = CommitNote(
            ref=commit_id, title=note_row.title, body=body, ts=note_row.ts, by=by,
        )
        view = CommitView(commit=commit, note=commit_note, note_seq=0)

        self._hooks.on_commit(branch_uid, view)
        self._logger.info(
            "commit %s on branch %s (kind=%s, moments=%d)",
            commit_id, branch_uid, kind, len(freeze_moments),
        )
        return view

    # ── Line history ────────────────────────────────────────────────────────

    def _line_log(self, branch_uid: str) -> list[CommitView]:
        """Return commit history in chronological order (oldest first)."""
        result: list[CommitView] = []
        ref = self._read_ref(branch_uid)
        cid = ref.commit_id if ref else None
        while cid:
            view = self._load_commit_view(cid)
            if view:
                result.append(view)
            meta = self._load_commit_meta(cid)
            parent = meta.get("parent") if meta else None
            cid = parent["commit_id"] if parent else None
        result.reverse()
        return result

    def _line_window(
        self, branch_uid: str, *, detail_n: int, summary_m: int
    ) -> BranchWindow:
        log = self._line_log(branch_uid)
        staging_id_order, staging_rows = self._resolve_staging_of(branch_uid)
        detail_moments = [_row_to_moment(staging_rows[mid]) for mid in staging_id_order]

        # summaries = older commits (beyond detail_n), plus staging forms the detail zone
        if summary_m == -1:
            summaries = log
        else:
            summaries = log[:summary_m]
        return BranchWindow(summaries=summaries, details=detail_moments)

    # ── Commit loading ──────────────────────────────────────────────────────

    def _load_commit_meta(self, commit_id: str) -> dict[str, Any] | None:
        meta_path = self._commit_dir(commit_id) / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _load_commit_view(self, commit_id: str) -> CommitView | None:
        commit_dir = self._commit_dir(commit_id)
        if not commit_dir.exists():
            return None
        meta = self._load_commit_meta(commit_id)
        if not meta:
            return None
        commit = Commit(id=commit_id, created=meta.get("created", _now_utc()))

        notes_path = commit_dir / "notes.jsonl"
        notes: list[CommitNote] = []
        if notes_path.exists():
            for obj in _read_jsonl_lines(notes_path):
                t = obj.get("t")
                if t == store.ROW_TYPE_COMMIT_NOTE and obj.get("ref") == commit_id:
                    notes.append(CommitNote(
                        ref=commit_id,
                        title=obj.get("title", ""),
                        body=obj.get("body", ""),
                        ts=obj.get("ts", _now_utc()),
                        by=obj.get("by", ""),
                    ))
        if not notes:
            notes.append(CommitNote(ref=commit_id))
        return CommitView(commit=commit, note=notes[-1], note_seq=len(notes) - 1)

    def _load_commit_moments(self, commit_id: str) -> list[MomentRecord]:
        moments_path = self._commit_dir(commit_id) / "moments.jsonl"
        if not moments_path.exists():
            return []
        result: list[MomentRecord] = []
        for obj in _read_jsonl_lines(moments_path):
            if obj.get("t") == store.ROW_TYPE_MOMENT:
                result.append(_row_to_moment(store.FrozenMomentRow(**obj)))
        return result

    def _read_status(self, branch_uid: str) -> store.BranchStatusRow | None:
        path = self._status_path(branch_uid)
        if not path.exists():
            return None
        try:
            return store.BranchStatusRow(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            self._logger.warning("failed to parse status.json for %s", branch_uid)
            return None

    def _write_status(self, branch_uid: str, status: store.BranchStatusRow) -> None:
        self._workspace_dir(branch_uid).mkdir(parents=True, exist_ok=True)
        _write_atomic(
            self._status_path(branch_uid),
            json.dumps(status.model_dump(mode="json"), ensure_ascii=False, indent=2),
        )

    # ── Crash recovery ──────────────────────────────────────────────────────

    def _recover(self) -> None:
        """Run crash recovery on Memento instance load. Idempotent.

        Recovery priorities (FORMAT v3 §11):
        1. commits.jsonl tail-line integrity
        2. workspace ↔ branches.jsonl consistency
        3. head → workspace existence
        4. index file torn tail lines
        """
        # 1. commits.jsonl tail integrity → truncate orphan staging
        commits_path = self._commits_jsonl_path()
        if commits_path.exists():
            ref_rows = _read_jsonl_lines(commits_path)
            if ref_rows:
                last_ref = ref_rows[-1]
                last_commit_id = last_ref.get("commit_id", "")
                last_branch_uid = last_ref.get("branch_uid", "")
                if last_commit_id and last_branch_uid:
                    commit_dir = self._commit_dir(last_commit_id)
                    staging_path = self._staging_path(last_branch_uid)
                    if commit_dir.exists() and staging_path.exists():
                        _, staging_rows = self._resolve_staging_of(last_branch_uid)
                        commit_moment_ids = {
                            m.id for m in self._load_commit_moments(last_commit_id)
                        }
                        # If all staging ids are in the last commit, truncate
                        if staging_rows and set(staging_rows.keys()).issubset(commit_moment_ids):
                            self._logger.info(
                                "recovery: truncating staging for %s (already in commit %s)",
                                last_branch_uid, last_commit_id,
                            )
                            staging_path.write_text("", encoding="utf-8")

        # 2. workspace ↔ branches.jsonl consistency
        ws_dir = self._owner_dir() / "ws"
        if ws_dir.exists():
            branches_path = self._branches_jsonl_path()
            known_uids: set[str] = set()
            if branches_path.exists():
                for obj in _read_jsonl_lines(branches_path):
                    uid = obj.get("uid", "")
                    if uid:
                        known_uids.add(uid)
            for ws_entry in ws_dir.iterdir():
                if ws_entry.is_dir() and ws_entry.name not in known_uids:
                    # Workspace exists but no branches.jsonl entry — add one
                    self._logger.warning(
                        "recovery: workspace %s missing from branches.jsonl, adding entry",
                        ws_entry.name,
                    )
                    ref = self._read_ref(ws_entry.name)
                    fork_ref = _branch_ref_to_fields(ref) if ref else store.BranchRefFields(
                        commit_id="",
                    )
                    status = self._read_status(ws_entry.name)
                    status_str = status.status if status else store.BRANCH_STATUS_ACTIVE
                    _append_jsonl_lines(branches_path, [{
                        "t": store.ROW_TYPE_BRANCH_META,
                        "uid": ws_entry.name,
                        "name": self._head_name_for(ws_entry.name) or ws_entry.name,
                        "status": status_str,
                        "fork_ref": fork_ref.model_dump(mode="json"),
                        "created": _now_utc().isoformat(),
                        "updated": _now_utc().isoformat(),
                    }])

        # 3. head → workspace existence
        heads_dir = self._heads_dir()
        if heads_dir.exists():
            for entry in heads_dir.iterdir():
                if entry.is_file():
                    uid = _read_text_file(entry)
                    if uid and not self._workspace_dir(uid).exists():
                        self._logger.warning(
                            "recovery: head '%s' → uid %s has no workspace, removing head",
                            entry.name, uid,
                        )
                        entry.unlink()

        # 4. Index file torn tail rows — handled passively by _read_jsonl_lines.


# ── Line management (create / get / list / delete) ─────────────────────────


    def create_line(
        self,
        name: str,
        *,
        from_ref: BranchRef | None = None,
        overlay: dict[str, Any] | None = None,
    ) -> FsLine:
        branch_uid = new_branch_id()
        self._logger.info("creating line %s (uid=%s, from=%s)", name, branch_uid, from_ref)

        # Write head pointer
        self._write_head(name, branch_uid)

        # Write ref
        initial_ref = from_ref or BranchRef(origin="", commit_id="")
        self._write_ref(branch_uid, initial_ref)

        # Initial status
        self._write_status(branch_uid, store.BranchStatusRow(
            status=store.BRANCH_STATUS_ACTIVE,
            title="",
            description="",
        ))

        # Append branches.jsonl
        fork_fields = _branch_ref_to_fields(initial_ref) if from_ref else store.BranchRefFields(
            commit_id="",
        )
        now = _now_utc()
        _append_jsonl_lines(self._branches_jsonl_path(), [{
            "t": store.ROW_TYPE_BRANCH_META,
            "uid": branch_uid,
            "name": name,
            "status": store.BRANCH_STATUS_ACTIVE,
            "fork_ref": fork_fields.model_dump(mode="json"),
            "created": now.isoformat(),
            "updated": now.isoformat(),
        }])

        # Append checkouts.jsonl — record the fork event
        _append_jsonl_lines(self._checkouts_jsonl_path(), [{
            "t": store.ROW_TYPE_CHECKOUT,
            "branch_uid": branch_uid,
            "from_ref": fork_fields.model_dump(mode="json"),
            "owner": self._owner,
            "created": now.isoformat(),
        }])

        # Write overlay to meta.json if provided
        if overlay:
            meta_path = self._meta_json_path()
            existing: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    existing = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            existing["overlay"] = overlay
            _write_atomic(meta_path, json.dumps(existing, ensure_ascii=False, indent=2))

        line = FsLine(self, branch_uid)
        self._hooks.on_line_created(name, from_ref)
        self._hooks.on_branch_checkout(branch_uid, from_ref)
        return line

    def _ensure_main_line(self) -> FsLine:
        """Degenerate path: create the default 'main' line on first use.

        Per FORMAT v3 §13, the degenerate form does not require explicit
        create_line — get_line('main') creates it implicitly on first access.
        """
        self._logger.info("auto-creating degenerate main line for owner %s", self._owner)
        return self.create_line("main")

    def get_line(self, identifier: str, *, origin: str | None = None) -> FsLine:
        # Cross-owner read-only handle
        if origin is not None and origin != self._owner:
            # For cross-owner, identifier is always a uid
            # We need to verify the branch exists in the origin owner
            origin_ws = self._root / origin / "ws" / identifier
            if not origin_ws.exists():
                raise BranchNotFoundError(
                    f"branch uid {identifier!r} not found for owner {origin!r}"
                )
            return FsLine(self, identifier, readonly=True)

        # Degenerate path: if 'main' head does not exist, auto-create on first access
        if identifier == "main":
            head_path = self._head_path("main")
            if not head_path.exists():
                return self._ensure_main_line()

        # Check if identifier is a head name; if not found and looks like uid, try direct
        head_path = self._head_path(identifier)
        if head_path.exists():
            uid = _read_text_file(head_path)
            if uid:
                return FsLine(self, uid)

        # Check if it's a uid directly
        ws = self._workspace_dir(identifier)
        if ws.exists():
            return FsLine(self, identifier)

        raise LineNotFoundError(f"line {identifier!r} not found for owner {self._owner!r}")

    def list_lines(self) -> list[str]:
        heads_dir = self._heads_dir()
        if not heads_dir.exists():
            return []
        return sorted(
            entry.name for entry in heads_dir.iterdir() if entry.is_file()
        )

    def list_all_branches(self) -> list[BranchMeta]:
        branches_path = self._branches_jsonl_path()
        if not branches_path.exists():
            return []
        # Reconstruct current name for each uid from heads/
        uid_to_name: dict[str, str] = {}
        heads_dir = self._heads_dir()
        if heads_dir.exists():
            for entry in heads_dir.iterdir():
                if entry.is_file():
                    content = _read_text_file(entry)
                    if content:
                        uid_to_name[content] = entry.name

        result: list[BranchMeta] = []
        seen: set[str] = set()
        for obj in reversed(_read_jsonl_lines(branches_path)):
            if obj.get("t") != store.ROW_TYPE_BRANCH_META:
                continue
            uid = obj.get("uid", "")
            if not uid or uid in seen:
                continue
            seen.add(uid)
            rf = obj.get("fork_ref", {})
            fork_ref = BranchRef(**rf) if rf else None
            result.append(BranchMeta(
                uid=uid,
                name=uid_to_name.get(uid, obj.get("name", uid)),
                status=obj.get("status", store.BRANCH_STATUS_ACTIVE),
                fork_ref=fork_ref,
                created=obj.get("created", _now_utc()),
                updated=obj.get("updated", _now_utc()),
            ))
        result.reverse()
        return result

    def delete_line(self, name: str) -> None:
        head_path = self._head_path(name)
        if not head_path.exists():
            raise LineNotFoundError(f"line {name!r} not found")
        # Read the uid before deleting (for hook)
        uid = _read_text_file(head_path)
        head_path.unlink()
        self._logger.info("deleted head '%s' (uid=%s) — workspace and commits survive", name, uid)
        self._hooks.on_line_deleted(name)

    # ── Commit read & interpretation ────────────────────────────────────────

    def show(self, commit_id: str) -> CommitDetail:
        commit_dir = self._commit_dir(commit_id)
        if not commit_dir.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")

        meta = self._load_commit_meta(commit_id)
        if not meta:
            raise CommitNotFoundError(f"commit {commit_id!r} meta not readable")

        commit = Commit(id=commit_id, created=meta.get("created", _now_utc()))
        moments = self._load_commit_moments(commit_id)

        notes_path = commit_dir / "notes.jsonl"
        notes: list[CommitNote] = []
        if notes_path.exists():
            for obj in _read_jsonl_lines(notes_path):
                t = obj.get("t")
                ref = obj.get("ref", "")
                if t == store.ROW_TYPE_COMMIT_NOTE and ref == commit_id:
                    notes.append(CommitNote(
                        ref=commit_id,
                        title=obj.get("title", ""),
                        body=obj.get("body", ""),
                        ts=obj.get("ts", _now_utc()),
                        by=obj.get("by", ""),
                    ))
                elif t == store.ROW_TYPE_MOMENT_NOTE:
                    # Apply moment-level thread annotations
                    for m in moments:
                        if m.id == ref:
                            m.threads = obj.get("threads", [])
        if not notes:
            notes.append(CommitNote(ref=commit_id))
        return CommitDetail(commit=commit, moments=moments, notes=notes)

    def notes(self, commit_id: str) -> list[CommitNote]:
        notes_path = self._commit_dir(commit_id) / "notes.jsonl"
        if not notes_path.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        result: list[CommitNote] = []
        for obj in _read_jsonl_lines(notes_path):
            if obj.get("t") == store.ROW_TYPE_COMMIT_NOTE and obj.get("ref") == commit_id:
                result.append(CommitNote(
                    ref=commit_id,
                    title=obj.get("title", ""),
                    body=obj.get("body", ""),
                    ts=obj.get("ts", _now_utc()),
                    by=obj.get("by", ""),
                ))
        if not result:
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        return result

    def annotate(
        self, commit_id: str, title: str = "", body: str = "", *, by: str = ""
    ) -> CommitView:
        notes_path = self._commit_dir(commit_id) / "notes.jsonl"
        if not notes_path.parent.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        note_row = store.CommitNoteRow(ref=commit_id, title=title, body=body, by=by)
        _append_jsonl_lines(notes_path, [
            {"t": store.ROW_TYPE_COMMIT_NOTE, **note_row.model_dump(mode="json")}
        ])
        view = self._load_commit_view(commit_id)
        if view is None:
            raise CommitNotFoundError(f"commit {commit_id!r} view missing after annotate")
        self._hooks.on_reinterpreted(commit_id, view)
        return view

    def annotate_moment(
        self, commit_id: str, moment_id: str, threads: Sequence[str], *, by: str = ""
    ) -> None:
        notes_path = self._commit_dir(commit_id) / "notes.jsonl"
        if not notes_path.parent.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        # Verify moment exists in this commit
        moments = self._load_commit_moments(commit_id)
        found = any(m.id == moment_id for m in moments)
        if not found:
            raise MomentNotInCommitError(
                f"moment {moment_id!r} not in commit {commit_id!r}"
            )
        moment_note_row = store.MomentNoteRow(ref=moment_id, threads=list(threads), by=by)
        _append_jsonl_lines(notes_path, [
            {"t": store.ROW_TYPE_MOMENT_NOTE, **moment_note_row.model_dump(mode="json")}
        ])

    # ── Owner-level queries ─────────────────────────────────────────────────

    def log(self) -> list[CommitRef]:
        commits_path = self._commits_jsonl_path()
        if not commits_path.exists():
            return []
        result: list[CommitRef] = []
        for obj in _read_jsonl_lines(commits_path):
            if obj.get("t") != store.ROW_TYPE_COMMIT_REF:
                continue
            parent_data = obj.get("parent")
            parent = BranchRef(**parent_data) if parent_data else None
            result.append(CommitRef(
                commit_id=obj["commit_id"],
                branch=obj.get("branch_uid", ""),
                parent=parent,
                ts=obj.get("ts", _now_utc()),
                kind=obj.get("kind", store.COMMIT_KIND_SEMANTIC),
            ))
        return result

    def commit_space(self, commit_id: str) -> str:
        return str(self._commit_dir(commit_id).resolve())

    def checkouts(self) -> list[CheckoutRecord]:
        path = self._checkouts_jsonl_path()
        if not path.exists():
            return []
        result: list[CheckoutRecord] = []
        for obj in _read_jsonl_lines(path):
            if obj.get("t") != store.ROW_TYPE_CHECKOUT:
                continue
            from_data = obj.get("from_ref", {})
            from_ref = BranchRef(**from_data) if from_data else None
            result.append(CheckoutRecord(
                branch_uid=obj["branch_uid"],
                from_ref=from_ref,
                owner=obj.get("owner", ""),
                created=obj.get("created", _now_utc()),
            ))
        return result

    def confluents(self) -> list[ConfluentRecord]:
        path = self._confluents_jsonl_path()
        if not path.exists():
            return []
        result: list[ConfluentRecord] = []
        for obj in _read_jsonl_lines(path):
            if obj.get("t") != store.ROW_TYPE_CONFLUENT:
                continue
            result.append(ConfluentRecord(
                from_branch_uid=obj["from_branch_uid"],
                from_owner=obj["from_owner"],
                to_branch_uid=obj["to_branch_uid"],
                to_owner=obj["to_owner"],
                kind=obj.get("kind", store.CONFLUENT_KIND_REFERENCE),
                created=obj.get("created", _now_utc()),
            ))
        return result


def _fsync_path(path: Path) -> None:
    """fsync a directory or file."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass  # best-effort


# ── Public factory ─────────────────────────────────────────────────────────────


def new_filesystem_memento(
    root: str | Path,
    owner: str,
    hooks: MementoHooks | None = None,
    logger: LoggerItf | None = None,
) -> FsMemento:
    """Create a FsMemento instance and run crash recovery.

    Crash recovery is idempotent — safe to call on every load.
    """
    memento = FsMemento(str(root), owner, hooks=hooks, logger=logger)
    memento._recover()
    return memento
