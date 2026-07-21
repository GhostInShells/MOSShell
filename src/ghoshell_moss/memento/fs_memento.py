"""
FsMemento — Memento FORMAT v2 的文件系统参考实现.

per-owner 分片 jsonl, filesystem-first. jsonl 是唯一 truth, .cache/ 不存在.
commit 原子动作序列见 FORMAT.md §11, 崩溃恢复在 line 装入时执行.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ulid import ULID

from ghoshell_moss.memento.abc import (
    BranchRef,
    BranchWindow,
    Commit,
    CommitDetail,
    CommitNote,
    CommitNotFoundError,
    CommitRef,
    CommitView,
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
    new_commit_id,
)


# ── 磁盘格式常量 (FORMAT.md v2) ─────────────────────────────────────────────

_MEMENTO_DIR = "memento"
_COMMITS_DIR = "commits"
_BRANCHES_DIR = "branches"
_REF_FILE = "ref"
_STAGING_FILE = "staging.jsonl"
_COMMITS_JSONL = "commits.jsonl"
_META_JSON = "meta.json"
_MOMENTS_JSONL = "moments.jsonl"
_NOTES_JSONL = "notes.jsonl"

_T_MOMENT = "moment"
_T_MOMENT_NOTE = "moment_note"
_T_COMMIT_NOTE = "commit_note"
_T_COMMIT_REF = "commit_ref"
_T_COMMIT = "commit"

_F_T = "t"
_F_ID = "id"
_F_REF = "ref"
_F_THREADS = "threads"
_F_TITLE = "title"
_F_BODY = "body"
_F_TS = "ts"
_F_BY = "by"
_F_CREATED = "created"
_F_PARENT = "parent"
_F_BRANCH = "branch"
_F_KIND = "kind"
_F_COMMIT_ID = "commit_id"
_F_MOMENT_IDS = "moment_ids"
_F_ORIGIN = "origin"

_KIND_SEMANTIC = "semantic"
_KIND_MECHANICAL = "mechanical"
_CMT_PREFIX = "cmt_"

_TMP_SUFFIX = ".tmp"
_TMP_PREFIX = ".tmp_"


# ── 工具 ─────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now().astimezone()


def _y_m(commit_id: str) -> str:
    """commit_id → Y-m (FORMAT.md §5.0). ULID 时间戳解码, UTC."""
    from datetime import timezone as tz

    ulid = ULID.from_str(commit_id[4:])
    return datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")


def _dump(obj: dict[str, Any]) -> str:
    """单行 JSON, 紧凑分隔 (见证层 diff 最小)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """读 jsonl, 跳过撕裂尾行 (FORMAT.md §2). 中段损坏抛错."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    result: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                pass  # torn last line
            else:
                raise MementoError(f"corrupt jsonl at line {i+1} in {path}")
    return result


def _append_lines(path: Path, objs: Sequence[dict[str, Any]]) -> None:
    """追加行到 jsonl. 创建父目录若需."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for obj in objs:
            f.write(_dump(obj) + "\n")


def _write_atomic(path: Path, content: str) -> None:
    """原子写: tmp + os.replace. 创建父目录若需."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + _TMP_SUFFIX)
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _fsync_path(path: Path) -> None:
    """fsync 单个文件或目录."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ── staging 解析 ─────────────────────────────────────────────────────────────


def _resolve_staging(staging_path: Path) -> tuple[list[str], dict[str, MomentRecord]]:
    """
    读 staging.jsonl → (id 首现序, id → last-wins MomentRecord).
    应用 moment_note (整体替换 threads).
    """
    order: list[str] = []
    records: dict[str, MomentRecord] = {}
    if not staging_path.exists():
        return order, records
    for obj in _read_lines(staging_path):
        t = obj.get("t")
        if t == _T_MOMENT:
            rid = obj["id"]
            if rid not in records:
                order.append(rid)
            records[rid] = MomentRecord(**obj)
        elif t == _T_MOMENT_NOTE:
            ref = obj.get("ref")
            if ref and ref in records:
                records[ref].threads = obj.get(_F_THREADS, [])
    return order, records


# ── FsLine ───────────────────────────────────────────────────────────────────


class FsLine:
    """Line Protocol 参考实现. 所有操作委托到 FsMemento."""

    __slots__ = ("_memento", "_name", "_readonly")

    def __init__(self, memento: FsMemento, name: str, readonly: bool = False):
        self._memento = memento
        self._name = name
        self._readonly = readonly

    @property
    def name(self) -> str:
        return self._name

    @property
    def ref(self) -> BranchRef | None:
        return self._memento._read_ref(self._name)

    @property
    def readonly(self) -> bool:
        return self._readonly

    def __repr__(self) -> str:
        return f"<FsLine {self._name!r} ro={self._readonly}>"

    # ── 延线 ──

    def record(self, record: MomentRecord) -> None:
        if self._readonly:
            raise ReadonlyLineError(f"line {self._name!r} is readonly")
        self._memento._record(self._name, record)

    def commit(
        self,
        text: str = "",
        *,
        kind: str = _KIND_SEMANTIC,
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        boundary_moment_id: str | None = None,
        by: str = "",
    ) -> CommitView:
        if self._readonly:
            raise ReadonlyLineError(f"line {self._name!r} is readonly")
        return self._memento._commit(
            self._name,
            text=text,
            kind=kind,
            threads=threads,
            resumes=resumes,
            suspends=suspends,
            extra_trailers=extra_trailers,
            boundary_moment_id=boundary_moment_id,
            by=by,
        )

    # ── 读 ──

    def staging(self) -> list[MomentRecord]:
        _, records = self._memento._resolve_staging_of(self._name)
        return list(records.values())

    def log(self) -> list[CommitView]:
        return self._memento._line_log(self._name)

    def window(self, *, detail_n: int = 10, summary_m: int = -1) -> BranchWindow:
        return self._memento._line_window(self._name, detail_n=detail_n, summary_m=summary_m)


# ── FsMemento ────────────────────────────────────────────────────────────────


class FsMemento(MementoABC):
    """Memento v2 参考实现."""

    def __init__(self, root: str | Path, owner: str, hooks: MementoHooks | None = None):
        self._root = Path(root)
        self._owner = owner
        self._hooks = hooks or NullHooks()

    @property
    def owner(self) -> str:
        return self._owner

    # ── 内部路径 ──

    def _owner_dir(self) -> Path:
        return self._root / self._owner

    def _branch_dir(self, name: str) -> Path:
        return self._owner_dir() / "branches" / name

    def _ref_path(self, name: str) -> Path:
        return self._branch_dir(name) / "ref"

    def _staging_path(self, name: str) -> Path:
        return self._branch_dir(name) / _STAGING_FILE

    def _commit_dir_for(self, owner: str, commit_id: str) -> Path:
        """commit 自治目录路径，可指定 owner（跨 owner 查阅）."""
        if not commit_id.startswith(_CMT_PREFIX):
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        try:
            ym = _y_m(commit_id)
        except (ValueError, IndexError):
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        return self._root / owner / "commits" / ym / commit_id

    def _commit_dir(self, commit_id: str) -> Path:
        """commit 自治目录路径 (本 owner)."""
        return self._commit_dir_for(self._owner, commit_id)

    def _commits_jsonl(self) -> Path:
        return self._owner_dir() / _COMMITS_JSONL

    def _meta_json(self) -> Path:
        return self._owner_dir() / _META_JSON

    # ── ref 读写 ──

    def _read_ref(self, name: str) -> BranchRef | None:
        path = self._ref_path(name)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            return None
        return BranchRef(**data)

    def _write_ref(self, name: str, ref: BranchRef) -> None:
        self._ref_path(name).parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(self._ref_path(name), _dump(ref.model_dump(mode="json")))

    # ── staging ──

    def _resolve_staging_of(self, name: str) -> tuple[list[str], dict[str, MomentRecord]]:
        return _resolve_staging(self._staging_path(name))

    def _frozen_ids(self, name: str) -> set[str]:
        """沿 parent 链收集本 branch 所有 commit 中的已冻结 moment id."""
        frozen: set[str] = set()
        ref = self._read_ref(name)
        cid = ref.commit_id if ref else None
        while cid:
            mp = self._commit_dir(cid) / _MOMENTS_JSONL
            if mp.exists():
                for obj in _read_lines(mp):
                    if obj.get("t") == _T_MOMENT:
                        frozen.add(obj["id"])
            meta = self._load_commit_meta(cid)
            p = meta.get("parent") if meta else None
            cid = p["commit_id"] if p else None
        return frozen

    def _record(self, name: str, record: MomentRecord) -> None:
        staging_path = self._staging_path(name)
        if not staging_path.parent.exists():
            raise LineNotFoundError(
                f"line {name!r} not found for owner {self._owner!r}: branch dir missing"
            )
        if record.id in self._frozen_ids(name):
            raise MomentFrozenError(f"moment {record.id!r} is frozen")
        _append_lines(staging_path, [{"t": _T_MOMENT, **record.model_dump(mode="json")}])
        self._hooks.on_record_staged(name, record)

    # ── commit ──

    def _commit(
        self,
        name: str,
        *,
        text: str = "",
        kind: str = _KIND_SEMANTIC,
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        boundary_moment_id: str | None = None,
        by: str = "",
    ) -> CommitView:
        order, records = self._resolve_staging_of(name)
        if not records:
            raise EmptyStagingError(f"line {name!r} staging is empty")

        # boundary: 只冻结前缀
        if boundary_moment_id is not None:
            if boundary_moment_id not in order:
                raise MementoError(
                    f"boundary_moment_id {boundary_moment_id!r} not in staging"
                )
            idx = order.index(boundary_moment_id)
            frozen_order = order[: idx + 1]
            remaining_order = order[idx + 1 :]
        else:
            frozen_order = list(order)
            remaining_order = []

        # 组装 trailer → body
        pairs: list[tuple[str, str]] = [("Kind", kind)]
        for t in threads:
            pairs.append(("Thread", t))
        for rid in resumes:
            pairs.append(("Resumes", rid))
        for s in suspends:
            pairs.append(("Suspends", s))
        pairs.extend(extra_trailers)
        body = join_trailers(text, pairs)

        commit_id = new_commit_id()
        ym = _y_m(commit_id)
        now = _now()

        # parent = 当前 ref
        parent_ref = None
        cur = self._read_ref(name)
        if cur is not None:
            parent_ref = BranchRef(origin=cur.origin, commit_id=cur.commit_id, moment_id=cur.moment_id)

        # 冻结 moments
        frozen_moments = [records[rid] for rid in frozen_order]

        # tmp 目录
        commit_parent = self._owner_dir() / "commits" / ym
        commit_parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=str(commit_parent), prefix=_TMP_PREFIX))

        try:
            # meta.json
            meta = {
                _F_COMMIT_ID: commit_id,
                _F_PARENT: parent_ref.model_dump(mode="json") if parent_ref else None,
                _F_BRANCH: name,
                "kind": kind,
                _F_CREATED: now.isoformat(),
            }
            (tmp_dir / _META_JSON).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # moments.jsonl: 成员行 + 冻结 moment 行
            moment_objs: list[dict[str, Any]] = [
                {"t": "commit", "id": commit_id, _F_MOMENT_IDS: frozen_order, _F_CREATED: now.isoformat()}
            ]
            for mr in frozen_moments:
                moment_objs.append({"t": _T_MOMENT, **mr.model_dump(mode="json")})
            (tmp_dir / _MOMENTS_JSONL).write_text(
                "\n".join(_dump(o) for o in moment_objs) + "\n", encoding="utf-8"
            )

            # notes.jsonl: 初始 commit_note
            title = text.split("\n")[0].strip() if text else ""
            note_obj = {
                "t": _T_COMMIT_NOTE,
                "ref": commit_id,
                _F_TITLE: title,
                _F_BODY: body,
                "ts": now.isoformat(),
                "by": by,
            }
            (tmp_dir / _NOTES_JSONL).write_text(_dump(note_obj) + "\n", encoding="utf-8")

            # fsync tmp
            for p in tmp_dir.iterdir():
                _fsync_path(p)
            _fsync_path(tmp_dir)

            # 原子 rename
            target_dir = commit_parent / commit_id
            os.rename(str(tmp_dir), str(target_dir))

            # append commits.jsonl
            _append_lines(self._commits_jsonl(), [{
                "t": _T_COMMIT_REF,
                _F_COMMIT_ID: commit_id,
                _F_BRANCH: name,
                _F_PARENT: parent_ref.model_dump(mode="json") if parent_ref else None,
                "ts": now.isoformat(),
                "kind": kind,
            }])
            if self._commits_jsonl().exists():
                _fsync_path(self._commits_jsonl())

            # rewrite ref
            self._write_ref(name, BranchRef(origin=self._owner, commit_id=commit_id))

            # truncate staging
            staging_path = self._staging_path(name)
            if remaining_order:
                remaining_lines: list[dict[str, Any]] = []
                for rid in remaining_order:
                    remaining_lines.append({"t": _T_MOMENT, **records[rid].model_dump(mode="json")})
                staging_path.write_text(
                    "\n".join(_dump(o) for o in remaining_lines) + "\n", encoding="utf-8"
                )
            else:
                staging_path.write_text("", encoding="utf-8")

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise

        commit = Commit(id=commit_id, created=meta[_F_CREATED])
        note = CommitNote(ref=commit_id, title=title, body=body, ts=now, by=by)
        view = CommitView(commit=commit, note=note, note_seq=0)
        self._hooks.on_commit(name, view)
        return view

    # ── line 历史 ──

    def _line_log(self, name: str) -> list[CommitView]:
        result: list[CommitView] = []
        ref = self._read_ref(name)
        cid = ref.commit_id if ref else None
        while cid:
            view = self._load_commit_view(cid)
            if view is None:
                break
            result.append(view)
            meta = self._load_commit_meta(cid)
            p = meta.get(_F_PARENT) if meta else None
            cid = p[_F_COMMIT_ID] if p else None
        result.reverse()
        return result

    def _line_window(self, name: str, *, detail_n: int, summary_m: int) -> BranchWindow:
        order, records = self._resolve_staging_of(name)
        staging_list = [records[rid] for rid in order]
        details = staging_list[-detail_n:] if detail_n > 0 else []

        summaries: list[CommitView] = []
        ref = self._read_ref(name)
        cid = ref.commit_id if ref else None
        while cid and (summary_m == -1 or len(summaries) < summary_m):
            view = self._load_commit_view(cid)
            if view is None:
                break
            summaries.append(view)
            meta = self._load_commit_meta(cid)
            p = meta.get(_F_PARENT) if meta else None
            cid = p[_F_COMMIT_ID] if p else None
        summaries.reverse()

        return BranchWindow(summaries=summaries, details=details)

    # ── commit 加载 ──

    def _load_commit_meta(self, commit_id: str) -> dict[str, Any] | None:
        path = self._commit_dir(commit_id) / _META_JSON
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_commit_view(self, commit_id: str) -> CommitView | None:
        commit_dir = self._commit_dir(commit_id)
        if not commit_dir.exists():
            return None
        moments_path = commit_dir / _MOMENTS_JSONL
        if not moments_path.exists():
            return None
        mom_lines = _read_lines(moments_path)
        if not mom_lines or mom_lines[0].get("t") != "commit":
            return None
        member = mom_lines[0]
        commit = Commit(id=member["id"], created=member[_F_CREATED])

        notes = _read_lines(commit_dir / _NOTES_JSONL) if (commit_dir / _NOTES_JSONL).exists() else []
        cn_notes = [n for n in notes if n.get("t") == _T_COMMIT_NOTE]
        if cn_notes:
            last = cn_notes[-1]
            note = CommitNote(
                ref=last["ref"],
                title=last.get(_F_TITLE, ""),
                body=last.get(_F_BODY, ""),
                ts=last.get("ts", _now()),
                by=last.get("by", ""),
            )
            note_seq = len(cn_notes) - 1
        else:
            note = CommitNote(ref=commit_id)
            note_seq = 0

        return CommitView(commit=commit, note=note, note_seq=note_seq)

    def _load_commit_moments(self, commit_id: str) -> list[MomentRecord]:
        path = self._commit_dir(commit_id) / _MOMENTS_JSONL
        if not path.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        result: list[MomentRecord] = []
        for obj in _read_lines(path):
            if obj.get("t") == _T_MOMENT:
                result.append(MomentRecord(**obj))
        # apply moment_note from notes.jsonl (appended after freeze)
        notes_path = self._commit_dir(commit_id) / _NOTES_JSONL
        if notes_path.exists():
            for obj in _read_lines(notes_path):
                if obj.get("t") == _T_MOMENT_NOTE:
                    ref = obj.get("ref")
                    for mr in result:
                        if mr.id == ref:
                            mr.threads = obj.get(_F_THREADS, [])
        return result

    def _find_moment_commit(self, moment_id: str) -> str | None:
        """grep commits/ 找 moment 所在 commit (FORMAT.md §18.3)."""
        commits_root = self._owner_dir() / "commits"
        if not commits_root.exists():
            return None
        for ym_dir in sorted(commits_root.iterdir()):
            if not ym_dir.is_dir():
                continue
            for commit_dir in sorted(ym_dir.iterdir()):
                mp = commit_dir / _MOMENTS_JSONL
                if not mp.exists():
                    continue
                for obj in _read_lines(mp):
                    if obj.get("t") == _T_MOMENT and obj.get("id") == moment_id:
                        return commit_dir.name
        return None

    # ── 公共接口 ──

    def create_line(
        self, name: str, *, from_ref: BranchRef | None = None, overlay: dict[str, Any] | None = None
    ) -> FsLine:
        branch_dir = self._branch_dir(name)
        if branch_dir.exists():
            raise MementoError(f"line {name!r} already exists for owner {self._owner!r}")
        if from_ref is not None:
            ref_owner = from_ref.origin
            if not self._commit_dir_for(ref_owner, from_ref.commit_id).exists():
                raise CommitNotFoundError(f"from_ref commit {from_ref.commit_id!r} not found")
        branch_dir.mkdir(parents=True)
        if from_ref is not None:
            self._write_ref(name, from_ref)
        if overlay is not None:
            mp = self._meta_json()
            meta: dict[str, Any] = {}
            if mp.exists():
                meta = json.loads(mp.read_text(encoding="utf-8"))
            now = _now()
            meta.setdefault("owner", self._owner)
            meta.setdefault(_F_CREATED, now.isoformat())
            meta["overlay"] = overlay
            meta["updated"] = now.isoformat()
            _write_atomic(mp, json.dumps(meta, ensure_ascii=False, indent=2))
        self._hooks.on_line_created(name, from_ref)
        return FsLine(self, name)

    def get_line(self, name: str, *, origin: str | None = None) -> FsLine:
        target_owner = origin if origin is not None else self._owner
        if target_owner != self._owner:
            other_dir = self._root / target_owner / "branches" / name
            if not other_dir.exists():
                raise LineNotFoundError(f"line {name!r} not found for owner {target_owner!r}")
            return FsLine(self, name, readonly=True)
        if not self._branch_dir(name).exists():
            raise LineNotFoundError(f"line {name!r} not found for owner {self._owner!r}")
        self._recover(name)
        return FsLine(self, name)

    def list_lines(self) -> list[str]:
        d = self._owner_dir() / "branches"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def delete_line(self, name: str) -> None:
        d = self._branch_dir(name)
        if not d.exists():
            raise LineNotFoundError(f"line {name!r} not found")
        shutil.rmtree(str(d))
        self._hooks.on_line_deleted(name)

    def reset_line(self, name: str, to: BranchRef) -> None:
        if not self._branch_dir(name).exists():
            raise LineNotFoundError(f"line {name!r} not found")
        _, records = self._resolve_staging_of(name)
        if records:
            self._commit(name, text="", kind=_KIND_MECHANICAL, by="memento.reset")
        self._write_ref(name, to)

    def show(self, commit_id: str) -> CommitDetail:
        if not self._commit_dir(commit_id).exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        moments = self._load_commit_moments(commit_id)
        all_notes = self.notes(commit_id)
        member = self._load_commit_meta(commit_id)
        commit = Commit(id=commit_id, created=member[_F_CREATED] if member else _now())
        return CommitDetail(commit=commit, moments=moments, notes=all_notes)

    def notes(self, commit_id: str) -> list[CommitNote]:
        d = self._commit_dir(commit_id)
        if not d.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        np = d / _NOTES_JSONL
        if not np.exists():
            return []
        result: list[CommitNote] = []
        for obj in _read_lines(np):
            if obj.get("t") == _T_COMMIT_NOTE:
                result.append(CommitNote(
                    ref=obj["ref"],
                    title=obj.get(_F_TITLE, ""),
                    body=obj.get(_F_BODY, ""),
                    ts=obj.get("ts", _now()),
                    by=obj.get("by", ""),
                ))
        return result

    def annotate(self, commit_id: str, title: str = "", body: str = "", *, by: str = "") -> CommitView:
        d = self._commit_dir(commit_id)
        if not d.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        _append_lines(d / _NOTES_JSONL, [{
            "t": _T_COMMIT_NOTE,
            "ref": commit_id,
            _F_TITLE: title,
            _F_BODY: body,
            "ts": _now().isoformat(),
            "by": by,
        }])
        view = self._load_commit_view(commit_id)
        if view is None:
            raise MementoError(f"failed to reload commit {commit_id!r}")
        self._hooks.on_reinterpreted(commit_id, view)
        return view

    def annotate_moment(
        self, commit_id: str, moment_id: str, threads: Sequence[str], *, by: str = ""
    ) -> None:
        # 检查 staging
        for name in self.list_lines():
            order, records = self._resolve_staging_of(name)
            if moment_id in records:
                _append_lines(self._staging_path(name), [{
                    "t": _T_MOMENT_NOTE, "ref": moment_id,
                    _F_THREADS: list(threads), "ts": _now().isoformat(), "by": by,
                }])
                return
        # 已冻结
        found = self._find_moment_commit(moment_id)
        if found is None:
            raise MomentNotInCommitError(f"moment {moment_id!r} not found")
        _append_lines(self._commit_dir(found) / _NOTES_JSONL, [{
            "t": _T_MOMENT_NOTE, "ref": moment_id,
            _F_THREADS: list(threads), "ts": _now().isoformat(), "by": by,
        }])

    def log(self) -> list[CommitRef]:
        result: list[CommitRef] = []
        for obj in _read_lines(self._commits_jsonl()):
            if obj.get("t") != _T_COMMIT_REF:
                continue
            p = obj.get(_F_PARENT)
            parent = BranchRef(**p) if p else None
            result.append(CommitRef(
                commit_id=obj[_F_COMMIT_ID],
                branch=obj.get(_F_BRANCH, ""),
                parent=parent,
                ts=obj.get("ts", _now()),
                kind=obj.get("kind", _KIND_SEMANTIC),
            ))
        return result

    def commit_space(self, commit_id: str) -> str:
        d = self._commit_dir(commit_id)
        if not d.exists():
            raise CommitNotFoundError(f"commit {commit_id!r} not found")
        return str(d.resolve())

    # ── 崩溃恢复 ──

    def _recover(self, name: str) -> None:
        """FORMAT.md §11 崩溃恢复. get_line 时自动执行."""
        cjl = self._commits_jsonl()
        if not cjl.exists():
            return
        lines = _read_lines(cjl)
        if not lines:
            return
        last = lines[-1]
        if last.get("t") != _T_COMMIT_REF:
            return

        cid = last[_F_COMMIT_ID]
        commit_dir = self._commit_dir(cid)
        if not commit_dir.exists():
            self._truncate_commits_jsonl_tail()
            return

        # 补完 ref — 只在缺失或指向不存在 commit 时修复
        cur = self._read_ref(name)
        if cur is not None and self._commit_dir(cur.commit_id).exists():
            return  # ref 有效, 不动 (可能是 reset 故意移走的)
        if cur is None or cur.commit_id != cid:
            self._write_ref(name, BranchRef(origin=self._owner, commit_id=cid))

        # 清理 staging 残留
        sp = self._staging_path(name)
        if sp.exists():
            _, recs = self._resolve_staging_of(name)
            frozen = self._frozen_ids(name)
            if recs and all(rid in frozen for rid in recs):
                sp.write_text("", encoding="utf-8")

    def _truncate_commits_jsonl_tail(self) -> None:
        cjl = self._commits_jsonl()
        if not cjl.exists():
            return
        raw = cjl.read_text(encoding="utf-8")
        ls = raw.rstrip("\n").split("\n")
        while ls:
            last_line = ls[-1].strip()
            if not last_line:
                ls.pop()
                continue
            try:
                obj = json.loads(last_line)
                if obj.get("t") == _T_COMMIT_REF:
                    ls.pop()
                    break
                break
            except json.JSONDecodeError:
                ls.pop()
                continue
        cjl.write_text("\n".join(ls) + ("\n" if ls else ""), encoding="utf-8")


def new_filesystem_memento(
    root: str | Path, owner: str, hooks: MementoHooks | None = None
) -> FsMemento:
    return FsMemento(root, owner, hooks)
