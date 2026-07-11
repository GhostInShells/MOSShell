"""
FsMemento — FORMAT.md 的文件系统参考实现.

主权层代码: 可丢弃, jsonl 是唯一 truth. 本实现的索引全部在内存中即时重建,
不落盘 .cache/ (契约 §7 "删缓存行为不变" 的最平凡满足). 需要持久索引时
重做本文件即可, 不动契约.

并发模型: owner-isolated 单写者 (FORMAT.md §1). 跨 owner 只读走文件系统.
不内置进程间锁 — 跨进程共享 owner 时在外层仲裁.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ghoshell_moss.core.memento.abc import (
    TRAILER_KIND,
    TRAILER_RESUMES,
    TRAILER_SUSPENDS,
    TRAILER_THREAD,
    BasePointer,
    BranchMeta,
    BranchNotFoundError,
    BranchWindow,
    Commit,
    CommitKind,
    CommitNote,
    CommitNotFoundError,
    CommitView,
    EmptyStagingError,
    Memento,
    MementoBranch,
    MementoError,
    MementoHooks,
    MomentFrozenError,
    MomentPool,
    MomentRecord,
    NullHooks,
    ReadonlyBranchError,
    join_trailers,
)

__all__ = ["FsMomentPool", "FsMementoBranch", "FsMemento", "new_filesystem_memento"]

_MOMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
_OWNER_PATTERN = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


def _now() -> datetime:
    return datetime.now().astimezone()


def _dump_line(obj: dict[str, Any]) -> str:
    # FORMAT.md §2: UTF-8, ensure_ascii=False, 紧凑分隔, LF
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def _append_lines(path: Path, objs: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(_dump_line(o) for o in objs)
    with path.open("a", encoding="utf-8") as f:
        f.write(payload)


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """FORMAT.md §2: 撕裂尾行静默跳过, 非尾行解析失败 = 损坏, 抛错."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for i, line in enumerate(raw):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if i == len(raw) - 1:
                break
            raise MementoError(f"corrupted jsonl (non-tail unparsable line) at {path}:{i + 1}")
        if not isinstance(obj, dict):
            raise MementoError(f"corrupted jsonl (line is not an object) at {path}:{i + 1}")
        out.append(obj)
    return out


def _model_line(t: str, model: Any) -> dict[str, Any]:
    return {"t": t, **model.model_dump(mode="json", exclude_none=True)}


def _parse_model(model_cls: type, obj: dict[str, Any]) -> Any:
    return model_cls.model_validate({k: v for k, v in obj.items() if k != "t"})


# ────────────────────────────────────────────────────────────────────────────
# MomentPool
# ────────────────────────────────────────────────────────────────────────────


class FsMomentPool(MomentPool):
    """moments/{owner}/{YYYY-MM}/moments.jsonl (FORMAT.md §3)."""

    def __init__(self, root: Path):
        self._dir = root / "moments"
        self._file_of: dict[str, Path] = {}
        self._scanned = False

    def _scan(self) -> None:
        if self._scanned:
            return
        if self._dir.exists():
            for f in sorted(self._dir.glob("*/*/moments.jsonl")):
                for obj in _read_lines(f):
                    if obj.get("t") == "moment" and isinstance(obj.get("id"), str):
                        # 首次出现的文件是该 id 的家 (§3.2/§3.4: 覆盖与释义都追加到这里)
                        self._file_of.setdefault(obj["id"], f)
        self._scanned = True

    def put(self, record: MomentRecord, *, owner: str) -> None:
        self._scan()
        if not _MOMENT_ID_PATTERN.match(record.id):
            raise MementoError(f"invalid moment id: {record.id!r} (FORMAT.md §2.1)")
        if not _OWNER_PATTERN.match(owner):
            raise MementoError(f"invalid owner: {owner!r} (FORMAT.md §1)")
        path = self._file_of.get(record.id)
        if path is None:
            year_month = record.created.astimezone(timezone.utc).strftime("%Y-%m")
            path = self._dir / owner / year_month / "moments.jsonl"
        _append_lines(path, [_model_line("moment", record)])
        self._file_of.setdefault(record.id, path)

    def annotate(self, moment_id: str, threads: Sequence[str], *, owner: str, by: str = "") -> None:
        self._scan()
        path = self._file_of.get(moment_id)
        if path is None:
            raise MementoError(f"moment not found for annotate: {moment_id!r}")
        line: dict[str, Any] = {
            "t": "note",
            "ref": moment_id,
            "threads": list(threads),
            "ts": _now().isoformat(),
        }
        if by:
            line["by"] = by
        _append_lines(path, [line])

    def get(self, moment_id: str) -> MomentRecord | None:
        self._scan()
        path = self._file_of.get(moment_id)
        if path is None:
            return None
        return self._resolve_file(path).get(moment_id)

    def get_many(self, moment_ids: Iterable[str]) -> dict[str, MomentRecord]:
        self._scan()
        by_file: dict[Path, list[str]] = {}
        for mid in moment_ids:
            path = self._file_of.get(mid)
            if path is not None:
                by_file.setdefault(path, []).append(mid)
        out: dict[str, MomentRecord] = {}
        for path, ids in by_file.items():
            resolved = self._resolve_file(path)
            for mid in ids:
                if mid in resolved:
                    out[mid] = resolved[mid]
        return out

    @staticmethod
    def _resolve_file(path: Path) -> dict[str, MomentRecord]:
        """整文件 last-wins 解析: 记录行覆盖 + threads 释义整体替换 (FORMAT.md §2.2/§3.3)."""
        records: dict[str, MomentRecord] = {}
        for obj in _read_lines(path):
            t = obj.get("t")
            if t == "moment":
                record = _parse_model(MomentRecord, obj)
                records[record.id] = record
            elif t == "note":
                ref = obj.get("ref")
                if isinstance(ref, str) and ref in records:
                    threads = obj.get("threads")
                    if isinstance(threads, list):
                        records[ref] = records[ref].model_copy(update={"threads": list(threads)})
            # 未知 t: 跳过 (前向兼容, FORMAT.md §2)
        return records

    def close(self) -> None:
        pass


# ────────────────────────────────────────────────────────────────────────────
# Branch
# ────────────────────────────────────────────────────────────────────────────


def _load_meta(branch_dir: Path) -> BranchMeta:
    meta_path = branch_dir / "meta.json"
    if not meta_path.exists():
        raise BranchNotFoundError(f"branch meta not found: {meta_path}")
    return BranchMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))


def _save_meta(branch_dir: Path, meta: BranchMeta) -> None:
    branch_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(meta.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2)
    (branch_dir / "meta.json").write_text(text + "\n", encoding="utf-8")


def _commit_files(branch_dir: Path) -> list[Path]:
    d = branch_dir / "commits"
    if not d.exists():
        return []
    files = [p for p in d.iterdir() if p.suffix == ".jsonl" and p.stem.isdigit()]
    # FORMAT.md §5: 按解析后的整数排序, 不按文件名字典序
    return sorted(files, key=lambda p: int(p.stem))


def _load_commit_file(path: Path) -> tuple[Commit, list[CommitNote]]:
    objs = _read_lines(path)
    if not objs or objs[0].get("t") != "commit":
        raise MementoError(f"corrupted commit file (first line must be member line): {path}")
    commit: Commit = _parse_model(Commit, objs[0])
    notes: list[CommitNote] = []
    for obj in objs[1:]:
        if obj.get("t") == "note" and obj.get("ref") == commit.id:
            notes.append(_parse_model(CommitNote, obj))
    return commit, notes


def _view_of(commit: Commit, notes: list[CommitNote]) -> CommitView:
    if notes:
        return CommitView(commit=commit, note=notes[-1], note_seq=len(notes) - 1)
    # 容错: 初始释义行缺失 (写入中断的撕裂尾行). 合成空释义.
    return CommitView(commit=commit, note=CommitNote(ref=commit.id, body=""), note_seq=0)


class FsMementoBranch(MementoBranch):
    def __init__(
        self,
        branch_dir: Path,
        branches_root: Path,
        pool: FsMomentPool,
        hooks: MementoHooks,
        *,
        readonly: bool,
    ):
        self._dir = branch_dir
        self._branches_root = branches_root
        self._pool = pool
        self._hooks = hooks
        self._readonly = readonly
        self._meta = _load_meta(branch_dir)
        self._frozen_ids: set[str] | None = None

    @property
    def meta(self) -> BranchMeta:
        return self._meta

    @property
    def readonly(self) -> bool:
        return self._readonly

    def _check_writable(self) -> None:
        if self._readonly:
            raise ReadonlyBranchError(f"branch {self._meta.branch_id} is readonly for this handle")

    def _frozen(self) -> set[str]:
        if self._frozen_ids is None:
            frozen: set[str] = set()
            for path in _commit_files(self._dir):
                commit, _ = _load_commit_file(path)
                frozen.update(commit.moment_ids)
            self._frozen_ids = frozen
        return self._frozen_ids

    def _staging_path(self) -> Path:
        return self._dir / "staging.jsonl"

    def _staging_ids(self) -> list[str]:
        """去重保序 (保留首次出现的位置, FORMAT.md §4.2)."""
        seen: set[str] = set()
        ordered: list[str] = []
        for obj in _read_lines(self._staging_path()):
            if obj.get("t") != "stage":
                continue
            mid = obj.get("moment_id")
            if isinstance(mid, str) and mid not in seen:
                seen.add(mid)
                ordered.append(mid)
        return ordered

    # ── 写入路径 ──

    def update(self, record: MomentRecord) -> None:
        self._check_writable()
        if record.id in self._frozen():
            raise MomentFrozenError(
                f"moment {record.id!r} is frozen by a commit of branch {self._meta.branch_id}; "
                f"updates after freeze are illegal (FORMAT.md §3.2)"
            )
        self._pool.put(record, owner=self._meta.fork)
        if record.id not in self._staging_ids():
            _append_lines(
                self._staging_path(),
                [{"t": "stage", "moment_id": record.id, "ts": _now().isoformat()}],
            )
        self._hooks.on_record_staged(self._meta.branch_id, record)

    def annotate_moment(self, moment_id: str, threads: Sequence[str], *, by: str = "") -> None:
        self._check_writable()
        self._pool.annotate(moment_id, threads, owner=self._meta.fork, by=by)

    def commit(
        self,
        text: str = "",
        *,
        kind: CommitKind,
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        by: str = "",
    ) -> CommitView:
        self._check_writable()
        staged = self._staging_ids()
        if not staged:
            raise EmptyStagingError(f"staging of branch {self._meta.branch_id} is empty")
        seq = self._max_seq() + 1
        commit = Commit(seq=seq, moment_ids=staged)
        trailers: list[tuple[str, str]] = []
        trailers += [(TRAILER_THREAD, t) for t in threads]
        trailers += [(TRAILER_RESUMES, r) for r in resumes]
        trailers += [(TRAILER_SUSPENDS, s) for s in suspends]
        trailers += list(extra_trailers)
        trailers.append((TRAILER_KIND, kind))
        note = CommitNote(ref=commit.id, body=join_trailers(text, trailers), by=by)
        path = self._dir / "commits" / f"{seq:04d}.jsonl"
        # 成员行 + 初始释义行一次写入 (FORMAT.md §5)
        _append_lines(path, [_model_line("commit", commit), _model_line("note", note)])
        # truncate staging — 全格式唯一允许清空的文件 (FORMAT.md §4.2)
        self._staging_path().write_text("", encoding="utf-8")
        self._meta = self._meta.model_copy(update={"updated": _now()})
        _save_meta(self._dir, self._meta)
        self._frozen().update(staged)
        view = CommitView(commit=commit, note=note, note_seq=0)
        self._hooks.on_commit(self._meta.branch_id, view)
        return view

    def reinterpret(self, commit_id: str, body: str, *, by: str = "") -> CommitView:
        self._check_writable()
        for path in _commit_files(self._dir):
            commit, notes = _load_commit_file(path)
            if commit.id == commit_id:
                note = CommitNote(ref=commit_id, body=body, by=by)
                _append_lines(path, [_model_line("note", note)])
                view = CommitView(commit=commit, note=note, note_seq=len(notes))
                self._hooks.on_reinterpreted(self._meta.branch_id, view)
                return view
        raise CommitNotFoundError(
            f"commit {commit_id!r} is not an own commit of branch {self._meta.branch_id}; "
            f"祖先的释义归祖先的 owner"
        )

    # ── 读取路径 ──

    def _max_seq(self) -> int:
        files = _commit_files(self._dir)
        return int(files[-1].stem) if files else 0

    def head(self) -> CommitView | None:
        files = _commit_files(self._dir)
        if not files:
            return None
        return _view_of(*_load_commit_file(files[-1]))

    def staging(self) -> list[MomentRecord]:
        ids = self._staging_ids()
        found = self._pool.get_many(ids)
        return [found[mid] for mid in ids if mid in found]

    def own_commits(self) -> list[CommitView]:
        return [_view_of(*_load_commit_file(p)) for p in _commit_files(self._dir)]

    def _ancestor_views(self) -> list[CommitView]:
        views: list[CommitView] = []
        for bp in self._meta.ancestry:
            adir = self._branches_root / bp.fork / bp.branch_id
            for path in _commit_files(adir):
                if int(path.stem) <= bp.commit_seq:
                    views.append(_view_of(*_load_commit_file(path)))
        return views

    def all_commits(self) -> list[CommitView]:
        return self._ancestor_views() + self.own_commits()

    def get_commit(self, commit_id: str) -> CommitView | None:
        for view in reversed(self.all_commits()):
            if view.id == commit_id:
                return view
        return None

    def commit_records(self, commit_id: str) -> list[MomentRecord]:
        view = self.get_commit(commit_id)
        if view is None:
            raise CommitNotFoundError(
                f"commit {commit_id!r} not in history of branch {self._meta.branch_id}"
            )
        found = self._pool.get_many(view.commit.moment_ids)
        missing = [mid for mid in view.commit.moment_ids if mid not in found]
        if missing:
            raise MementoError(f"commit {commit_id!r} members missing from pool: {missing}")
        return [found[mid] for mid in view.commit.moment_ids]

    def notes(self, commit_id: str) -> list[CommitNote]:
        dirs = [self._dir] + [
            self._branches_root / bp.fork / bp.branch_id for bp in reversed(self._meta.ancestry)
        ]
        for branch_dir in dirs:
            for path in _commit_files(branch_dir):
                commit, notes = _load_commit_file(path)
                if commit.id == commit_id:
                    return notes
        raise CommitNotFoundError(
            f"commit {commit_id!r} not in history of branch {self._meta.branch_id}"
        )

    def window(self, *, detail_n: int = 10, summary_m: int = -1) -> BranchWindow:
        details: list[MomentRecord] = self.staging()[-detail_n:] if detail_n > 0 else []
        all_views = self.all_commits()
        boundary = len(all_views)
        need = (detail_n - len(details)) if detail_n > 0 else 0
        while need > 0 and boundary > 0:
            view = all_views[boundary - 1]
            found = self._pool.get_many(view.commit.moment_ids)
            expanded = [found[mid] for mid in view.commit.moment_ids if mid in found]
            taken = expanded[-need:]
            details = taken + details
            need -= len(taken) if taken else need
            boundary -= 1
        summaries = all_views[:boundary]
        if summary_m == 0:
            summaries = []
        elif summary_m > 0:
            summaries = summaries[-summary_m:]
        return BranchWindow(summaries=summaries, details=details)


# ────────────────────────────────────────────────────────────────────────────
# Memento facade
# ────────────────────────────────────────────────────────────────────────────


class FsMemento(Memento):
    """
    :param root: memento 根目录 (即 FORMAT.md §1 的 {root}/memento 本身).
    :param owner: 本实例绑定的 owner 命名空间.
    """

    def __init__(self, root: str | Path, owner: str, *, hooks: MementoHooks | None = None):
        if not _OWNER_PATTERN.match(owner):
            raise MementoError(f"invalid owner: {owner!r} (FORMAT.md §1)")
        self._root = Path(root)
        self._owner = owner
        self._hooks: MementoHooks = hooks if hooks is not None else NullHooks()
        self._pool = FsMomentPool(self._root)
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        gitignore = self._root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(".cache/\n", encoding="utf-8")

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def pool(self) -> FsMomentPool:
        """暴露池给 porcelain / 诊断. 写入仍应经 branch.update."""
        return self._pool

    def _branches_root(self) -> Path:
        return self._root / "branches"

    def _owner_dir(self) -> Path:
        return self._branches_root() / self._owner

    def _branch(self, fork: str, branch_id: str, *, readonly: bool) -> FsMementoBranch:
        branch_dir = self._branches_root() / fork / branch_id
        branch = FsMementoBranch(
            branch_dir, self._branches_root(), self._pool, self._hooks, readonly=readonly
        )
        self._validate_ancestry(branch.meta)
        return branch

    def _validate_ancestry(self, meta: BranchMeta) -> None:
        """FORMAT.md §4.1: ancestry == base_branch.ancestry + [base]. 不一致 MUST 抛错."""
        if meta.base is None:
            if meta.ancestry:
                raise MementoError(f"root branch {meta.branch_id} has non-empty ancestry")
            return
        if not meta.ancestry or meta.ancestry[-1] != meta.base:
            raise MementoError(f"branch {meta.branch_id}: ancestry[-1] != base")
        base_meta = _load_meta(self._branches_root() / meta.base.fork / meta.base.branch_id)
        if meta.ancestry[:-1] != base_meta.ancestry:
            raise MementoError(
                f"branch {meta.branch_id}: frozen ancestry diverges from base chain (FORMAT.md §4.1)"
            )

    # ── 当前指针 ──

    def _head_path(self) -> Path:
        return self._owner_dir() / "HEAD.json"

    def _write_head(self, branch_id: str) -> None:
        self._owner_dir().mkdir(parents=True, exist_ok=True)
        self._head_path().write_text(
            json.dumps({"current": branch_id}, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def current(self) -> MementoBranch:
        head_path = self._head_path()
        if head_path.exists():
            branch_id = json.loads(head_path.read_text(encoding="utf-8"))["current"]
            return self._branch(self._owner, branch_id, readonly=False)
        meta = BranchMeta(fork=self._owner, name="main")
        _save_meta(self._owner_dir() / meta.branch_id, meta)
        self._write_head(meta.branch_id)
        self._hooks.on_branch_created(meta)
        return self._branch(self._owner, meta.branch_id, readonly=False)

    def switch(self, branch_id: str) -> None:
        if not (self._owner_dir() / branch_id / "meta.json").exists():
            raise BranchNotFoundError(f"branch {branch_id!r} not found under owner {self._owner!r}")
        self._write_head(branch_id)
        self._hooks.on_branch_switched(branch_id)

    # ── fork 边界: 化身只能从 commit 出生 ──

    def checkout(
        self,
        *,
        base_fork: str,
        base_branch_id: str,
        base_commit_id: str | None = None,
        name: str = "",
        overlay: dict[str, Any] | None = None,
    ) -> MementoBranch:
        source = self._branch(base_fork, base_branch_id, readonly=True)
        base = self._locate_base(source, base_commit_id)
        base_meta = _load_meta(self._branches_root() / base.fork / base.branch_id)
        meta = BranchMeta(
            fork=self._owner,
            name=name,
            base=base,
            # fork 时刻冻结展平祖先链: 恒等式 ancestry == base 的 ancestry + [base]
            ancestry=list(base_meta.ancestry) + [base],
            overlay=overlay or {},
        )
        _save_meta(self._owner_dir() / meta.branch_id, meta)
        self._hooks.on_branch_created(meta)
        return self._branch(self._owner, meta.branch_id, readonly=False)

    def _locate_base(self, source: FsMementoBranch, base_commit_id: str | None) -> BasePointer:
        if base_commit_id is None:
            head = source.head()
            if head is None:
                raise MementoError(
                    f"source branch {source.meta.branch_id} has no commit; "
                    f"化身只能从 commit 出生, 永不从 staging"
                )
            return BasePointer(
                fork=source.meta.fork,
                branch_id=source.meta.branch_id,
                commit_id=head.id,
                commit_seq=head.seq,
            )
        for view in source.own_commits():
            if view.id == base_commit_id:
                return BasePointer(
                    fork=source.meta.fork,
                    branch_id=source.meta.branch_id,
                    commit_id=view.id,
                    commit_seq=view.seq,
                )
        # 起点落在祖先段: base 直接指向那个祖先 branch
        for bp in source.meta.ancestry:
            adir = self._branches_root() / bp.fork / bp.branch_id
            for path in _commit_files(adir):
                if int(path.stem) > bp.commit_seq:
                    continue
                commit, _ = _load_commit_file(path)
                if commit.id == base_commit_id:
                    return BasePointer(
                        fork=bp.fork,
                        branch_id=bp.branch_id,
                        commit_id=commit.id,
                        commit_seq=commit.seq,
                    )
        raise CommitNotFoundError(
            f"commit {base_commit_id!r} not in history of branch {source.meta.branch_id}"
        )

    # ── 浏览 ──

    def get_branch(self, branch_id: str, *, fork: str | None = None) -> MementoBranch:
        fork = fork if fork is not None else self._owner
        return self._branch(fork, branch_id, readonly=fork != self._owner)

    def list_branches(self, fork: str | None = None) -> list[BranchMeta]:
        fork_dir = self._branches_root() / (fork if fork is not None else self._owner)
        if not fork_dir.exists():
            return []
        return [
            _load_meta(child)
            for child in sorted(fork_dir.iterdir())
            if child.is_dir() and (child / "meta.json").exists()
        ]

    def list_forks(self) -> list[str]:
        root = self._branches_root()
        if not root.exists():
            return []
        return sorted(child.name for child in root.iterdir() if child.is_dir())

    def close(self) -> None:
        self._pool.close()


def new_filesystem_memento(
    root: str | Path, owner: str, *, hooks: MementoHooks | None = None
) -> FsMemento:
    return FsMemento(root, owner, hooks=hooks)
