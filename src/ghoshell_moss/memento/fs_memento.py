"""
FsMemento — FORMAT.md v1.1 的文件系统参考实现.

主权层代码: 可丢弃, jsonl 是唯一 truth. 本实现的索引全部在内存中即时重建,
不落盘 .cache/ (契约 §7 "删缓存行为不变" 的最平凡满足). 需要持久索引时
重做本文件即可, 不动契约.

布局速览 (FORMAT.md §1 / §14):
- 无独立 moment 池 — moment 记录的物理归属 = staging 或某个 commit 文件.
- staging.jsonl 直接容纳 moment 真身 (t:"moment") + moment 级释义 (t:"moment_note").
- commit 文件 (commits/NNNN.jsonl) 自包含: 成员行 + m 个冻结 moment 行 +
  commit 释义 + 后续追加的 moment 级释义.
- 崩溃恢复 (§12): 装入 branch 时判定 staging 与最大 seq commit 文件的一致性.

并发模型: owner-isolated 单写者 (FORMAT.md §1). 跨 owner 只读走文件系统.
不内置进程间锁 — 跨进程共享 owner 时在外层仲裁.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from ghoshell_moss.memento.abc import (
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
    MomentNotInCommitError,
    MomentRecord,
    NullHooks,
    ReadonlyBranchError,
    join_trailers,
)

__all__ = ["FsMementoBranch", "FsMemento", "new_filesystem_memento"]

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


def _fsync_file(path: Path) -> None:
    """FORMAT.md §12: commit 文件写入后 MUST fsync, 保恢复规则的原子性."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _resolve_staging(staging_path: Path) -> tuple[list[str], dict[str, MomentRecord]]:
    """
    读 staging.jsonl -> (id 首现序, id -> 最新 record).

    - t:"moment": 同 id last-wins (覆盖后写入 record 全字段).
    - t:"moment_note": ref 指向同文件 record 时, 整体替换其 threads.
    - 未知 t: 跳过 (前向兼容, FORMAT.md §2).
    """
    order: list[str] = []
    seen: set[str] = set()
    records: dict[str, MomentRecord] = {}
    for obj in _read_lines(staging_path):
        t = obj.get("t")
        if t == "moment":
            mid = obj.get("id")
            if not isinstance(mid, str):
                continue
            records[mid] = _parse_model(MomentRecord, obj)
            if mid not in seen:
                seen.add(mid)
                order.append(mid)
        elif t == "moment_note":
            ref = obj.get("ref")
            if isinstance(ref, str) and ref in records:
                threads = obj.get("threads")
                if isinstance(threads, list):
                    records[ref] = records[ref].model_copy(update={"threads": list(threads)})
    return order, records


def _load_commit_file(
    path: Path,
) -> tuple[Commit, list[MomentRecord], list[CommitNote]]:
    """
    读一个 commit 文件 -> (成员行, 冻结 moment 列表 (按 moment_ids 序, 应用 moment_note last-wins),
    commit 释义列表 (按行序即版本序)).

    结构 (FORMAT.md §5): 第 1 行 t:"commit" -> m 行 t:"moment" (冻结) ->
    追加的 t:"commit_note" 与 t:"moment_note" 混合 (按行序 last-wins).
    """
    objs = _read_lines(path)
    if not objs or objs[0].get("t") != "commit":
        raise MementoError(f"corrupted commit file (first line must be member line): {path}")
    commit: Commit = _parse_model(Commit, objs[0])
    frozen: dict[str, MomentRecord] = {}
    commit_notes: list[CommitNote] = []
    for obj in objs[1:]:
        t = obj.get("t")
        if t == "moment":
            record = _parse_model(MomentRecord, obj)
            if record.id in commit.moment_ids:
                frozen[record.id] = record
        elif t == "commit_note":
            if obj.get("ref") == commit.id:
                commit_notes.append(_parse_model(CommitNote, obj))
        elif t == "moment_note":
            ref = obj.get("ref")
            if isinstance(ref, str) and ref in frozen:
                threads = obj.get("threads")
                if isinstance(threads, list):
                    frozen[ref] = frozen[ref].model_copy(update={"threads": list(threads)})
        # 未知 t / 悬空 ref: 跳过 (前向兼容 + 撕裂容错)
    ordered_records = [frozen[mid] for mid in commit.moment_ids if mid in frozen]
    return commit, ordered_records, commit_notes


def _view_of(commit: Commit, notes: list[CommitNote]) -> CommitView:
    if notes:
        return CommitView(commit=commit, note=notes[-1], note_seq=len(notes) - 1)
    # 容错: 初始释义行缺失 (写入中断). 合成空释义, 不阻塞读路径.
    return CommitView(commit=commit, note=CommitNote(ref=commit.id, body=""), note_seq=0)


def _commit_files(branch_dir: Path) -> list[Path]:
    d = branch_dir / "commits"
    if not d.exists():
        return []
    files = [p for p in d.iterdir() if p.suffix == ".jsonl" and p.stem.isdigit()]
    # FORMAT.md §5: 按解析后的整数排序, 不按文件名字典序
    return sorted(files, key=lambda p: int(p.stem))


def _load_meta(branch_dir: Path) -> BranchMeta:
    meta_path = branch_dir / "meta.json"
    if not meta_path.exists():
        raise BranchNotFoundError(f"branch meta not found: {meta_path}")
    return BranchMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))


def _save_meta(branch_dir: Path, meta: BranchMeta) -> None:
    branch_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(meta.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2)
    (branch_dir / "meta.json").write_text(text + "\n", encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────────
# Branch — staging 持真身, commit 文件自包含
# ────────────────────────────────────────────────────────────────────────────


class FsMementoBranch(MementoBranch):
    def __init__(
        self,
        branch_dir: Path,
        branches_root: Path,
        hooks: MementoHooks,
        *,
        readonly: bool,
    ):
        self._dir = branch_dir
        self._branches_root = branches_root
        self._hooks = hooks
        self._readonly = readonly
        self._meta = _load_meta(branch_dir)
        # 冻结 id 集合按需重建 (读 commits/ 一遍即得)
        self._frozen_ids: set[str] | None = None
        # 崩溃恢复: 装入时执行一次 (FORMAT.md §12).
        # 仅对可写 handle 执行 — readonly 视图不改写文件.
        if not readonly:
            self._recover_from_crash()

    @property
    def meta(self) -> BranchMeta:
        return self._meta

    @property
    def readonly(self) -> bool:
        return self._readonly

    def _check_writable(self) -> None:
        if self._readonly:
            raise ReadonlyBranchError(f"branch {self._meta.branch_id} is readonly for this handle")

    def _staging_path(self) -> Path:
        return self._dir / "staging.jsonl"

    def _frozen(self) -> set[str]:
        if self._frozen_ids is None:
            frozen: set[str] = set()
            for path in _commit_files(self._dir):
                objs = _read_lines(path)
                if objs and objs[0].get("t") == "commit":
                    ids = objs[0].get("moment_ids")
                    if isinstance(ids, list):
                        frozen.update(mid for mid in ids if isinstance(mid, str))
            self._frozen_ids = frozen
        return self._frozen_ids

    def _recover_from_crash(self) -> None:
        """
        FORMAT.md §12: commit 原子动作 = 写 commit 文件 -> fsync -> truncate staging.
        崩溃恢复规则 (幂等):
          - 无 commit 文件: staging 是活跃写面, 无操作.
          - commit 文件成员行残缺 (t:"commit" 首行缺失): 删该文件, 物理身份缺失即无 commit.
          - commit 文件完整, 且 staging 全部 id 都是 last commit 成员: 崩溃残留, truncate staging.
          - 其它情况 (staging 含 last commit 之后的新 id): 合法状态, 无操作.
        """
        files = _commit_files(self._dir)
        if not files:
            return
        last = files[-1]
        objs = _read_lines(last)
        if not objs or objs[0].get("t") != "commit":
            last.unlink()
            return
        staging = self._staging_path()
        if not staging.exists() or staging.stat().st_size == 0:
            return
        last_ids = set(objs[0].get("moment_ids") or [])
        staged_ids = {
            obj["id"]
            for obj in _read_lines(staging)
            if obj.get("t") == "moment" and isinstance(obj.get("id"), str)
        }
        if staged_ids and staged_ids.issubset(last_ids):
            # 全是 last commit 的成员 = truncate 步骤未落, 崩溃残留.
            staging.write_text("", encoding="utf-8")

    # ── 写入路径 ──

    def update(self, record: MomentRecord) -> None:
        self._check_writable()
        if not _MOMENT_ID_PATTERN.match(record.id):
            raise MementoError(f"invalid moment id: {record.id!r} (FORMAT.md §2.1)")
        # 冻结即物理 (FORMAT.md §3.2 / 不变量 #13): staging 无此 id 的可写槽位.
        if record.id in self._frozen():
            raise MomentFrozenError(
                f"moment {record.id!r} is frozen by a commit of branch {self._meta.branch_id}; "
                f"updates after freeze are illegal (FORMAT.md §3.2)"
            )
        _append_lines(self._staging_path(), [_model_line("moment", record)])
        self._hooks.on_record_staged(self._meta.branch_id, record)

    def annotate_moment(self, moment_id: str, threads: Sequence[str], *, by: str = "") -> None:
        self._check_writable()
        if moment_id in self._frozen():
            # 冻结: 追加到 moment 所在的 commit 文件 (FORMAT.md §2.2 / §5.3).
            target = self._find_commit_of_moment(moment_id)
            if target is None:
                raise MementoError(
                    f"annotate_moment: moment {moment_id!r} frozen but source commit not found"
                )
            line: dict[str, Any] = {
                "t": "moment_note",
                "ref": moment_id,
                "threads": list(threads),
                "ts": _now().isoformat(),
            }
            if by:
                line["by"] = by
            _append_lines(target, [line])
            return
        # 未冻结: 追加到 staging.jsonl (§4.2).
        line = {
            "t": "moment_note",
            "ref": moment_id,
            "threads": list(threads),
            "ts": _now().isoformat(),
        }
        if by:
            line["by"] = by
        _append_lines(self._staging_path(), [line])

    def _find_commit_of_moment(self, moment_id: str) -> Path | None:
        for path in _commit_files(self._dir):
            objs = _read_lines(path)
            if objs and objs[0].get("t") == "commit":
                if moment_id in (objs[0].get("moment_ids") or []):
                    return path
        return None

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
        order, records = _resolve_staging(self._staging_path())
        if not order:
            raise EmptyStagingError(f"staging of branch {self._meta.branch_id} is empty")
        # 只保留 records 里存活的 id (悬空 moment_note 可能在 order 里没对应 moment).
        moment_ids = [mid for mid in order if mid in records]
        if not moment_ids:
            raise EmptyStagingError(
                f"staging of branch {self._meta.branch_id} has no resolvable moment record"
            )
        seq = self._max_seq() + 1
        commit = Commit(seq=seq, moment_ids=moment_ids)
        trailers: list[tuple[str, str]] = []
        trailers += [(TRAILER_THREAD, t) for t in threads]
        trailers += [(TRAILER_RESUMES, r) for r in resumes]
        trailers += [(TRAILER_SUSPENDS, s) for s in suspends]
        trailers += list(extra_trailers)
        trailers.append((TRAILER_KIND, kind))
        note = CommitNote(ref=commit.id, body=join_trailers(text, trailers), by=by)
        path = self._dir / "commits" / f"{seq:04d}.jsonl"
        # FORMAT.md §5: 成员行 + m 个冻结 moment 行 + 初始 commit_note. 一次写入.
        lines: list[dict[str, Any]] = [_model_line("commit", commit)]
        for mid in moment_ids:
            lines.append(_model_line("moment", records[mid]))
        lines.append(_model_line("commit_note", note))
        _append_lines(path, lines)
        # FORMAT.md §12: fsync commit 文件后再 truncate staging (原子性锚点).
        _fsync_file(path)
        self._staging_path().write_text("", encoding="utf-8")
        self._meta = self._meta.model_copy(update={"updated": _now()})
        _save_meta(self._dir, self._meta)
        self._frozen().update(moment_ids)
        view = CommitView(commit=commit, note=note, note_seq=0)
        self._hooks.on_commit(self._meta.branch_id, view)
        return view

    def reinterpret(self, commit_id: str, body: str, *, by: str = "") -> CommitView:
        self._check_writable()
        for path in _commit_files(self._dir):
            commit, _records, notes = _load_commit_file(path)
            if commit.id == commit_id:
                note = CommitNote(ref=commit_id, body=body, by=by)
                _append_lines(path, [_model_line("commit_note", note)])
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
        commit, _records, notes = _load_commit_file(files[-1])
        return _view_of(commit, notes)

    def staging(self) -> list[MomentRecord]:
        order, records = _resolve_staging(self._staging_path())
        return [records[mid] for mid in order if mid in records]

    def own_commits(self) -> list[CommitView]:
        out: list[CommitView] = []
        for path in _commit_files(self._dir):
            commit, _records, notes = _load_commit_file(path)
            out.append(_view_of(commit, notes))
        return out

    def _ancestor_views(self) -> list[CommitView]:
        """
        沿冻结祖先链展开每段祖先 branch 的 commit 到 commit_seq 为止 (含).
        BasePointer.moment_id 只影响该段最末 commit 的成员切片 (作用于
        commit_records / window, 不改变 view 本身的元数据).
        """
        views: list[CommitView] = []
        for bp in self._meta.ancestry:
            adir = self._branches_root / bp.fork / bp.branch_id
            for path in _commit_files(adir):
                seq = int(path.stem)
                if seq > bp.commit_seq:
                    continue
                commit, _records, notes = _load_commit_file(path)
                views.append(_view_of(commit, notes))
        return views

    def all_commits(self) -> list[CommitView]:
        return self._ancestor_views() + self.own_commits()

    def get_commit(self, commit_id: str) -> CommitView | None:
        for view in reversed(self.all_commits()):
            if view.id == commit_id:
                return view
        return None

    def _load_records_of(
        self, commit_id: str, *, apply_prefix: bool = True
    ) -> list[MomentRecord]:
        """
        找到 commit 所在的 commit 文件, 读回它的冻结 moment 列表.
        apply_prefix=True 时, 若该 commit 命中 ancestry 中的某段 BasePointer.moment_id,
        列表按 moment_seq (含) 切片 (FORMAT.md §4.1.1 inclusive).
        """
        # own commit
        for path in _commit_files(self._dir):
            commit, records, _notes = _load_commit_file(path)
            if commit.id == commit_id:
                return records
        # ancestor commit
        for bp in self._meta.ancestry:
            adir = self._branches_root / bp.fork / bp.branch_id
            for path in _commit_files(adir):
                if int(path.stem) > bp.commit_seq:
                    continue
                commit, records, _notes = _load_commit_file(path)
                if commit.id != commit_id:
                    continue
                if (
                    apply_prefix
                    and int(path.stem) == bp.commit_seq
                    and bp.moment_id is not None
                    and bp.moment_seq is not None
                ):
                    if bp.moment_seq >= len(records) or (
                        records[bp.moment_seq].id != bp.moment_id
                    ):
                        raise MomentNotInCommitError(
                            f"BasePointer.moment_id {bp.moment_id!r} does not match "
                            f"member at moment_seq={bp.moment_seq} in commit {commit_id!r}"
                        )
                    return records[: bp.moment_seq + 1]
                return records
        raise CommitNotFoundError(
            f"commit {commit_id!r} not in history of branch {self._meta.branch_id}"
        )

    def commit_records(self, commit_id: str) -> list[MomentRecord]:
        return self._load_records_of(commit_id, apply_prefix=True)

    def notes(self, commit_id: str) -> list[CommitNote]:
        # own commits
        for path in _commit_files(self._dir):
            commit, _records, notes = _load_commit_file(path)
            if commit.id == commit_id:
                return notes
        # ancestor commits (跨 owner 只读)
        for bp in reversed(self._meta.ancestry):
            adir = self._branches_root / bp.fork / bp.branch_id
            for path in _commit_files(adir):
                if int(path.stem) > bp.commit_seq:
                    continue
                commit, _records, notes = _load_commit_file(path)
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
            records = self._load_records_of(view.id, apply_prefix=True)
            taken = records[-need:] if need <= len(records) else records
            details = list(taken) + details
            if len(taken) == 0:
                break
            need -= len(taken)
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
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        gitignore = self._root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(".cache/\n", encoding="utf-8")

    @property
    def owner(self) -> str:
        return self._owner

    def _branches_root(self) -> Path:
        return self._root / "branches"

    def _owner_dir(self) -> Path:
        return self._branches_root() / self._owner

    def _branch(self, fork: str, branch_id: str, *, readonly: bool) -> FsMementoBranch:
        branch_dir = self._branches_root() / fork / branch_id
        branch = FsMementoBranch(
            branch_dir, self._branches_root(), self._hooks, readonly=readonly
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
        base_moment_id: str | None = None,
        name: str = "",
        overlay: dict[str, Any] | None = None,
    ) -> MementoBranch:
        source = self._branch(base_fork, base_branch_id, readonly=True)
        base = self._locate_base(source, base_commit_id, base_moment_id)
        base_meta = _load_meta(self._branches_root() / base.fork / base.branch_id)
        meta = BranchMeta(
            fork=self._owner,
            name=name,
            base=base,
            # fork 时刻冻结展平祖先链: ancestry == base 的 ancestry + [base]
            ancestry=list(base_meta.ancestry) + [base],
            overlay=overlay or {},
        )
        _save_meta(self._owner_dir() / meta.branch_id, meta)
        self._hooks.on_branch_created(meta)
        return self._branch(self._owner, meta.branch_id, readonly=False)

    def _locate_base(
        self,
        source: FsMementoBranch,
        base_commit_id: str | None,
        base_moment_id: str | None,
    ) -> BasePointer:
        commit, records = self._resolve_target(source, base_commit_id)
        fork, branch_id = self._locate_commit_owner(source, commit.id)
        moment_id_field: str | None = None
        moment_seq_field: int | None = None
        if base_moment_id is not None:
            found = [
                (idx, r) for idx, r in enumerate(records) if r.id == base_moment_id
            ]
            if not found:
                raise MomentNotInCommitError(
                    f"base_moment_id {base_moment_id!r} not in commit {commit.id!r} members"
                )
            idx, _ = found[0]
            moment_id_field = base_moment_id
            moment_seq_field = idx
        return BasePointer(
            fork=fork,
            branch_id=branch_id,
            commit_id=commit.id,
            commit_seq=commit.seq,
            moment_id=moment_id_field,
            moment_seq=moment_seq_field,
        )

    def _resolve_target(
        self, source: FsMementoBranch, base_commit_id: str | None
    ) -> tuple[Commit, list[MomentRecord]]:
        if base_commit_id is None:
            files = _commit_files(source._dir)
            if not files:
                raise MementoError(
                    f"source branch {source.meta.branch_id} has no commit; "
                    f"化身只能从 commit 出生, 永不从 staging"
                )
            commit, records, _notes = _load_commit_file(files[-1])
            return commit, records
        # own commit
        for path in _commit_files(source._dir):
            commit, records, _notes = _load_commit_file(path)
            if commit.id == base_commit_id:
                return commit, records
        # ancestor commit
        for bp in source.meta.ancestry:
            adir = self._branches_root() / bp.fork / bp.branch_id
            for path in _commit_files(adir):
                if int(path.stem) > bp.commit_seq:
                    continue
                commit, records, _notes = _load_commit_file(path)
                if commit.id == base_commit_id:
                    if (
                        int(path.stem) == bp.commit_seq
                        and bp.moment_id is not None
                        and bp.moment_seq is not None
                    ):
                        records = records[: bp.moment_seq + 1]
                    return commit, records
        raise CommitNotFoundError(
            f"commit {base_commit_id!r} not in history of branch {source.meta.branch_id}"
        )

    def _locate_commit_owner(
        self, source: FsMementoBranch, commit_id: str
    ) -> tuple[str, str]:
        for path in _commit_files(source._dir):
            objs = _read_lines(path)
            if objs and objs[0].get("t") == "commit" and objs[0].get("id") == commit_id:
                return source.meta.fork, source.meta.branch_id
        for bp in source.meta.ancestry:
            adir = self._branches_root() / bp.fork / bp.branch_id
            for path in _commit_files(adir):
                if int(path.stem) > bp.commit_seq:
                    continue
                objs = _read_lines(path)
                if objs and objs[0].get("t") == "commit" and objs[0].get("id") == commit_id:
                    return bp.fork, bp.branch_id
        raise CommitNotFoundError(
            f"commit {commit_id!r} owner not resolvable (this is a bug: caller must ensure existence)"
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
        pass


def new_filesystem_memento(
    root: str | Path, owner: str, *, hooks: MementoHooks | None = None
) -> FsMemento:
    return FsMemento(root, owner, hooks=hooks)
