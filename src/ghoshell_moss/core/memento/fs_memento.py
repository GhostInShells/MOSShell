"""FileSystem-backed Branch + Memento 实现.

存储布局 (以 branches_root 为相对路径):

    branches_root/
      branches/
        {owner}/
          _branches.json          # {"current": branch_id, "names": {name: branch_id}}
          {branch_id}/
            meta.json             # BranchMeta
            staging.jsonl         # one moment_id per line
            commits/
              0001.json           # Commit
              0002.json

并发模型: owner-isolated. 每个 owner 命名空间只有一个 Memento 实例 (单进程单线程)
做写入. 跨 owner 读取走文件系统 (readonly). 因此:
- staging.jsonl: append-only, 单 writer 无 race.
- commits/{seq:04d}.json: write-once, seq 在进程内通过目录扫描决定.
- meta.json / _branches.json: tmp + atomic rename.

第一版不内置进程间锁. 若未来需要 (例如 fractal cell 跨进程共享 owner),
在外层 (session.cache) 加锁即可.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dateutil import tz

from ghoshell_moss.contracts.workspace import Storage
from ghoshell_moss.core.memento.abc import (
    BasePointer,
    BranchMeta,
    BranchNotFoundError,
    Commit,
    CommitNotFoundError,
    Memento,
    MementoBranch,
    MementoError,
    MementoHooks,
    MementoRef,
    Moment,
    MomentStore,
    NullHooks,
    ReadonlyBranchError,
)
from ghoshell_moss.core.memento.sqlite_moment_store import SqliteMomentStore
from ghoshell_moss.message import Message

__all__ = ["FsMemento", "new_filesystem_memento"]


# ────────────────────────────────────────────────────────────────────────────
# 内部 helpers
# ────────────────────────────────────────────────────────────────────────────

_COMMIT_FILE_RE = re.compile(r"^(\d{4,})\.json$")


def _now() -> datetime:
    return datetime.now(tz.gettz())


def _atomic_write_text(path: Path, content: str) -> None:
    """tmp + rename, 防止读到半成品."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _owner_dir(branches_root: Path, owner: str) -> Path:
    return branches_root / "branches" / owner


def _branch_dir(branches_root: Path, owner: str, branch_id: str) -> Path:
    return _owner_dir(branches_root, owner) / branch_id


def _read_branch_index(branches_root: Path, owner: str) -> dict:
    """{current: branch_id | None, names: {name: branch_id}}"""
    f = _owner_dir(branches_root, owner) / "_branches.json"
    if not f.exists():
        return {"current": None, "names": {}}
    return json.loads(f.read_text(encoding="utf-8"))


def _write_branch_index(branches_root: Path, owner: str, index: dict) -> None:
    f = _owner_dir(branches_root, owner) / "_branches.json"
    _atomic_write_text(f, json.dumps(index, ensure_ascii=False, indent=2))


def _read_meta(branches_root: Path, owner: str, branch_id: str) -> BranchMeta:
    f = _branch_dir(branches_root, owner, branch_id) / "meta.json"
    if not f.exists():
        raise BranchNotFoundError(f"branch {owner}/{branch_id} meta.json missing")
    return BranchMeta.model_validate_json(f.read_text(encoding="utf-8"))


def _write_meta(branches_root: Path, owner: str, branch_id: str, meta: BranchMeta) -> None:
    f = _branch_dir(branches_root, owner, branch_id) / "meta.json"
    _atomic_write_text(f, meta.model_dump_json(indent=2))


def _load_own_commits(branches_root: Path, owner: str, branch_id: str) -> list[Commit]:
    """读 branch 自己的 commits/ 目录, 按 seq 升序."""
    commits_dir = _branch_dir(branches_root, owner, branch_id) / "commits"
    if not commits_dir.exists():
        return []
    files: list[tuple[int, Path]] = []
    for entry in commits_dir.iterdir():
        if m := _COMMIT_FILE_RE.match(entry.name):
            files.append((int(m.group(1)), entry))
    files.sort(key=lambda x: x[0])
    return [Commit.model_validate_json(p.read_text(encoding="utf-8")) for _, p in files]


# ────────────────────────────────────────────────────────────────────────────
# FsBranch
# ────────────────────────────────────────────────────────────────────────────


class FsBranch(MementoBranch):
    """FileSystem-backed branch."""

    def __init__(
        self,
        *,
        branches_root: Path,
        owner: str,
        branch_id: str,
        moment_store: MomentStore,
        hooks: MementoHooks,
        readonly: bool,
    ) -> None:
        self._branches_root = branches_root
        self._owner = owner
        self._branch_id = branch_id
        self._moment_store = moment_store
        self._hooks = hooks
        self._readonly = readonly
        self._write_lock = threading.Lock()  # 进程内 seq 计算保护

    @property
    def meta(self) -> BranchMeta:
        return _read_meta(self._branches_root, self._owner, self._branch_id)

    @property
    def readonly(self) -> bool:
        return self._readonly

    # ── 写入 ──

    def update(self, moment: Moment) -> None:
        if self._readonly:
            raise ReadonlyBranchError(
                f"branch {self._owner}/{self._branch_id} is readonly"
            )
        with self._write_lock:
            self._moment_store.put(moment, owner=self._owner)
            staging_path = self._staging_path()
            staging_path.parent.mkdir(parents=True, exist_ok=True)

            # 去重 — 同 id 不重复 append.
            existing_ids = self._read_staging_ids()
            if moment.id not in existing_ids:
                with staging_path.open("a", encoding="utf-8") as f:
                    f.write(moment.id + "\n")

        try:
            self._hooks.on_moment_updated(self._branch_id, moment)
        except Exception:
            pass

    def commit(self, summary: str) -> Commit:
        if self._readonly:
            raise ReadonlyBranchError(
                f"branch {self._owner}/{self._branch_id} is readonly"
            )
        with self._write_lock:
            moment_ids = self._read_staging_ids()
            if not moment_ids:
                raise MementoError(
                    f"branch {self._owner}/{self._branch_id} staging is empty"
                )

            own = _load_own_commits(self._branches_root, self._owner, self._branch_id)
            next_seq = (own[-1].seq + 1) if own else 1

            commit = Commit(
                seq=next_seq,
                summary=summary,
                moment_ids=moment_ids,
                created=_now(),
            )
            commit_path = (
                _branch_dir(self._branches_root, self._owner, self._branch_id)
                / "commits"
                / f"{next_seq:04d}.json"
            )
            _atomic_write_text(commit_path, commit.model_dump_json(indent=2))

            # 清空 staging
            self._staging_path().write_text("", encoding="utf-8")

            # 更新 meta.updated
            meta = _read_meta(self._branches_root, self._owner, self._branch_id)
            meta = meta.model_copy(update={"updated": _now()})
            _write_meta(self._branches_root, self._owner, self._branch_id, meta)

        try:
            self._hooks.on_commit(self._branch_id, commit)
        except Exception:
            pass
        return commit

    # ── 读取 ──

    def head(self) -> Commit | None:
        own = _load_own_commits(self._branches_root, self._owner, self._branch_id)
        if own:
            return own[-1]
        # 自己没 commit 时, head 应该看 base chain 最后一个
        meta = self.meta
        if meta.base is None:
            return None
        base_commits = _walk_base_chain(self._branches_root, meta.base)
        return base_commits[-1] if base_commits else None

    def staging(self) -> list[Moment]:
        ids = self._read_staging_ids()
        if not ids:
            return []
        loaded = self._moment_store.get_many(ids)
        # 按 staging 写入顺序返回 (保留缺失项为 None 会破坏类型, 跳过缺失即可)
        return [loaded[mid] for mid in ids if mid in loaded]

    def own_commits(self) -> list[Commit]:
        return _load_own_commits(self._branches_root, self._owner, self._branch_id)

    def all_commits(self) -> list[Commit]:
        meta = self.meta
        if meta.base is None:
            return self.own_commits()
        base_commits = _walk_base_chain(self._branches_root, meta.base)
        return base_commits + self.own_commits()

    def get_commit(self, commit_id: str) -> Commit | None:
        for c in self.all_commits():
            if c.id == commit_id:
                return c
        return None

    def commit_moments(self, commit_id: str) -> list[Moment]:
        commit = self.get_commit(commit_id)
        if commit is None:
            raise CommitNotFoundError(
                f"commit {commit_id} not found in branch {self._owner}/{self._branch_id}"
            )
        loaded = self._moment_store.get_many(commit.moment_ids)
        return [loaded[mid] for mid in commit.moment_ids if mid in loaded]

    def history(
        self,
        *,
        detail_n: int = 10,
        summary_m: int = -1,
    ) -> Iterable[Message]:
        all_commits = self.all_commits()
        staging_moments = self.staging()

        # 构建反向时间线 [(moment_id, source_type, source_ref)]:
        # source_type ∈ {'staging', 'commit'}
        # source_ref = commit (for 'commit'), None (for 'staging')
        # 反向 = 新到旧
        timeline_reverse: list[tuple[str, str, Commit | None]] = []
        for m in reversed(staging_moments):
            timeline_reverse.append((m.id, "staging", None))
        for c in reversed(all_commits):
            for mid in reversed(c.moment_ids):
                timeline_reverse.append((mid, "commit", c))

        # 决定 detail 窗口: 最近 detail_n 个 moment 全量展开
        detail_set: set[str] = set()
        commits_with_detail_moment: set[str] = set()
        if detail_n > 0:
            for mid, src, ref in timeline_reverse[:detail_n]:
                detail_set.add(mid)
                if src == "commit" and ref is not None:
                    commits_with_detail_moment.add(ref.id)

        # 决定要 summarize 的 commits: 自己没有 moment 在 detail 窗口的 commit
        summary_commits = [
            c for c in all_commits if c.id not in commits_with_detail_moment
        ]
        if summary_m >= 0:
            # 保留最新 summary_m 个 (按时间)
            summary_commits = summary_commits[-summary_m:] if summary_m > 0 else []

        summary_commit_ids = {c.id for c in summary_commits}

        # 正向 emit: 先 summary commits (按 commit 时间顺序), 再 detail moments (按时间顺序)
        for c in all_commits:
            if c.id not in summary_commit_ids:
                continue
            ref = MementoRef(
                fork=self._owner,
                branch_id=self._branch_id,
                commit_id=c.id,
                commit_seq=c.seq,
            )
            summary_msg = Message.new(tag="commit-summary").with_content(c.summary)
            ref.set(summary_msg)
            yield summary_msg

        # 正向 detail
        detail_timeline = [
            (mid, src) for mid, src, _ref in reversed(timeline_reverse) if mid in detail_set
        ]
        for mid, _src in detail_timeline:
            m = self._moment_store.get(mid)
            if m is None:
                continue
            # 把 Moment 展开为 messages — 用 as_history_messages 保持 ghost runtime 一致语义
            yield from m.as_history_messages()

    # ── 私有 ──

    def _staging_path(self) -> Path:
        return _branch_dir(self._branches_root, self._owner, self._branch_id) / "staging.jsonl"

    def _read_staging_ids(self) -> list[str]:
        p = self._staging_path()
        if not p.exists():
            return []
        ids: list[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                ids.append(s)
        return ids


def _walk_base_chain(branches_root: Path, base: BasePointer) -> list[Commit]:
    """沿 BasePointer 回溯, 返回 ancestor commits (按时间升序, 截至 base.commit_seq)."""
    chain: list[Commit] = []
    cur: BasePointer | None = base
    while cur is not None:
        ancestor_own = _load_own_commits(branches_root, cur.fork, cur.branch_id)
        truncated = [c for c in ancestor_own if c.seq <= cur.commit_seq]
        chain = truncated + chain  # prepend — 更老的放前面
        try:
            meta = _read_meta(branches_root, cur.fork, cur.branch_id)
        except BranchNotFoundError:
            break
        cur = meta.base
    return chain


# ────────────────────────────────────────────────────────────────────────────
# FsMemento
# ────────────────────────────────────────────────────────────────────────────


class FsMemento(Memento):
    """FileSystem-backed Memento facade. 绑定一个 owner."""

    def __init__(
        self,
        *,
        branches_storage: Storage,
        owner: str,
        moment_store: MomentStore,
        hooks: MementoHooks | None = None,
    ) -> None:
        self._branches_root = branches_storage.abspath()
        self._owner = owner
        self._moment_store = moment_store
        self._hooks = hooks or NullHooks()
        self._owner_lock = threading.Lock()  # _branches.json 写入保护

    @property
    def owner(self) -> str:
        return self._owner

    # ── current ──

    def current(self) -> MementoBranch:
        with self._owner_lock:
            index = _read_branch_index(self._branches_root, self._owner)
            branch_id = index.get("current")
            if branch_id is None or not (
                _branch_dir(self._branches_root, self._owner, branch_id) / "meta.json"
            ).exists():
                # 自动创建 root 'main' branch
                branch_id = self._create_root_branch(name="main")
                index["current"] = branch_id
                index.setdefault("names", {})["main"] = branch_id
                _write_branch_index(self._branches_root, self._owner, index)
        return self._make_handle(self._owner, branch_id)

    def switch(self, branch_id: str) -> None:
        with self._owner_lock:
            if not (
                _branch_dir(self._branches_root, self._owner, branch_id) / "meta.json"
            ).exists():
                raise BranchNotFoundError(
                    f"branch {self._owner}/{branch_id} not found"
                )
            index = _read_branch_index(self._branches_root, self._owner)
            index["current"] = branch_id
            _write_branch_index(self._branches_root, self._owner, index)
        try:
            self._hooks.on_branch_switched(branch_id)
        except Exception:
            pass

    # ── branch 管理 ──

    def checkout(
        self,
        *,
        base_fork: str,
        base_branch_id: str,
        base_commit_id: str | None = None,
        new_name: str = "",
        recap: str = "",
    ) -> MementoBranch:
        # resolve base commit
        ancestor_meta_path = (
            _branch_dir(self._branches_root, base_fork, base_branch_id) / "meta.json"
        )
        if not ancestor_meta_path.exists():
            raise BranchNotFoundError(f"branch {base_fork}/{base_branch_id} not found")

        ancestor_own_commits = _load_own_commits(
            self._branches_root, base_fork, base_branch_id
        )
        if base_commit_id is None:
            # 取源 branch 的 head (含 base chain)
            ancestor_branch = self._make_handle(base_fork, base_branch_id)
            head = ancestor_branch.head()
            if head is None:
                raise MementoError(
                    f"branch {base_fork}/{base_branch_id} has no commits to fork from"
                )
            base_commit = head
        else:
            # 在源 branch 的 all_commits 中找
            ancestor_branch = self._make_handle(base_fork, base_branch_id)
            base_commit = ancestor_branch.get_commit(base_commit_id)
            if base_commit is None:
                raise CommitNotFoundError(
                    f"commit {base_commit_id} not found in {base_fork}/{base_branch_id}"
                )

        # 找 base_commit 的真实归属 branch (可能是源 branch 自己, 也可能是它的 ancestor)
        if any(c.id == base_commit.id for c in ancestor_own_commits):
            owning_fork = base_fork
            owning_branch_id = base_branch_id
        else:
            # 在 base chain 里找
            owning_fork, owning_branch_id = _find_commit_owner(
                self._branches_root,
                _read_meta(self._branches_root, base_fork, base_branch_id).base,
                base_commit.id,
            )
            if owning_fork is None:
                raise CommitNotFoundError(
                    f"could not locate owner of commit {base_commit.id}"
                )

        base_pointer = BasePointer(
            fork=owning_fork,
            branch_id=owning_branch_id,
            commit_id=base_commit.id,
            commit_seq=base_commit.seq,
        )

        new_meta = BranchMeta(
            fork=self._owner,
            name=new_name,
            recap=recap,
            base=base_pointer,
            created=_now(),
            updated=_now(),
        )
        with self._owner_lock:
            _write_meta(self._branches_root, self._owner, new_meta.branch_id, new_meta)
            if new_name:
                index = _read_branch_index(self._branches_root, self._owner)
                index.setdefault("names", {})[new_name] = new_meta.branch_id
                _write_branch_index(self._branches_root, self._owner, index)

        try:
            self._hooks.on_branch_created(new_meta)
        except Exception:
            pass
        return self._make_handle(self._owner, new_meta.branch_id)

    def get_branch(
        self,
        branch_id: str,
        *,
        fork: str | None = None,
    ) -> MementoBranch:
        target_fork = fork or self._owner
        if not (
            _branch_dir(self._branches_root, target_fork, branch_id) / "meta.json"
        ).exists():
            raise BranchNotFoundError(f"branch {target_fork}/{branch_id} not found")
        return self._make_handle(target_fork, branch_id)

    def list_branches(self, fork: str | None = None) -> list[BranchMeta]:
        target_fork = fork or self._owner
        owner_dir = _owner_dir(self._branches_root, target_fork)
        if not owner_dir.exists():
            return []
        metas: list[BranchMeta] = []
        for entry in owner_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_file = entry / "meta.json"
            if meta_file.exists():
                metas.append(BranchMeta.model_validate_json(meta_file.read_text(encoding="utf-8")))
        metas.sort(key=lambda m: m.created)
        return metas

    def list_forks(self) -> list[str]:
        root = self._branches_root / "branches"
        if not root.exists():
            return []
        return sorted([e.name for e in root.iterdir() if e.is_dir()])

    # ── merge message ──

    def make_merge_message(
        self,
        branch_id: str,
        commit_id: str,
        *,
        fork: str | None = None,
    ) -> Message:
        target_fork = fork or self._owner
        branch = self.get_branch(branch_id, fork=target_fork)
        commit = branch.get_commit(commit_id)
        if commit is None:
            raise CommitNotFoundError(
                f"commit {commit_id} not found in {target_fork}/{branch_id}"
            )
        ref = MementoRef(
            fork=target_fork,
            branch_id=branch_id,
            commit_id=commit.id,
            commit_seq=commit.seq,
        )
        msg = Message.new(tag="memento-merge").with_content(commit.summary)
        ref.set(msg)
        return msg

    # ── 生命周期 ──

    def close(self) -> None:
        self._moment_store.close()

    # ── 内部 ──

    def _create_root_branch(self, *, name: str) -> str:
        meta = BranchMeta(
            fork=self._owner,
            name=name,
            base=None,
            created=_now(),
            updated=_now(),
        )
        _write_meta(self._branches_root, self._owner, meta.branch_id, meta)
        try:
            self._hooks.on_branch_created(meta)
        except Exception:
            pass
        return meta.branch_id

    def _make_handle(self, fork: str, branch_id: str) -> FsBranch:
        return FsBranch(
            branches_root=self._branches_root,
            owner=fork,
            branch_id=branch_id,
            moment_store=self._moment_store,
            hooks=self._hooks,
            readonly=(fork != self._owner),
        )


def _find_commit_owner(
    branches_root: Path,
    base: BasePointer | None,
    target_commit_id: str,
) -> tuple[str | None, str | None]:
    """沿 base chain 找一个 commit_id 实际归属的 (fork, branch_id)."""
    cur = base
    while cur is not None:
        ancestor_own = _load_own_commits(branches_root, cur.fork, cur.branch_id)
        if any(c.id == target_commit_id for c in ancestor_own):
            return cur.fork, cur.branch_id
        try:
            meta = _read_meta(branches_root, cur.fork, cur.branch_id)
        except BranchNotFoundError:
            return None, None
        cur = meta.base
    return None, None


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────


def new_filesystem_memento(
    *,
    branches_storage: Storage,
    owner: str,
    moment_store: MomentStore | None = None,
    hooks: MementoHooks | None = None,
) -> FsMemento:
    """构建一个 FsMemento. moment_store 默认在 branches_storage/moments.db 自动创建."""
    if moment_store is None:
        moment_store = SqliteMomentStore(branches_storage.abspath() / "moments.db")
    return FsMemento(
        branches_storage=branches_storage,
        owner=owner,
        moment_store=moment_store,
        hooks=hooks,
    )
