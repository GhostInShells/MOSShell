"""
见证层 (FORMAT.md §9) — git sidecar.

git 不是 memento 的结构, 是见证: fork 是纯 memento 层操作, git 只见证文件.
两个地址空间: memento id = 身份, git sha = 完整性 (历史未被事后篡改的证明).

主权层, v1 选型 subprocess git (低频单写者, 进程开销无关, toolchain 与人类
手工 git 完全一致). dulwich 是嵌入升级路径, 不引 libgit2/C 依赖.

本模块只提供原语 (ensure_repo / snapshot / Witness 收集器), 不含调度 —
何时快照 (commit 事件去抖/定时) 由集成方接线, 永不放进写入热路径.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ghoshell_moss.memento.abc import TRAILER_MEMENTO_REF, MementoError

__all__ = ["ensure_witness_repo", "snapshot", "Witness"]


def _git(memento_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=memento_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise MementoError(f"witness git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def ensure_witness_repo(memento_root: str | Path) -> Path:
    """
    确保 {memento_root}/.git 存在且独立. sidecar repo 的工作树就是 memento/ 本身,
    绝不能被外层代码仓库吞掉 (memento root 应被外层 ignore 或位于仓库之外).
    """
    root = Path(memento_root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        _git(root, "init", "--quiet")
        # 机器身份, 只服务于 sidecar repo 自身运转
        _git(root, "config", "user.name", "moss-memento-witness")
        _git(root, "config", "user.email", "witness@localhost")
    return root


def snapshot(
    memento_root: str | Path,
    refs: Sequence[str] = (),
    *,
    ts: datetime | None = None,
) -> str | None:
    """
    做一次快照. 无变更返回 None, 否则返回 git sha.

    :param refs: 自上次快照以来新增的 memento commit id, 写入 message 的
        Memento-Ref trailer — `git log --grep=cmt_xxx` 即反查见证时刻.
    """
    root = ensure_witness_repo(memento_root)
    _git(root, "add", "-A")
    staged = _git(root, "status", "--porcelain")
    if not staged:
        return None
    stamp = (ts or datetime.now().astimezone()).isoformat()
    message = f"snapshot: {stamp}"
    if refs:
        message += "\n\n" + "\n".join(f"{TRAILER_MEMENTO_REF}: {r}" for r in refs)
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD")


class Witness:
    """
    memento commit id 收集器 + 显式 flush.

    用法: hook 侧只调 note_commit() (纯内存, 不碰热路径);
    集成方在自己的低频调度点 (定时器/空闲回调) 调 flush().
    """

    def __init__(self, memento_root: str | Path):
        self._root = Path(memento_root)
        self._pending: list[str] = []

    def note_commit(self, commit_id: str) -> None:
        self._pending.append(commit_id)

    @property
    def pending(self) -> list[str]:
        return list(self._pending)

    def flush(self) -> str | None:
        """快照并清空 pending. 无变更也清空 (变更已被之前的快照覆盖)."""
        sha = snapshot(self._root, self._pending)
        self._pending.clear()
        return sha
