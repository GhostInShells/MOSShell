"""Pin observation — mtime + hash 对账 (K17).

Pin.seen_mtime / seen_hash 存的是"上次承认时的观察". update() / pin() 时
对目标做一次 observe(), 拿到 Observation 更新 Pin. 帧渲染时对每个 pin
再 observe() 一次, 与 seen_* 对比: hash 相同 → 不打扰; hash 不同 →
帧内标 changed-on-disk.

Impl 契约:

- observe(parsed, root) 是 IO 密集 async, 内部对独立 stat/read 用
  asyncio.to_thread 或 gather. 单调用只观察一个 addr, 帧渲染层需要并行
  观察 N 个 pin 时在外面用 asyncio.gather(*[observe(...) for pin in pins]).
- 不 raise 对账错误 (文件不存在等) — Observation.exists 表达状态, 帧
  渲染判定. Raise 只在 addr 越界 (PathOutsideRootError, 来自 _addr).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ghoshell_moss.core.desktop._addr import (
    ParsedAddr,
    resolve_file_addr,
    resolve_glob_addr,
)

__all__ = ["Observation", "observe", "observe_sync"]


@dataclass(frozen=True)
class Observation:
    """一次对 Pin addr 的观察.

    进 Pin.seen_mtime / seen_hash 的对账三元:

    - exists: 目标是否存在.
      file/range: 文件存在 → True.
      glob: 恒 True (空命中集也是合法状态, mtime 为 None, hash 为空集 hash).
      exists=False 时 mtime 和 hash 都为 None.
    - mtime: file/range: 文件 mtime. glob: 命中集里最大 mtime (None 若空集).
    - hash: sha256 hex.
      file: 内容全文 hash.
      range: 该行区间内容 hash (行 splitkeepends 后切片).
      glob: 命中路径列表 (相对 root 排序后) 拼接 hash — 不含内容, 只关注
      命中集变化.
    """

    exists: bool
    mtime: float | None
    hash: str | None


_EMPTY_HASH = hashlib.sha256(b"").hexdigest()


async def observe(parsed: ParsedAddr, root: Path) -> Observation:
    """观察一次 addr 的当前状态. async wrapper — 内部 IO 卸载到线程池.

    见 ``observe_sync`` 的语义说明. 帧渲染层批量调用时用
    ``asyncio.gather(*(observe(p, root) for p in pins))`` 并发.
    """
    return await asyncio.to_thread(observe_sync, parsed, root)


def observe_sync(parsed: ParsedAddr, root: Path) -> Observation:
    """sync 版本. Ground.pin() 沿用它做 "立即观察一次" 契约 —
    pin 是模型的第一人称动作, 允许 blocking IO (K17 pin 时刻的 initial 承认).

    实现分支按 kind:
    - glob: expand + hash 命中路径列表 (不读文件内容)
    - file: read + hash 全文
    - range: read + slicelines + hash 行区间

    地址越界 (PathOutsideRootError) 传播不吞.
    """
    if parsed.kind == "glob":
        return _observe_glob_sync(parsed, root)
    return _observe_path_sync(parsed, root)


def _observe_glob_sync(parsed: ParsedAddr, root: Path) -> Observation:
    matches = resolve_glob_addr(parsed, root)
    if not matches:
        return Observation(exists=True, mtime=None, hash=_EMPTY_HASH)

    mtimes: list[float] = []
    for m in matches:
        try:
            mtimes.append(m.stat().st_mtime)
        except FileNotFoundError:
            continue
    latest = max(mtimes) if mtimes else None

    root_abs = root.resolve()
    rels = sorted(str(m.relative_to(root_abs)) for m in matches)
    digest = hashlib.sha256("\n".join(rels).encode("utf-8")).hexdigest()
    return Observation(exists=True, mtime=latest, hash=digest)


def _observe_path_sync(parsed: ParsedAddr, root: Path) -> Observation:
    target = resolve_file_addr(parsed, root)
    try:
        mtime = target.stat().st_mtime
    except FileNotFoundError:
        return Observation(exists=False, mtime=None, hash=None)

    if parsed.kind == "file":
        content = target.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        return Observation(exists=True, mtime=mtime, hash=digest)

    # range: parsed.start / end 均 1-based 含端点
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    assert parsed.start is not None and parsed.end is not None
    slice_text = "".join(lines[parsed.start - 1 : parsed.end])
    digest = hashlib.sha256(slice_text.encode("utf-8")).hexdigest()
    return Observation(exists=True, mtime=mtime, hash=digest)
