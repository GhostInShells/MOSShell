"""
Golden tests — FORMAT.md 契约锚点.

两条硬条款:
1. 字节等价: 独立照 FORMAT.md 各写一份历史, 互读对方字节, 重建等价.
   本文件里的 "独立实现" 就是纯 stdlib (json + pathlib) 直接照 FORMAT.md
   拼字节, 与 FsMemento 无共享代码路径.
2. 退化态纯净: 单 branch + 自动 commit 的用例代码里, fork 词汇一个都不出现.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from ghoshell_moss.core.memento import (
    FsMemento,
    MomentRecord,
    new_filesystem_memento,
    split_trailers,
    trailer_values,
)


# ============================================================
# 独立最小写入器: 只依赖 json + pathlib, 照 FORMAT.md 手工拼字节.
# 用来给 FsMemento 读; 反过来 FsMemento 写的也用 stdlib 校验.
# ============================================================


def _hand_write_history(
    root: Path,
    owner: str,
    branch_id: str,
    records: list[dict],
    commit_id: str,
    commit_seq: int,
    body: str,
) -> None:
    """按 FORMAT.md 逐字节拼一份最小历史."""
    now = datetime.now().astimezone().isoformat()
    ym = datetime.now().astimezone().astimezone().strftime("%Y-%m")
    # 简化: 直接 UTC 换算月份 (FORMAT.md §3.4)
    from datetime import timezone
    ym = datetime.now().astimezone(timezone.utc).strftime("%Y-%m")

    root.mkdir(parents=True, exist_ok=True)

    # moments/{owner}/{YYYY-MM}/moments.jsonl
    mfile = root / "moments" / owner / ym / "moments.jsonl"
    mfile.parent.mkdir(parents=True, exist_ok=True)
    with mfile.open("w", encoding="utf-8") as f:
        for r in records:
            line = {"t": "moment", "id": r["id"], "created": r.get("created", now),
                    "type": r.get("type", "test.data/v1"), "payload": r["payload"]}
            if r.get("threads"):
                line["threads"] = r["threads"]
            f.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")

    # branches/{owner}/{branch_id}/meta.json
    bdir = root / "branches" / owner / branch_id
    bdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "branch_id": branch_id, "fork": owner, "name": "main",
        "ancestry": [], "overlay": {}, "created": now, "updated": now,
    }
    (bdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # HEAD.json
    (root / "branches" / owner / "HEAD.json").write_text(
        json.dumps({"current": branch_id}, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # commits/0001.jsonl: 成员行 + 初始释义行
    cdir = bdir / "commits"
    cdir.mkdir(exist_ok=True)
    with (cdir / f"{commit_seq:04d}.jsonl").open("w", encoding="utf-8") as f:
        member = {"t": "commit", "id": commit_id, "seq": commit_seq,
                  "moment_ids": [r["id"] for r in records], "created": now}
        note = {"t": "note", "ref": commit_id, "body": body, "ts": now}
        f.write(json.dumps(member, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.write(json.dumps(note, ensure_ascii=False, separators=(",", ":")) + "\n")

    # staging.jsonl 空文件
    (bdir / "staging.jsonl").write_text("", encoding="utf-8")


# ============================================================
# 硬条款 1: 字节等价 — hand-written -> FsMemento 读
# ============================================================


def test_hand_written_history_readable_by_fs(tmp_path: Path):
    root = tmp_path / "memento"
    _hand_write_history(
        root,
        owner="alice",
        branch_id="brn_HAND0000000000000000001",
        records=[
            {"id": "m1", "payload": {"text": "早"}},
            {"id": "m2", "payload": {"text": "line1\nline2"}, "threads": ["t"]},
        ],
        commit_id="cmt_HAND0000000000000000001",
        commit_seq=1,
        body="hand-written history\n\nThread: t\nKind: semantic",
    )

    m = new_filesystem_memento(root, "alice")
    b = m.current()
    assert b.meta.branch_id == "brn_HAND0000000000000000001"
    head = b.head()
    assert head.id == "cmt_HAND0000000000000000001"
    assert head.summary() == "hand-written history"
    assert head.note.threads() == ["t"]
    assert head.note.kind() == "semantic"
    records = b.commit_records(head.id)
    assert records[0].payload == {"text": "早"}
    assert records[1].payload == {"text": "line1\nline2"}
    assert records[1].threads == ["t"]


# ============================================================
# 硬条款 1: 字节等价 — FsMemento -> hand-written 校验字节
# ============================================================


def test_fs_written_history_conforms_to_format(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "alice")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"text": "早"}))
    b.update(MomentRecord(id="m2", type="test.data/v1", payload={"text": "line1\nline2"}))
    view = b.commit("golden", kind="semantic", threads=["t"])
    m.close()

    # 池文件
    mfiles = list((root / "moments" / "alice").rglob("moments.jsonl"))
    assert len(mfiles) == 1
    lines = mfiles[0].read_text(encoding="utf-8").splitlines()
    # 每行 JSON object, 有 t 字段
    for line in lines:
        obj = json.loads(line)
        assert "t" in obj
        assert obj["t"] in {"moment", "note"}
    # 非 ASCII 无转义 (ensure_ascii=False)
    assert "早" in mfiles[0].read_text(encoding="utf-8")
    # 换行按 JSON 标准转义 (物理行内不含未转义 \n)
    assert r"line1\nline2" in mfiles[0].read_text(encoding="utf-8")

    # commit 文件: 首行 t=commit, 次行 t=note, ref=commit id
    cfile = root / "branches" / "alice" / b.meta.branch_id / "commits" / "0001.jsonl"
    cl = cfile.read_text(encoding="utf-8").splitlines()
    assert len(cl) == 2
    m0, n0 = json.loads(cl[0]), json.loads(cl[1])
    assert m0["t"] == "commit" and m0["seq"] == 1 and m0["id"] == view.id
    assert n0["t"] == "note" and n0["ref"] == view.id
    # note body 含 Kind trailer
    text, trailers = split_trailers(n0["body"])
    assert text == "golden"
    assert trailer_values(trailers, "Kind") == ["semantic"]
    assert trailer_values(trailers, "Thread") == ["t"]

    # id 前缀
    assert view.id.startswith("cmt_")
    assert b.meta.branch_id.startswith("brn_")

    # HEAD.json
    head_json = json.loads((root / "branches" / "alice" / "HEAD.json").read_text())
    assert head_json == {"current": b.meta.branch_id}

    # 时间戳带时区偏移 (FORMAT.md §2: MUST 带偏移)
    assert re.search(r"[+-]\d{2}:\d{2}$", m0["created"])


# ============================================================
# 硬条款 1: 字节等价 — 独立读取器读 FsMemento 的输出, 结构等价
# ============================================================


def _hand_read_head(root: Path, owner: str) -> dict:
    """纯 stdlib 读取, 复原 (head_commit_id, summary, moment_ids, payloads) 视图."""
    head_json = json.loads((root / "branches" / owner / "HEAD.json").read_text())
    bid = head_json["current"]
    bdir = root / "branches" / owner / bid
    cfiles = sorted((bdir / "commits").glob("*.jsonl"), key=lambda p: int(p.stem))
    latest = cfiles[-1]
    lines = latest.read_text(encoding="utf-8").splitlines()
    member = json.loads(lines[0])
    notes = [json.loads(line) for line in lines[1:] if json.loads(line)["t"] == "note"]
    body = notes[-1]["body"]  # last-wins
    # 展开 payloads: 扫所有池文件
    payloads = {}
    threads_of = {}
    for f in (root / "moments" / owner).rglob("moments.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if obj["t"] == "moment":
                payloads[obj["id"]] = obj["payload"]
                threads_of[obj["id"]] = obj.get("threads", [])
            elif obj["t"] == "note":
                threads_of[obj["ref"]] = obj["threads"]  # last-wins
    return {
        "commit_id": member["id"],
        "moment_ids": member["moment_ids"],
        "summary": body.split("\n\n")[0] if "\n\n" in body else "",
        "payloads": {mid: payloads[mid] for mid in member["moment_ids"]},
        "threads": {mid: threads_of.get(mid, []) for mid in member["moment_ids"]},
    }


def test_fs_output_reads_equivalent_via_stdlib(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "alice")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    b.update(MomentRecord(id="m2", type="test.data/v1", payload={"n": 2}, threads=["x"]))
    view = b.commit("cross-read", kind="semantic")
    m.close()

    # FsMemento 视图
    m2 = new_filesystem_memento(root, "alice")
    fs_view = {
        "commit_id": view.id,
        "moment_ids": ["m1", "m2"],
        "summary": "cross-read",
        "payloads": {"m1": {"n": 1}, "m2": {"n": 2}},
        "threads": {"m1": [], "m2": ["x"]},
    }
    hand = _hand_read_head(root, "alice")
    assert hand == fs_view
    m2.close()


# ============================================================
# 硬条款 1: 撕裂尾行与前向兼容
# ============================================================


def test_torn_tail_line_is_tolerated(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    b.commit("s", kind="mechanical")
    m.close()

    # 追加撕裂尾行
    mfile = next((root / "moments" / "o").rglob("moments.jsonl"))
    with mfile.open("a", encoding="utf-8") as f:
        f.write('{"t":"moment","id":"m2","incompl')

    m2 = new_filesystem_memento(root, "o")
    # 应仍能读到 m1
    assert m2.pool.get("m1").payload == {"n": 1}
    # 撕裂行不成 record
    assert m2.pool.get("m2") is None


def test_unknown_t_field_is_ignored(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={}))
    b.commit("s", kind="mechanical")
    m.close()

    mfile = next((root / "moments" / "o").rglob("moments.jsonl"))
    with mfile.open("a", encoding="utf-8") as f:
        f.write('{"t":"future-thing-v99","x":1}\n')

    m2 = new_filesystem_memento(root, "o")
    assert m2.pool.get("m1") is not None  # 未知 t 静默跳过, 不影响


# ============================================================
# 硬条款 1: 索引可再生 — 删 .cache/ 后行为不变
# 本参考实现无 .cache/, 属于最平凡满足. 显式测: 无 .cache/ 也运行正常.
# ============================================================


def test_cache_directory_is_regenerable(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("s", kind="mechanical")
    m.close()

    # 若存在 .cache/, 删掉
    cache = root / ".cache"
    if cache.exists():
        import shutil
        shutil.rmtree(cache)

    m2 = new_filesystem_memento(root, "o")
    assert m2.current().head().id == view.id
    assert m2.pool.get("m1").payload == {"n": 1}


# ============================================================
# 硬条款 2: 退化态纯净 — 蠢记忆用例, fork 词汇一个不出现
# 用 inspect + regex 静态扫描本函数源码, 保证契约级纯净.
# ============================================================


def test_dumb_memory_degenerate_form(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "single-user")
    conversation = m.current()

    conversation.update(MomentRecord(id="turn-1", type="test.data/v1", payload={"user": "hi"}))
    conversation.commit("greeting", kind="mechanical")
    conversation.update(MomentRecord(id="turn-2", type="test.data/v1", payload={"user": "who r u"}))
    conversation.commit("intro", kind="mechanical")

    history = conversation.all_commits()
    assert [v.summary() for v in history] == ["greeting", "intro"]
    m.close()


def test_dumb_memory_source_has_no_fork_vocabulary():
    """契约级条款: 退化态用例代码里, fork 相关词汇一个都不出现 (FEATURE.md §8)."""
    src = inspect.getsource(test_dumb_memory_degenerate_form)
    banned = ["fork", "checkout", "ancestry", "overlay", "base", "reinterpret", "MementoRef"]
    hits = [w for w in banned if re.search(rf"\b{w}\b", src)]
    assert hits == [], f"fork-family vocabulary leaked into dumb-memory test: {hits}"
