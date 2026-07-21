"""
Golden tests — FORMAT.md v1.1 契约锚点.

两条硬条款:
1. 字节等价: 独立照 FORMAT.md 各写一份历史, 互读对方字节, 重建等价.
   本文件里的 "独立实现" 就是纯 stdlib (json + pathlib) 直接照 FORMAT.md
   拼字节, 与 FsMemento 无共享代码路径.
2. 退化态纯净: 单 branch + 自动 commit 的用例代码里, fork 词汇一个都不出现.

§14 布局: 无独立 moment 池, staging 持真身, commit 文件自包含.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime
from pathlib import Path

from ghoshell_moss.memento import (
    MomentRecord,
    new_filesystem_memento,
    split_trailers,
    trailer_values,
)


# ============================================================
# 独立最小写入器: 只依赖 json + pathlib, 照 FORMAT.md §14 手工拼字节.
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
    """按 FORMAT.md §14 逐字节拼一份最小历史 (commit 文件自包含, 无 pool)."""
    now = datetime.now().astimezone().isoformat()

    root.mkdir(parents=True, exist_ok=True)

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

    # commits/NNNN.jsonl: 成员行 + m 行冻结 moment + 初始 commit_note
    cdir = bdir / "commits"
    cdir.mkdir(exist_ok=True)
    lines: list[dict] = []
    member = {
        "t": "commit", "id": commit_id, "seq": commit_seq,
        "moment_ids": [r["id"] for r in records], "created": now,
    }
    lines.append(member)
    for r in records:
        mline = {
            "t": "moment",
            "id": r["id"],
            "created": r.get("created", now),
            "type": r.get("type", "test.data/v1"),
            "payload": r["payload"],
        }
        if r.get("threads"):
            mline["threads"] = r["threads"]
        lines.append(mline)
    note = {"t": "commit_note", "ref": commit_id, "body": body, "ts": now}
    lines.append(note)
    with (cdir / f"{commit_seq:04d}.jsonl").open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

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
# 硬条款 1: 字节等价 — FsMemento -> stdlib 校验字节
# ============================================================


def test_fs_written_history_conforms_to_format(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "alice")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"text": "早"}))
    b.update(MomentRecord(id="m2", type="test.data/v1", payload={"text": "line1\nline2"}))
    view = b.commit("golden", kind="semantic", threads=["t"])
    m.close()

    # §14: 无独立 pool 目录, moment 真身在 commit 文件里
    assert not (root / "moments").exists(), "§14: moments/ 目录已废除"

    # commit 文件: 首行 t=commit, 中间 m 行 t=moment (冻结全文), 末尾 t=commit_note
    cfile = root / "branches" / "alice" / b.meta.branch_id / "commits" / "0001.jsonl"
    text = cfile.read_text(encoding="utf-8")
    cl = text.splitlines()
    assert len(cl) == 4, "member + 2 moments + 1 commit_note"
    m0 = json.loads(cl[0])
    m1_line = json.loads(cl[1])
    m2_line = json.loads(cl[2])
    n0 = json.loads(cl[3])
    assert m0["t"] == "commit" and m0["seq"] == 1 and m0["id"] == view.id
    assert m0["moment_ids"] == ["m1", "m2"]
    assert m1_line["t"] == "moment" and m1_line["id"] == "m1"
    assert m1_line["payload"] == {"text": "早"}
    assert m2_line["t"] == "moment" and m2_line["id"] == "m2"
    assert m2_line["payload"] == {"text": "line1\nline2"}
    assert n0["t"] == "commit_note" and n0["ref"] == view.id

    # note body 含 Kind trailer
    body_text, trailers = split_trailers(n0["body"])
    assert body_text == "golden"
    assert trailer_values(trailers, "Kind") == ["semantic"]
    assert trailer_values(trailers, "Thread") == ["t"]

    # 非 ASCII 无转义 (ensure_ascii=False)
    assert "早" in text
    # 物理行内换行按 JSON 标准转义
    assert r"line1\nline2" in text

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
    """
    纯 stdlib 读取, 复原 (head_commit_id, summary, moment_ids, payloads) 视图.
    §14: 所有身份直接在 commit 文件里, 一次读一个文件即得整视图.
    """
    head_json = json.loads((root / "branches" / owner / "HEAD.json").read_text())
    bid = head_json["current"]
    bdir = root / "branches" / owner / bid
    cfiles = sorted((bdir / "commits").glob("*.jsonl"), key=lambda p: int(p.stem))
    latest = cfiles[-1]
    objs = [json.loads(line) for line in latest.read_text(encoding="utf-8").splitlines()]
    member = objs[0]
    assert member["t"] == "commit"
    payloads: dict[str, dict] = {}
    threads_of: dict[str, list[str]] = {}
    body = ""
    for obj in objs[1:]:
        t = obj["t"]
        if t == "moment":
            payloads[obj["id"]] = obj["payload"]
            threads_of[obj["id"]] = obj.get("threads", [])
        elif t == "moment_note":
            threads_of[obj["ref"]] = obj["threads"]  # last-wins
        elif t == "commit_note":
            body = obj["body"]  # last-wins
    return {
        "commit_id": member["id"],
        "moment_ids": member["moment_ids"],
        "summary": body.split("\n\n")[0] if "\n\n" in body else body,
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

    fs_view = {
        "commit_id": view.id,
        "moment_ids": ["m1", "m2"],
        "summary": "cross-read",
        "payloads": {"m1": {"n": 1}, "m2": {"n": 2}},
        "threads": {"m1": [], "m2": ["x"]},
    }
    hand = _hand_read_head(root, "alice")
    assert hand == fs_view


# ============================================================
# 硬条款 1: 撕裂尾行与前向兼容
# ============================================================


def test_torn_tail_line_is_tolerated(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("s", kind="mechanical")
    m.close()

    # 追加撕裂尾行到 commit 文件末尾 (无换行结尾)
    cfile = root / "branches" / "o" / b.meta.branch_id / "commits" / "0001.jsonl"
    with cfile.open("a", encoding="utf-8") as f:
        f.write('{"t":"moment_note","ref":"m1","incompl')

    m2 = new_filesystem_memento(root, "o")
    b2 = m2.current()
    # 应仍能读到 commit 与 m1 的正常 record
    records = b2.commit_records(view.id)
    assert len(records) == 1
    assert records[0].id == "m1"
    assert records[0].payload == {"n": 1}


def test_unknown_t_field_is_ignored(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={}))
    view = b.commit("s", kind="mechanical")
    m.close()

    cfile = root / "branches" / "o" / b.meta.branch_id / "commits" / "0001.jsonl"
    with cfile.open("a", encoding="utf-8") as f:
        f.write('{"t":"future-thing-v99","x":1}\n')

    m2 = new_filesystem_memento(root, "o")
    b2 = m2.current()
    # 未知 t 静默跳过, 不影响正常读取
    records = b2.commit_records(view.id)
    assert len(records) == 1
    assert records[0].id == "m1"


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
    b2 = m2.current()
    assert b2.head().id == view.id
    records = b2.commit_records(view.id)
    assert records[0].payload == {"n": 1}


# ============================================================
# §14 崩溃恢复: commit 落盘后 truncate staging 前中断, 装入时幂等收敛
# ============================================================


def test_crash_recovery_truncates_stale_staging(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("committed", kind="mechanical")
    m.close()

    # 模拟崩溃: 手工把 m1 的原始 stage 记录塞回 staging.jsonl
    staging = root / "branches" / "o" / b.meta.branch_id / "staging.jsonl"
    staging.write_text(
        json.dumps({"t": "moment", "id": "m1", "created": "2026-07-19T00:00:00+08:00",
                    "type": "test.data/v1", "payload": {"n": 1}},
                   ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    # 重新装入: 装入路径应自动 truncate 该残留 (幂等)
    m2 = new_filesystem_memento(root, "o")
    b2 = m2.current()
    assert b2.staging() == []
    assert b2.head().id == view.id


# ============================================================
# 硬条款 2: 退化态纯净 — 蠢记忆用例, fork 词汇一个不出现
# 用 inspect + regex 静态扫描本函数源码, 保证契约级纯净.
# ============================================================


# ============================================================
# §14: checkout from (commit_id, moment_id) — commit 内前缀切片
# ============================================================


def test_checkout_from_moment_id_slices_commit_inclusive(tmp_path: Path):
    """(commit, moment_id) 化身: ancestry 最末 commit 只贡献到该 moment 为止 (inclusive)."""
    from ghoshell_moss.memento import MomentNotInCommitError

    root = tmp_path / "memento"
    src = new_filesystem_memento(root, "alpha")
    b = src.current()
    b.update(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    b.update(MomentRecord(id="m2", type="test.data/v1", payload={"n": 2}))
    b.update(MomentRecord(id="m3", type="test.data/v1", payload={"n": 3}))
    b.update(MomentRecord(id="m4", type="test.data/v1", payload={"n": 4}))
    view = b.commit("full commit", kind="mechanical")

    # inclusive 切到 m3 — m1..m3 应可见, m4 应缺席
    beta = new_filesystem_memento(root, "beta")
    forked = beta.checkout(
        base_fork="alpha",
        base_branch_id=b.meta.branch_id,
        base_commit_id=view.id,
        base_moment_id="m3",
        name="slice",
    )
    assert forked.meta.base is not None
    assert forked.meta.base.moment_id == "m3"
    assert forked.meta.base.moment_seq == 2

    records = forked.commit_records(view.id)
    assert [r.id for r in records] == ["m1", "m2", "m3"], "inclusive 切片, m4 不应可见"

    win = forked.window(detail_n=10, summary_m=-1)
    detail_ids = [r.id for r in win.details]
    assert "m4" not in detail_ids, f"m4 泄漏进 window: {detail_ids}"

    # 篡改测试: 无效 moment_id 应抛
    try:
        beta.checkout(
            base_fork="alpha",
            base_branch_id=b.meta.branch_id,
            base_commit_id=view.id,
            base_moment_id="does-not-exist",
        )
        raise AssertionError("should have raised MomentNotInCommitError")
    except MomentNotInCommitError:
        pass


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
