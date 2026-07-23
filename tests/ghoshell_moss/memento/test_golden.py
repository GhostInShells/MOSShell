"""
Golden tests — FORMAT v2 契约锚点.

两条硬条款:
1. 字节等价: 独立照 FORMAT.md 各写一份历史, 互读对方字节, 重建等价.
   本文件里的 "独立实现" 就是纯 stdlib (json + pathlib) + ulid 直接照 FORMAT.md
   拼字节, 与 FsMemento 无共享代码路径.
2. 退化态纯净: 单 line + 自动 commit 的用例代码里, fork 词汇一个都不出现.

v2 布局: commit 自治目录 + Y-m 分桶, staging 持真身, notes.jsonl 释义.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timezone as tz
from pathlib import Path

from ulid import ULID

from ghoshell_moss.memento import (
    BranchRef,
    MomentNotInCommitError,
    MomentRecord,
    new_filesystem_memento,
    new_commit_id,
    split_trailers,
    trailer_values,
)


# ============================================================
# 独立最小写入器: 只依赖 json + pathlib + ulid, 照 FORMAT v2 手工拼字节.
# 用来给 FsMemento 读; 反过来 FsMemento 写的也用 stdlib 校验.
# ============================================================


def _hand_write_history(
    root: Path,
    owner: str,
    line_name: str,
    records: list[dict],
    commit_id: str,
    body: str,
) -> None:
    """按 FORMAT v2 逐字节拼一份最小历史 (commit 自治目录, Y-m 分桶)."""
    now = datetime.now().astimezone().isoformat()
    root.mkdir(parents=True, exist_ok=True)

    # Y-m from ULID timestamp (UTC)
    ulid = ULID.from_str(commit_id[4:])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")

    owner_dir = root / owner
    owner_dir.mkdir(parents=True, exist_ok=True)

    # branches/{name}/ref
    bdir = owner_dir / "branches" / line_name
    bdir.mkdir(parents=True, exist_ok=True)
    ref = {"origin": owner, "commit_id": commit_id}
    (bdir / "ref").write_text(
        json.dumps(ref, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    # staging.jsonl (empty)
    (bdir / "staging.jsonl").write_text("", encoding="utf-8")

    # commits.jsonl
    cr_line = {
        "t": "commit_ref",
        "commit_id": commit_id,
        "branch": line_name,
        "parent": None,
        "ts": now,
        "kind": "semantic",
    }
    (owner_dir / "commits.jsonl").write_text(
        json.dumps(cr_line, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    # commits/{Y-m}/{commit_id}/
    cdir = owner_dir / "commits" / ym / commit_id
    cdir.mkdir(parents=True, exist_ok=True)

    # meta.json
    meta = {
        "commit_id": commit_id,
        "parent": None,
        "branch": line_name,
        "kind": "semantic",
        "created": now,
    }
    (cdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # moments.jsonl: member line + m frozen moment lines
    moment_ids = [r["id"] for r in records]
    member = {
        "t": "commit",
        "id": commit_id,
        "moment_ids": moment_ids,
        "created": now,
    }
    lines: list[dict] = [member]
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
    with (cdir / "moments.jsonl").open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    # notes.jsonl: commit_note
    title = body.split("\n")[0].strip() if body else ""
    note = {
        "t": "commit_note",
        "ref": commit_id,
        "title": title,
        "body": body,
        "ts": now,
        "by": "",
    }
    (cdir / "notes.jsonl").write_text(
        json.dumps(note, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


# ============================================================
# 硬条款 1: 字节等价 — hand-written -> FsMemento 读
# ============================================================


def test_hand_written_history_readable_by_fs(tmp_path: Path):
    root = tmp_path / "memento"
    cid = new_commit_id()
    _hand_write_history(
        root,
        owner="alice",
        line_name="main",
        records=[
            {"id": "m1", "payload": {"text": "早"}},
            {"id": "m2", "payload": {"text": "line1\nline2"}, "threads": ["t"]},
        ],
        commit_id=cid,
        body="hand-written history\n\nThread: t\nKind: semantic",
    )

    m = new_filesystem_memento(root, "alice")
    b = m.get_line("main")
    head = b.log()[-1]
    assert head.id == cid
    assert head.summary() == "hand-written history"
    assert head.note.threads() == ["t"]
    assert trailer_values(head.note.trailers(), "Kind") == ["semantic"]
    records = m.show(cid).moments
    assert records[0].payload == {"text": "早"}
    assert records[1].payload == {"text": "line1\nline2"}
    assert records[1].threads == ["t"]


# ============================================================
# 硬条款 1: 字节等价 — FsMemento -> stdlib 校验字节
# ============================================================


def test_fs_written_history_conforms_to_format(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "alice")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"text": "早"}))
    b.record(MomentRecord(id="m2", type="test.data/v1", payload={"text": "line1\nline2"}))
    view = b.commit("golden", kind="semantic", threads=["t"])

    # 无独立 pool 目录
    assert not (root / "moments").exists(), "moments/ 目录已废除"

    # commit 自治目录 (Y-m 分桶)
    ulid = ULID.from_str(view.id[4:])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")
    cdir = root / "alice" / "commits" / ym / view.id
    assert cdir.exists()

    # moments.jsonl: 首行 t=commit, 后续 t=moment (冻结全文)
    mfile = cdir / "moments.jsonl"
    text = mfile.read_text(encoding="utf-8")
    cl = text.splitlines()
    assert len(cl) == 3, "member + 2 moments"
    m0 = json.loads(cl[0])
    m1_line = json.loads(cl[1])
    m2_line = json.loads(cl[2])
    assert m0["t"] == "commit" and m0["id"] == view.id
    assert m0["moment_ids"] == ["m1", "m2"]
    assert m1_line["t"] == "moment" and m1_line["id"] == "m1"
    assert m1_line["payload"] == {"text": "早"}
    assert m2_line["t"] == "moment" and m2_line["id"] == "m2"
    assert m2_line["payload"] == {"text": "line1\nline2"}

    # notes.jsonl: commit_note
    nfile = cdir / "notes.jsonl"
    ntext = nfile.read_text(encoding="utf-8")
    nl = ntext.splitlines()
    assert len(nl) == 1
    n0 = json.loads(nl[0])
    assert n0["t"] == "commit_note" and n0["ref"] == view.id

    # note body 含 Kind/Thread trailer
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

    # ref 文件
    ref = json.loads((root / "alice" / "branches" / "main" / "ref").read_text())
    assert ref["commit_id"] == view.id
    assert ref["origin"] == "alice"

    # commits.jsonl
    clines = (root / "alice" / "commits.jsonl").read_text().splitlines()
    assert len(clines) == 1
    cr = json.loads(clines[0])
    assert cr["t"] == "commit_ref" and cr["commit_id"] == view.id

    # 时间戳带时区偏移 (FORMAT.md §2: MUST 带偏移)
    assert re.search(r"[+-]\d{2}:\d{2}$", m0["created"])


# ============================================================
# 硬条款 1: 字节等价 — 独立读取器读 FsMemento 的输出, 结构等价
# ============================================================


def _hand_read_head(root: Path, owner: str) -> dict:
    """
    纯 stdlib + ulid 读取, 复原 (commit_id, summary, moment_ids, payloads) 视图.
    v2: commits.jsonl 找最新 commit → Y-m 分桶定位 → 读 moments.jsonl + notes.jsonl.
    """
    # commits.jsonl → latest commit_id
    cjl = root / owner / "commits.jsonl"
    objs = [json.loads(line) for line in cjl.read_text(encoding="utf-8").splitlines() if line.strip()]
    last = objs[-1]
    cid = last["commit_id"]

    # Y-m from ULID
    ulid = ULID.from_str(cid[4:])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")

    # moments.jsonl
    cdir = root / owner / "commits" / ym / cid
    mobjs = [
        json.loads(line)
        for line in (cdir / "moments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    member = mobjs[0]
    assert member["t"] == "commit"
    payloads: dict[str, dict] = {}
    threads_of: dict[str, list[str]] = {}
    for obj in mobjs[1:]:
        t = obj.get("t")
        if t == "moment":
            payloads[obj["id"]] = obj["payload"]
            threads_of[obj["id"]] = obj.get("threads", [])

    # notes.jsonl → body
    npath = cdir / "notes.jsonl"
    body = ""
    if npath.exists():
        for obj in [json.loads(line) for line in npath.read_text(encoding="utf-8").splitlines() if line.strip()]:
            if obj.get("t") == "commit_note":
                body = obj["body"]  # last-wins
    text_body, _ = split_trailers(body)

    return {
        "commit_id": member["id"],
        "moment_ids": member["moment_ids"],
        "summary": text_body.strip(),
        "payloads": {mid: payloads[mid] for mid in member["moment_ids"]},
        "threads": {mid: threads_of.get(mid, []) for mid in member["moment_ids"]},
    }


def test_fs_output_reads_equivalent_via_stdlib(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "alice")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    b.record(MomentRecord(id="m2", type="test.data/v1", payload={"n": 2}, threads=["x"]))
    view = b.commit("cross-read", kind="semantic")

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
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("s", kind="mechanical")

    # 追加撕裂尾行到 notes.jsonl 末尾 (无换行结尾)
    ulid = ULID.from_str(view.id[4:])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")
    nfile = root / "o" / "commits" / ym / view.id / "notes.jsonl"
    with nfile.open("a", encoding="utf-8") as f:
        f.write('{"t":"commit_note","ref":"x","incompl')

    m2 = new_filesystem_memento(root, "o")
    b2 = m2.get_line("main")
    records = m2.show(view.id).moments
    assert len(records) == 1
    assert records[0].id == "m1"
    assert records[0].payload == {"n": 1}


def test_unknown_t_field_is_ignored(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={}))
    view = b.commit("s", kind="mechanical")

    # 追加未知 t 行到 notes.jsonl
    ulid = ULID.from_str(view.id[4:])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")
    nfile = root / "o" / "commits" / ym / view.id / "notes.jsonl"
    with nfile.open("a", encoding="utf-8") as f:
        f.write('{"t":"future-thing-v99","x":1}\n')

    m2 = new_filesystem_memento(root, "o")
    records = m2.show(view.id).moments
    assert len(records) == 1
    assert records[0].id == "m1"


# ============================================================
# 硬条款 1: v2 无 .cache/ — 结构上不需要索引, 自然可再生
# ============================================================


def test_no_cache_directory_pollution(tmp_path: Path):
    """v2 消灭了 .cache/ — Y-m 分桶纯函数 + commits.jsonl 直读, 无需索引."""
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("s", kind="mechanical")

    # 检查无 .cache/ 目录
    cache = root / ".cache"
    assert not cache.exists(), "v2 不应产生 .cache/ 目录"

    # 重新装入, 无 .cache/ 也能正常工作
    m2 = new_filesystem_memento(root, "o")
    b2 = m2.get_line("main")
    assert b2.log()[-1].id == view.id
    records = m2.show(view.id).moments
    assert records[0].payload == {"n": 1}


# ============================================================
# 崩溃恢复: commit 落盘后 truncate staging 前中断, 装入时幂等收敛
# ============================================================


def test_crash_recovery_truncates_stale_staging(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("committed", kind="mechanical")

    # 模拟崩溃: 手工把 m1 的原始 stage 记录塞回 staging.jsonl
    staging = root / "o" / "branches" / "main" / "staging.jsonl"
    staging.write_text(
        json.dumps(
            {
                "t": "moment",
                "id": "m1",
                "created": "2026-07-19T00:00:00+08:00",
                "type": "test.data/v1",
                "payload": {"n": 1},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    # 重新装入: 装入路径应自动 truncate 该残留 (幂等)
    m2 = new_filesystem_memento(root, "o")
    b2 = m2.get_line("main")
    assert b2.staging() == []
    assert b2.log()[-1].id == view.id


# ============================================================
# checkout from (commit_id, moment_id) — commit 内前缀切片
# ============================================================


def test_checkout_from_moment_id_slices_commit_inclusive(tmp_path: Path):
    """(commit, moment_id) 化身: ref 记录 moment_id 切片点."""
    root = tmp_path / "memento"
    src = new_filesystem_memento(root, "alpha")
    b = src.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    b.record(MomentRecord(id="m2", type="test.data/v1", payload={"n": 2}))
    b.record(MomentRecord(id="m3", type="test.data/v1", payload={"n": 3}))
    b.record(MomentRecord(id="m4", type="test.data/v1", payload={"n": 4}))
    view = b.commit("full commit", kind="mechanical")

    # inclusive 切到 m3 — ref 带 moment_id
    beta = new_filesystem_memento(root, "beta")
    forked = beta.create_line(
        "slice",
        from_ref=BranchRef(
            origin="alpha",
            commit_id=view.id,
            moment_id="m3",
        ),
    )
    assert forked.ref is not None
    assert forked.ref.moment_id == "m3"
    assert forked.ref.origin == "alpha"

    # source commit 全量 moments 可查
    all_moments = src.show(view.id).moments
    assert [r.id for r in all_moments] == ["m1", "m2", "m3", "m4"]

    # 通过 ref 的 moment_id 手动切片验证: m1..m3
    m3_idx = next(i for i, r in enumerate(all_moments) if r.id == "m3")
    sliced = all_moments[: m3_idx + 1]
    assert [r.id for r in sliced] == ["m1", "m2", "m3"]

    # 无效 moment_id 应抛
    try:
        beta.create_line(
            "bad-slice",
            from_ref=BranchRef(
                origin="alpha",
                commit_id=view.id,
                moment_id="does-not-exist",
            ),
        )
        raise AssertionError("should have raised MomentNotInCommitError")
    except MomentNotInCommitError:
        pass


# ============================================================
# 硬条款 2: 退化态纯净 — 蠢记忆用例, fork 词汇一个不出现
# 用 inspect + regex 静态扫描本函数源码, 保证契约级纯净.
# ============================================================


def test_dumb_memory_degenerate_form(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "single-user")
    conversation = m.create_line("main")

    conversation.record(MomentRecord(id="turn-1", type="test.data/v1", payload={"user": "hi"}))
    conversation.commit("greeting", kind="mechanical")
    conversation.record(MomentRecord(id="turn-2", type="test.data/v1", payload={"user": "who r u"}))
    conversation.commit("intro", kind="mechanical")

    history = conversation.log()
    assert [v.summary() for v in history] == ["greeting", "intro"]


def test_dumb_memory_source_has_no_fork_vocabulary():
    """契约级条款: 退化态用例代码里, fork 相关词汇一个都不出现 (FEATURE.md §8)."""
    src = inspect.getsource(test_dumb_memory_degenerate_form)
    banned = ["fork", "checkout", "ancestry", "overlay", "base", "reinterpret", "MementoRef"]
    hits = [w for w in banned if re.search(rf"\b{w}\b", src)]
    assert hits == [], f"fork-family vocabulary leaked into dumb-memory test: {hits}"
