"""
Golden tests — FORMAT v3 contract anchors.

Two hard clauses:
1. Byte equivalence: independently write a history per FORMAT.md, read each
   other's bytes, reconstruct equivalence. The "independent implementation"
   in this file uses pure stdlib (json + pathlib) + ulid to assemble bytes
   per FORMAT.md, sharing no code path with FsMemento.
2. Degenerate-form purity: single-line + auto-commit usage code contains
   zero fork / branch / confluent vocabulary.

v3 layout: ws/{branch_uid}/ workspaces, heads/{name} pointers, commit
autonomous directories with Y-m bucketing, moments.jsonl (t:"moment"
rows only, no commit header), notes.jsonl, content field on moments.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timezone as tz
from pathlib import Path

from ulid import ULID

from ghoshell_moss.memento import _storage as store
from ghoshell_moss.memento import new_filesystem_memento
from ghoshell_moss.memento.abc import (
    BranchRef,
    MomentRecord,
    new_branch_id,
    new_commit_id,
    split_trailers,
    trailer_values,
)


# ============================================================
# Independent minimal writer: pure json + pathlib + ulid,
# assembles bytes per FORMAT v3 by hand.
# Used to feed FsMemento for read-back verification.
# ============================================================


def _hand_write_history_v3(
    root: Path,
    owner: str,
    branch_uid: str,
    head_name: str,
    records: list[dict],
    commit_id: str,
    body: str,
) -> None:
    """Assemble a minimal v3 history by hand (FORMAT v3 layout)."""
    now = datetime.now(tz.utc).isoformat()
    root.mkdir(parents=True, exist_ok=True)

    # Y-m from ULID timestamp (UTC)
    ulid = ULID.from_str(commit_id[len(store.COMMIT_ID_PREFIX):])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")

    owner_dir = root / owner
    owner_dir.mkdir(parents=True, exist_ok=True)

    # heads/{name} → uid pointer
    heads_dir = owner_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    (heads_dir / head_name).write_text(f"{branch_uid}\n", encoding="utf-8")

    # ws/{uid}/ref
    ws_dir = owner_dir / "ws" / branch_uid
    ws_dir.mkdir(parents=True, exist_ok=True)
    ref = {"origin": owner, "commit_id": commit_id}
    (ws_dir / "ref").write_text(
        json.dumps(ref, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    # ws/{uid}/staging.jsonl (empty after commit)
    (ws_dir / "staging.jsonl").write_text("", encoding="utf-8")

    # commits.jsonl
    cr_line = {
        "t": "commit_ref",
        "commit_id": commit_id,
        "branch_uid": branch_uid,
        "parent": None,
        "ts": now,
        "kind": "semantic",
    }
    (owner_dir / "commits.jsonl").write_text(
        json.dumps(cr_line, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    # branches.jsonl
    fork_ref = {"origin": "", "commit_id": ""}
    br_line = {
        "t": "branch_meta",
        "uid": branch_uid,
        "name": head_name,
        "status": "active",
        "fork_ref": fork_ref,
        "created": now,
        "updated": now,
    }
    (owner_dir / "branches.jsonl").write_text(
        json.dumps(br_line, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    # commits/{Y-m}/{commit_id}/
    cdir = owner_dir / "commits" / ym / commit_id
    cdir.mkdir(parents=True, exist_ok=True)

    # meta.json
    meta = {
        "commit_id": commit_id,
        "parent": None,
        "kind": "semantic",
        "created": now,
    }
    (cdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # moments.jsonl: t:"moment" rows only (v3 — no commit header row)
    lines: list[dict] = []
    for r in records:
        mline = {
            "t": "moment",
            "id": r["id"],
            "created": r.get("created", now),
            "type": r.get("type", "test.data/v1"),
            "content": r.get("content", ""),
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
# Hard clause 1: byte equivalence — hand-written → FsMemento read-back
# ============================================================


def test_hand_written_v3_history_readable_by_fs(tmp_path: Path):
    root = tmp_path / "memento"
    cid = new_commit_id()
    buid = new_branch_id()
    _hand_write_history_v3(
        root,
        owner="alice",
        branch_uid=buid,
        head_name="main",
        records=[
            {"id": "m1", "payload": {"text": "早"}, "content": "good morning"},
            {"id": "m2", "payload": {"text": "line1\nline2"}, "threads": ["t"],
             "content": "two lines"},
        ],
        commit_id=cid,
        body="hand-written history\n\nThread: t\nKind: semantic",
    )

    m = new_filesystem_memento(root, "alice")
    b = m.get_line("main")
    assert b.branch_identifier == buid
    assert b.name == "main"
    head = b.log()[-1]
    assert head.id == cid
    assert head.summary() == "hand-written history"
    assert head.note.threads() == ["t"]
    assert trailer_values(head.note.trailers(), "Kind") == ["semantic"]
    records = m.show(cid).moments
    assert records[0].payload == {"text": "早"}
    assert records[0].content == "good morning"
    assert records[1].payload == {"text": "line1\nline2"}
    assert records[1].threads == ["t"]
    assert records[1].content == "two lines"


# ============================================================
# Hard clause 1: byte equivalence — FsMemento → stdlib byte-scan
# ============================================================


def test_fs_written_v3_conforms_to_format(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "alice")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1",
                          payload={"text": "早"}, content="hello"))
    b.record(MomentRecord(id="m2", type="test.data/v1",
                          payload={"text": "line1\nline2"}, content="two"))
    view = b.commit("golden", kind="semantic", threads=["t"])
    buid = b.branch_identifier

    # No legacy pool directory
    assert not (root / "moments").exists(), "moments/ pool is abolished"

    # Commit autonomous directory (Y-m bucketed)
    ulid = ULID.from_str(view.id[len(store.COMMIT_ID_PREFIX):])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")
    cdir = root / "alice" / "commits" / ym / view.id
    assert cdir.exists()

    # moments.jsonl: t:"moment" rows only (v3 — no commit header)
    mfile = cdir / "moments.jsonl"
    text = mfile.read_text(encoding="utf-8")
    cl = text.splitlines()
    assert len(cl) == 2, "v3: only moment rows, no commit header"
    m0 = json.loads(cl[0])
    m1 = json.loads(cl[1])
    assert m0["t"] == "moment" and m0["id"] == "m1"
    assert m0["payload"] == {"text": "早"}
    assert m0["content"] == "hello"
    assert m1["t"] == "moment" and m1["id"] == "m2"
    assert m1["payload"] == {"text": "line1\nline2"}
    assert m1["content"] == "two"
    # No moment_ids header row — ids are on the moment rows themselves
    assert "moment_ids" not in m0

    # notes.jsonl: commit_note
    nfile = cdir / "notes.jsonl"
    ntext = nfile.read_text(encoding="utf-8")
    nl = ntext.splitlines()
    assert len(nl) == 1
    n0 = json.loads(nl[0])
    assert n0["t"] == "commit_note" and n0["ref"] == view.id

    body_text, trailers = split_trailers(n0["body"])
    assert body_text == "golden"
    # Thread trailer is present; Kind trailer is optional for semantic commits
    assert trailer_values(trailers, "Thread") == ["t"]

    # Non-ASCII without escaping (ensure_ascii=False)
    assert "早" in text
    # Physical newline escaped per JSON standard
    assert r"line1\nline2" in text

    # ID prefixes
    assert view.id.startswith(store.COMMIT_ID_PREFIX)

    # heads/{name} → uid pointer
    head = (root / "alice" / "heads" / "main").read_text().strip()
    assert head == buid

    # ws/{uid}/ref — origin is empty for same-owner (FORMAT v3 §4.2)
    ref = json.loads((root / "alice" / "ws" / buid / "ref").read_text())
    assert ref["commit_id"] == view.id
    assert ref["origin"] in ("", "alice")

    # commits.jsonl has branch_uid (not name)
    clines = (root / "alice" / "commits.jsonl").read_text().splitlines()
    assert len(clines) == 1
    cr = json.loads(clines[0])
    assert cr["t"] == "commit_ref" and cr["commit_id"] == view.id
    assert cr["branch_uid"] == buid

    # Timestamp carries timezone indicator (FORMAT.md §2: Z or +HH:MM)
    assert re.search(r"(Z|[+-]\d{2}:\d{2})$", m0["created"])


# ============================================================
# Hard clause 1: byte equivalence — independent reader
# ============================================================


def _hand_read_head_v3(root: Path, owner: str, head_name: str) -> dict:
    """Pure stdlib + ulid read: reconstruct (commit_id, summary, moment_ids,
    payloads, content) from v3 layout.

    v3: commits.jsonl → latest commit_id → Y-m from ULID → read
    moments.jsonl (t:"moment" rows, no header) + notes.jsonl.
    """
    # Resolve head name → uid
    head_path = root / owner / "heads" / head_name
    branch_uid = head_path.read_text(encoding="utf-8").strip()

    # commits.jsonl → latest commit_id
    cjl = root / owner / "commits.jsonl"
    objs = [
        json.loads(line)
        for line in cjl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    last = objs[-1]
    cid = last["commit_id"]
    assert last["branch_uid"] == branch_uid

    # Y-m from ULID
    ulid = ULID.from_str(cid[len(store.COMMIT_ID_PREFIX):])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")

    # moments.jsonl: t:"moment" rows (no commit header)
    cdir = root / owner / "commits" / ym / cid
    mobjs = [
        json.loads(line)
        for line in (cdir / "moments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    moment_ids: list[str] = []
    payloads: dict[str, dict] = {}
    content_map: dict[str, str] = {}
    threads_of: dict[str, list[str]] = {}
    for obj in mobjs:
        if obj.get("t") == "moment":
            mid = obj["id"]
            moment_ids.append(mid)
            payloads[mid] = obj["payload"]
            content_map[mid] = obj.get("content", "")
            threads_of[mid] = obj.get("threads", [])

    # notes.jsonl → body
    npath = cdir / "notes.jsonl"
    body = ""
    if npath.exists():
        for obj in [
            json.loads(line)
            for line in npath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]:
            if obj.get("t") == "commit_note":
                body = obj["body"]
    text_body, _ = split_trailers(body)

    return {
        "commit_id": cid,
        "moment_ids": moment_ids,
        "summary": text_body.strip(),
        "payloads": payloads,
        "content": content_map,
        "threads": threads_of,
    }


def test_fs_v3_output_reads_equivalent_via_stdlib(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "alice")
    b = m.create_line("main")
    buid = b.branch_identifier
    b.record(MomentRecord(id="m1", type="test.data/v1",
                          payload={"n": 1}, content="first"))
    b.record(MomentRecord(id="m2", type="test.data/v1",
                          payload={"n": 2}, threads=["x"], content="second"))
    view = b.commit("cross-read", kind="semantic")

    fs_view = {
        "commit_id": view.id,
        "moment_ids": ["m1", "m2"],
        "summary": "cross-read",
        "payloads": {"m1": {"n": 1}, "m2": {"n": 2}},
        "content": {"m1": "first", "m2": "second"},
        "threads": {"m1": [], "m2": ["x"]},
    }
    hand = _hand_read_head_v3(root, "alice", "main")
    assert hand == fs_view


# ============================================================
# Hard clause 1: torn tail line & forward compatibility
# ============================================================


def test_torn_tail_line_is_tolerated(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("s", kind="mechanical")

    ulid = ULID.from_str(view.id[len(store.COMMIT_ID_PREFIX):])
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

    ulid = ULID.from_str(view.id[len(store.COMMIT_ID_PREFIX):])
    ym = datetime.fromtimestamp(ulid.timestamp, tz=tz.utc).strftime("%Y-%m")
    nfile = root / "o" / "commits" / ym / view.id / "notes.jsonl"
    with nfile.open("a", encoding="utf-8") as f:
        f.write('{"t":"future-thing-v99","x":1}\n')

    m2 = new_filesystem_memento(root, "o")
    records = m2.show(view.id).moments
    assert len(records) == 1
    assert records[0].id == "m1"


# ============================================================
# Hard clause 1: no .cache/ — structure does not need indexing
# ============================================================


def test_no_cache_directory_pollution(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("s", kind="mechanical")

    assert not (root / ".cache").exists(), "v3 must not produce .cache/"

    m2 = new_filesystem_memento(root, "o")
    b2 = m2.get_line("main")
    assert b2.log()[-1].id == view.id
    records = m2.show(view.id).moments
    assert records[0].payload == {"n": 1}


# ============================================================
# Crash recovery: commit persisted but staging not truncated
# ============================================================


def test_crash_recovery_truncates_stale_staging(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    view = b.commit("committed", kind="mechanical")
    buid = b.branch_identifier

    # Simulate crash: re-insert m1 into staging manually
    staging = root / "o" / "ws" / buid / "staging.jsonl"
    staging.write_text(
        json.dumps(
            {
                "t": "moment",
                "id": "m1",
                "created": "2026-08-04T00:00:00+00:00",
                "type": "test.data/v1",
                "content": "",
                "payload": {"n": 1},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    m2 = new_filesystem_memento(root, "o")
    b2 = m2.get_line("main")
    assert b2.staging() == []
    assert b2.log()[-1].id == view.id


# ============================================================
# Content field round-trip
# ============================================================


def test_content_field_round_trip(tmp_path: Path):
    """v3: MomentRecord.content survives write → read cycle."""
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("main")
    b.record(MomentRecord(id="m1", type="test/v1",
                          payload={"x": 1}, content="the answer is 42"))
    view = b.commit("content test", kind="mechanical")
    records = m.show(view.id).moments
    assert records[0].content == "the answer is 42"


# ============================================================
# Branch uid stability: name delete doesn't kill workspace
# ============================================================


def test_delete_line_preserves_workspace_and_commits(tmp_path: Path):
    """v3: delete_line removes the head file only. Workspace and commits survive."""
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    b = m.create_line("temp-branch")
    buid = b.branch_identifier
    b.record(MomentRecord(id="m1", type="test/v1", payload={}))
    view = b.commit("anchor", kind="mechanical")

    m.delete_line("temp-branch")
    assert "temp-branch" not in m.list_lines()

    # Workspace still exists
    assert (root / "o" / "ws" / buid).exists()
    # Commit still accessible
    assert m.show(view.id).commit.id == view.id
    # Branch appears in list_all_branches
    all_uids = {br.uid for br in m.list_all_branches()}
    assert buid in all_uids


# ============================================================
# Degenerate path: get_line("main") auto-creates on first use
# ============================================================


def test_get_line_main_auto_creates(tmp_path: Path):
    """v3 degenerate form: get_line('main') implicitly creates the line."""
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    line = m.get_line("main")
    assert line.name == "main"
    assert line.branch_identifier.startswith(store.BRANCH_ID_PREFIX)
    assert "main" in m.list_lines()
    # Verify it actually works
    line.record(MomentRecord(id="m1", type="test/v1", payload={}))
    view = line.commit("auto", kind="mechanical")
    assert view.id.startswith(store.COMMIT_ID_PREFIX)


# ============================================================
# checkouts.jsonl correctness
# ============================================================


def test_checkouts_jsonl_records_fork_events(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    a = m.create_line("a")
    a.record(MomentRecord(id="a1", type="test/v1", payload={}))
    va = a.commit("a anchor", kind="semantic")

    b = m.create_line("b", from_ref=BranchRef(commit_id=va.id))

    checkouts = m.checkouts()
    assert len(checkouts) == 2  # a (root) + b (fork from a)
    # Last checkout should be b forking from a
    last = checkouts[-1]
    assert last.branch_uid == b.branch_identifier
    assert last.from_ref.commit_id == va.id

    # Forward: branches list includes both
    assert set(m.list_lines()) == {"a", "b"}


# ============================================================
# Confluent record (write side placeholder — test that the
# storage path exists and reads correctly)
# ============================================================


def test_confluents_jsonl_empty_by_default(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    m.get_line("main")  # auto-create
    assert m.confluents() == []


# ============================================================
# list_all_branches returns correct entries
# ============================================================


def test_list_all_branches_indexes_every_branch(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "o")
    m.create_line("a")
    m.get_line("main")  # auto-create
    m.create_line("b")

    branches = m.list_all_branches()
    assert len(branches) == 3
    names = {br.name for br in branches}
    assert names == {"a", "main", "b"}
    for br in branches:
        assert br.uid.startswith(store.BRANCH_ID_PREFIX)
        assert br.status == "active"


# ============================================================
# checkout from (commit_id, moment_id) — commit prefix slice
# ============================================================


def test_checkout_from_moment_id_slices_commit_inclusive(tmp_path: Path):
    """(commit, moment_id) checkout: ref records moment_id slice point."""
    root = tmp_path / "memento"
    src = new_filesystem_memento(root, "alpha")
    b = src.create_line("main")
    b.record(MomentRecord(id="m1", type="test.data/v1", payload={"n": 1}))
    b.record(MomentRecord(id="m2", type="test.data/v1", payload={"n": 2}))
    b.record(MomentRecord(id="m3", type="test.data/v1", payload={"n": 3}))
    b.record(MomentRecord(id="m4", type="test.data/v1", payload={"n": 4}))
    view = b.commit("full commit", kind="mechanical")

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

    all_moments = src.show(view.id).moments
    assert [r.id for r in all_moments] == ["m1", "m2", "m3", "m4"]

    m3_idx = next(i for i, r in enumerate(all_moments) if r.id == "m3")
    sliced = all_moments[: m3_idx + 1]
    assert [r.id for r in sliced] == ["m1", "m2", "m3"]

    # moment_id slice is recorded in ref for consumers to use at read time
    assert forked.ref.moment_id == "m3"


# ============================================================
# Hard clause 2: degenerate-form purity — no fork/confluent
# vocabulary in dumb-memory usage code.
# ============================================================


def test_dumb_memory_degenerate_form(tmp_path: Path):
    root = tmp_path / "memento"
    m = new_filesystem_memento(root, "single-user")
    conversation = m.get_line("main")  # v3: auto-create, no explicit create_line

    conversation.record(MomentRecord(id="turn-1", type="test.data/v1",
                                      payload={"user": "hi"}, content="hi"))
    conversation.commit("greeting", kind="mechanical")
    conversation.record(MomentRecord(id="turn-2", type="test.data/v1",
                                      payload={"user": "who r u"}, content="who"))
    conversation.commit("intro", kind="mechanical")

    history = conversation.log()
    assert [v.summary() for v in history] == ["greeting", "intro"]


def test_dumb_memory_source_has_no_fork_or_confluent_vocabulary():
    """Contract-level: degenerate form must not contain fork/confluent vocabulary."""
    src = inspect.getsource(test_dumb_memory_degenerate_form)
    banned = [
        "fork", "checkout", "ancestry", "overlay",
        "confluent", "merge", "MementoRef",
    ]
    hits = [w for w in banned if re.search(rf"\b{w}\b", src)]
    assert hits == [], (
        f"fork/confluent-family vocabulary leaked into dumb-memory test: {hits}"
    )
