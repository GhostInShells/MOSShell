from ghoshell_text_blocks.store import (
    Action,
    ActionLog,
    Block,
    BlockStore,
    BlockVersion,
    Diff,
)


class TestBlock:
    def test_new_block_defaults(self):
        b = Block(id=1)
        assert b.id == 1
        assert b.title == ""
        assert b.versions == []
        assert b.lock is None
        assert b.status == "sealed"

    def test_current_when_empty_returns_none(self):
        b = Block(id=1)
        assert b.current is None
        assert b.content == ""

    def test_new_version_creates_and_appends(self):
        b = Block(id=1)
        v = b.new_version("g", "hello")
        assert v.version == 1
        assert v.source == "g"
        assert v.content == "hello"
        assert b.current is v
        assert b.content == "hello"
        assert b.version_count == 1

    def test_append_to_current_creates_version_when_empty(self):
        b = Block(id=1)
        b.append_to_current("first chunk")
        assert b.content == "first chunk"
        assert b.version_count == 1

    def test_append_to_current_appends_to_existing(self):
        b = Block(id=1)
        b.new_version("g", "hello")
        b.append_to_current(" world")
        assert b.content == "hello world"
        assert b.version_count == 1  # no new version

    def test_with_line_numbers(self):
        b = Block(id=1)
        b.new_version("g", "line one\nline two\nline three")
        result = b.with_line_numbers
        lines = result.split("\n")
        assert lines[0].startswith("     1")
        assert "line one" in lines[0]
        assert lines[1].startswith("     2")

    def test_replace_lines(self):
        b = Block(id=1)
        b.new_version("g", "line one\nline two\nline three")
        replaced = b.replace_lines(line_no=2, count=1, new_text="LINE TWO")
        assert replaced == 1
        assert b.content == "line one\nLINE TWO\nline three"

    def test_replace_lines_beyond_end_appends(self):
        b = Block(id=1)
        b.new_version("g", "only line")
        replaced = b.replace_lines(line_no=10, count=1, new_text="new")
        assert b.content == "only line\nnew"

    def test_source_from_current(self):
        b = Block(id=1)
        b.new_version("u", "human text")
        assert b.source == "u"
        b.new_version("g", "model text")
        assert b.source == "g"


class TestDiff:
    def test_compute_basic_diff(self):
        diff = Diff.compute(block_id=5, old_content="hello", new_content="world")
        assert diff.block_id == 5
        assert diff.unified_diff
        assert "-hello" in diff.unified_diff
        assert "+world" in diff.unified_diff

    def test_compute_anchor_from_first_change(self):
        diff = Diff.compute(
            block_id=1,
            old_content="line one\nline two\nline three",
            new_content="line one\nCHANGED\nline three",
        )
        assert diff.block_id == 1
        assert "line two" in diff.anchor_quote

    def test_compute_no_change(self):
        diff = Diff.compute(block_id=1, old_content="same", new_content="same")
        assert diff.unified_diff == ""


class TestActionLog:
    def test_record_and_recent(self):
        log = ActionLog(maxsize=10)
        log.record(Action(kind="block_create", block_id=1, summary="created #1"))
        log.record(Action(kind="block_edit", block_id=1, summary="edited #1"))
        recent = log.recent(2)
        assert len(recent) == 2
        assert recent[0].summary == "edited #1"  # most recent first

    def test_maxsize_eviction(self):
        log = ActionLog(maxsize=3)
        for i in range(5):
            log.record(Action(kind="block_create", block_id=i, summary=f"#{i}"))
        recent = log.recent(10)
        assert len(recent) == 3
        assert recent[2].summary == "#2"  # oldest kept


class TestBlockStore:
    def test_create_block(self):
        store = BlockStore()
        bid = store.create(source="g", title="test", lock="g")
        assert bid == 1
        block = store.get(bid)
        assert block is not None
        assert block.title == "test"
        assert block.lock == "g"
        assert block.status == "streaming"

    def test_create_block_with_content(self):
        store = BlockStore()
        bid = store.create(source="u", content="human text")
        block = store.get(bid)
        assert block.content == "human text"
        assert block.status == "sealed"

    def test_append_and_seal(self):
        store = BlockStore()
        bid = store.create(source="g", lock="g")
        store.append_to_current(bid, "chunk1")
        store.append_to_current(bid, "chunk2")
        assert store.get(bid).content == "chunk1chunk2"
        store.seal(bid)
        assert store.get(bid).status == "sealed"
        assert store.get(bid).lock is None

    def test_snapshot_returns_ordered_blocks(self):
        store = BlockStore()
        store.create(source="g")
        store.create(source="u")
        store.create(source="g")
        blocks = store.snapshot()
        assert len(blocks) == 3
        assert [b.id for b in blocks] == [1, 2, 3]

    def test_diff_bucket(self):
        store = BlockStore()
        diff = Diff.compute(block_id=1, old_content="a", new_content="b")
        store.push_diff(diff)
        assert len(store.peek_diffs()) == 1
        drained = store.drain_diffs()
        assert len(drained) == 1
        assert len(store.peek_diffs()) == 0

    def test_summary_empty(self):
        store = BlockStore()
        assert store.summary() == "no blocks"

    def test_summary_with_blocks(self):
        store = BlockStore()
        store.create(source="g", title="one")
        store.create(source="u", title="two")
        summary = store.summary()
        assert "2 blocks" in summary
        assert "#1..#2" in summary

    def test_summary_with_streaming(self):
        store = BlockStore()
        store.create(source="g", lock="g")
        summary = store.summary()
        assert "streaming" in summary

    def test_index(self):
        store = BlockStore()
        store.create(source="g", title="first")
        store.create(source="u", title="second")
        idx = store.index()
        assert "#1" in idx
        assert "#2" in idx
        assert "first" in idx
        assert "second" in idx

    def test_dump_single_file(self, tmp_path):
        store = BlockStore()
        store.create(source="g", title="test", content="hello world")
        path = str(tmp_path / "output.md")
        result = store.dump(path=path)
        assert result.count == 1
        with open(path) as f:
            content = f.read()
        assert "test" in content
        assert "hello world" in content

    def test_dump_directory(self, tmp_path):
        store = BlockStore()
        store.create(source="g", title="alpha", content="aaa")
        store.create(source="u", title="beta", content="bbb")
        result = store.dump(path=str(tmp_path))
        assert result.count == 2
        files = sorted(tmp_path.glob("*.md"))
        assert len(files) == 2
        assert "alpha" in files[0].name or "aaa" in files[0].read_text()

    def test_dump_default_dir(self):
        store = BlockStore(session_uid="test123")
        result = store.dump()
        assert result.count == 0  # no blocks
        assert "test123" in result.path
        import os
        assert os.path.isdir(result.path)

    def test_action_log_integration(self):
        store = BlockStore()
        store.create(source="g", title="test")
        recent = store.action_log.recent(1)
        assert len(recent) == 1
        assert recent[0].kind == "block_create"

    def test_lock_acquire_release(self):
        store = BlockStore()
        bid = store.create(source="g")
        store.acquire_lock(bid, "g")
        assert store.get(bid).lock == "g"
        assert store.get(bid).status == "streaming"
        store.release_lock(bid)
        assert store.get(bid).lock is None

    def test_get_order(self):
        store = BlockStore()
        store.create(source="g")
        store.create(source="u")
        store.create(source="g")
        assert store.get_order() == [1, 2, 3]
