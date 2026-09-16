import pytest

from ghoshell_file_editor.structure import (
    BASE,
    Action,
    Seq,
    Thread,
    cascade_seqs,
    changes_content,
    compute_effect,
    diff_of,
    effect_of,
    rewind_target,
    tail_content,
)


def _thread(actions=(), base="base\n"):
    """Build a thread whose actions carry exactly the effects append would give."""
    t = Thread(id="t", label="doc", base_content=base)
    for n, (kind, payload, verdict) in enumerate(actions, start=1):
        t.actions[n] = Action(
            seq=Seq("t", n), author="g", kind=kind, description=kind,
            payload=payload, effect=compute_effect(t, kind, payload),
            verdict=verdict,
        )
    return t


def test_diff_of_produces_unified_diff():
    d = diff_of("a\nb\n", "a\nc\n")
    assert "-b" in d
    assert "+c" in d


def test_diff_of_no_change_is_empty():
    assert diff_of("same\n", "same\n") == ""


def test_effect_of_carries_before_after_diff():
    e = effect_of("x", "y")
    assert e.before == "x"
    assert e.after == "y"
    assert e.diff


def test_reference_carries_no_effect():
    t = _thread([("reference", "1,10", "pending")], base="one\n")
    assert t.actions[1].effect is None


def test_export_carries_the_content_it_would_write():
    t = _thread([("write", "two\n", "pending")], base="one\n")
    t.actions[2] = Action(
        seq=Seq("t", 2), author="g", kind="export", description="export",
        payload="/tmp/o.md", effect=compute_effect(t, "export", "/tmp/o.md"),
    )

    assert t.actions[2].effect.before == "two\n"
    assert t.actions[2].effect.after == "two\n"
    assert t.actions[2].effect.diff == ""
    assert not changes_content(t.actions[2])


def test_before_is_the_pending_tail_not_the_confirmed_head():
    # Action 1 is still pending; action 2 diffs against it, not against the
    # untouched baseline.
    t = _thread(
        [("write", "one\n", "pending"), ("write", "two\n", "pending")], base="zero\n",
    )

    assert t.actions[1].effect.before == "zero\n"
    assert t.actions[2].effect.before == "one\n"
    assert t.actions[2].effect.after == "two\n"


def test_tail_content_skips_rejected_actions():
    t = _thread(
        [("write", "one\n", "confirmed"), ("write", "two\n", "rejected")],
        base="zero\n",
    )
    assert tail_content(t) == "one\n"


def test_tail_content_skips_actions_without_effect():
    t = _thread(
        [("write", "one\n", "pending"), ("reference", "1,3", "pending")],
        base="zero\n",
    )
    assert tail_content(t) == "one\n"


def test_tail_content_falls_back_to_baseline():
    assert tail_content(_thread(base="zero\n")) == "zero\n"


def test_rewind_resolves_the_target_content():
    t = _thread(
        [("write", "v1\n", "confirmed"), ("write", "v2\n", "confirmed")],
        base="v0\n",
    )
    effect = compute_effect(t, "rewind", "1")

    assert effect.before == "v2\n"
    assert effect.after == "v1\n"


def test_rewind_to_base():
    t = _thread([("write", "v1\n", "confirmed")], base="v0\n")
    effect = compute_effect(t, "rewind", BASE)

    assert effect.before == "v1\n"
    assert effect.after == "v0\n"


def test_rewind_may_target_a_still_pending_action():
    # Going back to an earlier proposal inside the same burst: the target need
    # not have been confirmed to be addressable.
    t = _thread(
        [("write", "v1\n", "pending"), ("write", "v2\n", "pending")], base="v0\n",
    )
    effect = compute_effect(t, "rewind", "1")

    assert effect.before == "v2\n"
    assert effect.after == "v1\n"


def test_rewind_target_reads_base():
    assert rewind_target(_thread(), BASE) is None


def test_rewind_target_rejects_unknown_seq():
    with pytest.raises(ValueError):
        rewind_target(_thread([("write", "v1\n", "pending")], base="v0\n"), "9")


def test_rewind_target_rejects_a_rejected_action():
    with pytest.raises(ValueError):
        rewind_target(_thread([("write", "v1\n", "rejected")], base="v0\n"), "1")


def test_rewind_target_rejects_an_effectless_action():
    with pytest.raises(ValueError):
        rewind_target(_thread([("reference", "1,3", "pending")], base="v0\n"), "1")


def test_head_is_the_last_confirmed_action_that_changed_content():
    t = _thread(
        [
            ("write", "v1\n", "confirmed"),
            ("reference", "1,3", "confirmed"),
            ("write", "v2\n", "confirmed"),
        ],
        base="v0\n",
    )
    assert t.head.seq.n == 3
    assert t.content == "v2\n"


def test_head_ignores_rejected_and_pending_actions():
    t = _thread(
        [
            ("write", "v1\n", "confirmed"),
            ("write", "v2\n", "rejected"),
            ("write", "v3\n", "pending"),
        ],
        base="v0\n",
    )
    assert t.head.seq.n == 1
    assert t.content == "v1\n"


def test_head_is_none_before_anything_is_confirmed():
    t = _thread([("write", "v1\n", "pending")], base="v0\n")
    assert t.head is None
    assert t.content == "v0\n"


def test_versions_view_holds_confirmed_content_changes_only():
    t = _thread(
        [
            ("write", "v1\n", "confirmed"),
            ("reference", "1,3", "confirmed"),
            ("write", "v2\n", "pending"),
        ],
        base="v0\n",
    )
    assert [a.seq.n for a in t.versions] == [1]


def test_changes_content_is_false_for_a_no_op_write():
    t = _thread([("write", "v1\n", "confirmed")], base="v1\n")
    assert not changes_content(t.actions[1])
    assert t.head is None
    assert t.content == "v1\n"


def test_cascade_seqs_from_n():
    t = _thread([
        ("write", "a\n", "pending"),
        ("write", "b\n", "pending"),
        ("write", "c\n", "pending"),
    ])
    assert [s.n for s in cascade_seqs(t, 2)] == [2, 3]
