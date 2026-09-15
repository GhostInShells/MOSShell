from ghoshell_file_editor.structure import (
    Action,
    Seq,
    Thread,
    cascade_seqs,
    diff_of,
    effect_of,
    is_mutating,
    result_content,
)


def test_diff_of_produces_unified_diff():
    d = diff_of("a\nb\n", "a\nc\n")
    assert "-b" in d
    assert "+c" in d


def test_diff_of_no_change_is_empty():
    assert diff_of("same\n", "same\n") == ""


def test_is_mutating():
    assert is_mutating("write")
    assert is_mutating("str_replace")
    assert is_mutating("insert")
    assert is_mutating("rewind")
    assert not is_mutating("reference")
    assert not is_mutating("export")


def test_effect_of_carries_before_after_diff():
    e = effect_of("x", "y")
    assert e.before == "x"
    assert e.after == "y"
    assert e.diff


def test_result_content_mutation_is_payload():
    a = Action(seq=Seq("t", 1), author="g", kind="write", description="", payload="new")
    assert result_content(a, "old", lambda vid: "") == "new"


def test_result_content_rewind_resolves_target_version():
    a = Action(seq=Seq("t", 1), author="g", kind="rewind", description="", payload="t:v0")
    assert result_content(a, "now", lambda vid: "then") == "then"


def test_result_content_reference_is_unchanged():
    a = Action(seq=Seq("t", 1), author="g", kind="reference", description="", payload="1,10")
    assert result_content(a, "base", lambda vid: "") == "base"


def test_cascade_seqs_from_n():
    t = Thread(id="t", label="l")
    t.order = [Seq("t", 1), Seq("t", 2), Seq("t", 3)]
    out = cascade_seqs(t, Seq("t", 2))
    assert [s.n for s in out] == [2, 3]
