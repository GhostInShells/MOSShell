import pytest

from ghoshell_file_editor.structure import (
    Action,
    Thread,
    content_at,
    effect_of,
    render_source,
    replace_once,
    side_effect,
    slice_region,
)


def _thread(actions=(), base="base\n"):
    thread = Thread(id="t", label="doc", base=base)
    for n, kind, text in actions:
        thread.actions.append(
            Action(n=n, kind=kind, author="g", label=kind, text=text)
        )
    return thread


def test_effect_of_carries_before_after_diff():
    e = effect_of("a\n", "b\n")
    assert e.before == "a\n"
    assert e.after == "b\n"
    assert e.diff.startswith("--- before")


def test_replace_once_swaps_the_single_occurrence():
    assert replace_once("hello world", "world", "there") == "hello there"


def test_replace_once_refuses_missing():
    with pytest.raises(ValueError, match="does not appear"):
        replace_once("hello", "zzz", "x")


def test_replace_once_refuses_ambiguous():
    with pytest.raises(ValueError, match="appears 2 times"):
        replace_once("a b a", "a", "x")


def test_replace_once_refuses_empty_old():
    with pytest.raises(ValueError, match="non-empty"):
        replace_once("x", "", "y")


def test_slice_region_whole_text():
    assert slice_region("a\nb\nc\n", "") == "a\nb\nc\n"


def test_slice_region_range_is_inclusive():
    assert slice_region("a\nb\nc\n", "2-3") == "b\nc\n"


def test_slice_region_single_line():
    assert slice_region("a\nb\n", "2") == "b\n"


def test_slice_region_rejects_malformed():
    with pytest.raises(ValueError):
        slice_region("a\n", "x")


def test_slice_region_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        slice_region("a\n", "2-5")


def test_content_at_is_the_last_effect_at_or_before_n():
    thread = _thread()
    a1 = Action(n=1, kind="write", author="g", label="w", effect=effect_of("base\n", "one\n"))
    a2 = Action(n=2, kind="write", author="g", label="w", effect=effect_of("one\n", "two\n"))
    thread.actions = [a1, a2]
    assert content_at(thread, 1) == "one\n"
    assert content_at(thread, 2) == "two\n"


def test_render_source_varies_by_kind():
    write = Action(n=1, kind="write", author="g", label="w",
                   effect=effect_of("BASE\n", "NEXT\n"))
    read = Action(n=2, kind="read", author="g", label="r", text="NEXT\n")
    append = Action(n=3, kind="append", author="g", label="a", text="+tail\n")
    assert render_source(write) == "NEXT\n"
    assert render_source(read) == "NEXT\n"
    assert render_source(append) == "+tail\n"


def test_side_effect_is_mechanical_and_names_the_export_path():
    read = Action(n=1, kind="read", author="g", label="r")
    assert side_effect(read) == "none — read only"
    export = Action(n=2, kind="export", author="g", label="e", payload="/tmp/x")
    assert side_effect(export) == "disk — writes /tmp/x"
