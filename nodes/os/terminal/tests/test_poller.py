"""The poller exists because the subprocess layer has no incremental output API.
These tests pin the reconstruction: the window slides, drops old lines, and never
grows a cursor we can trust."""

import asyncio

import pytest

from ghoshell_terminal.poller import _new_lines, stream_output

from fakes import FakeManaged


def test_overlap_finds_the_boundary_when_the_window_slides():
    known = ["a\n", "b\n", "c\n", "d\n"]
    assert _new_lines(known, ["a\n", "b\n", "c\n", "d\n"]) == []
    assert _new_lines(known, ["b\n", "c\n", "d\n", "e\n"]) == ["e\n"]
    assert _new_lines(known, ["c\n", "d\n"]) == [], "a shrinking window adds nothing"


def test_no_overlap_yields_the_whole_window():
    assert _new_lines(["a\n"], ["x\n", "y\n"]) == ["x\n", "y\n"]


@pytest.mark.asyncio
async def test_stream_output_emits_only_new_lines():
    managed = FakeManaged(index=1, command="ls", name="ls", cwd="/tmp")
    managed.output.push("one\ntwo\n")
    managed.finish()

    seen: list[str] = []

    async def on_lines(lines):
        seen.extend(lines)

    await stream_output(managed, on_lines, interval=0.01)
    assert seen == ["one\n", "two\n"]


@pytest.mark.asyncio
async def test_stream_output_picks_up_lines_written_while_running():
    managed = FakeManaged(index=1, command="tail", name="tail", cwd="/tmp")
    seen: list[str] = []

    async def on_lines(lines):
        seen.extend(lines)

    async def writer():
        await asyncio.sleep(0.03)
        managed.output.push("first\n")
        await asyncio.sleep(0.05)
        managed.output.push("second\nthird\n")
        await asyncio.sleep(0.03)
        managed.finish()

    await asyncio.gather(stream_output(managed, on_lines, interval=0.02), writer())
    assert seen == ["first\n", "second\n", "third\n"]


@pytest.mark.asyncio
async def test_stream_output_windows_never_duplicate_a_line():
    """The window forgets; the diff must not re-emit what it already showed."""
    managed = FakeManaged(index=1, command="spam", name="spam", cwd="/tmp")
    for i in range(5):
        managed.output.push(f"line {i}\n")
    managed.finish()

    seen: list[str] = []

    async def on_lines(lines):
        seen.extend(lines)

    await stream_output(managed, on_lines, interval=0.01)
    assert seen == [f"line {i}\n" for i in range(5)]
