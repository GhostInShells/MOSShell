"""Tests for runtime_error channel — pull-only self-diagnosis over RuntimeErrorLog.

Covers:
- pull returns the latest n rendered records, newest last
- pull(critical=True) reads only the never-dropped CRITICAL buffer
- empty collector reports "no errors" rather than a blank result
- a missing RuntimeErrorLog degrades to a notice, never raises
"""

import pytest

from ghoshell_moss.channels.runtime_error_channel import new_runtime_error_channel
from ghoshell_moss.contracts.runtime_error import RuntimeErrorRecord


class _FakeLog:
    def __init__(self, records=(), critical=()):
        self._records = list(records)
        self._critical = list(critical)

    def pull(self, n):
        return self._records[-n:]

    def pull_critical(self):
        return list(self._critical)


def _record(level, message, location="x.py:1"):
    return RuntimeErrorRecord(levelname=level, message=message, created=0.0, location=location)


@pytest.mark.asyncio
async def test_pull_returns_latest_records_newest_last():
    log = _FakeLog(records=[
        _record("ERROR", "first"),
        _record("ERROR", "second"),
        _record("ERROR", "third"),
    ])
    chan = new_runtime_error_channel(log=log)

    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("pull", args=(False, 2))

    lines = result.splitlines()
    assert len(lines) == 2
    assert "second" in lines[0]
    assert "third" in lines[1]
    assert "first" not in result


@pytest.mark.asyncio
async def test_pull_critical_reads_only_critical_buffer():
    log = _FakeLog(
        records=[_record("ERROR", "noise")],
        critical=[_record("CRITICAL", "fatal")],
    )
    chan = new_runtime_error_channel(log=log)

    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("pull", args=(True,))

    assert "fatal" in result
    assert "noise" not in result


@pytest.mark.asyncio
async def test_pull_reports_no_errors_when_empty():
    chan = new_runtime_error_channel(log=_FakeLog())

    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("pull")

    assert "no errors" in result


@pytest.mark.asyncio
async def test_pull_degrades_when_no_runtime_error_log():
    # no injected log and no RuntimeErrorLog in the container → a notice, not a crash.
    chan = new_runtime_error_channel()

    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("pull")

    assert "not available" in result
