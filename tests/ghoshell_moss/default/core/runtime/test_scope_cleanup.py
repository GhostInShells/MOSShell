"""
Test: _uncommitted_scopes lifecycle — poisoned scope chain pollution.

When _set_interpreter_error cancels a __scope_enter__ task whose scope has
been opened but not yet committed, the scope's future is cancelled but the
scope remains in _uncommitted_scopes, poisoning all subsequent scopes.
"""
import asyncio

import pytest

from ghoshell_moss.core.concepts.command import BaseCommandTask, CommandMeta
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core.ctml import new_ctml_shell


@pytest.mark.asyncio
async def test_append_interpreter_baseline():
    """Verify kind='append' works for chaining interpreter sessions."""
    channel = PyChannel(name="test")

    got = []

    @channel.build.command()
    async def foo() -> int:
        got.append(1)
        return 123

    shell = new_ctml_shell("test_shell")
    shell.main_channel.import_channels(channel)

    async with shell:
        # Round 1: normal command
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<test:foo />")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            interpreter.raise_exception()
        assert len(got) == 1

        # Round 2: append another command
        async with await shell.interpreter(kind="append") as interpreter:
            interpreter.feed("<test:foo />")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            interpreter.raise_exception()
        assert len(got) == 2


@pytest.mark.asyncio
async def test_append_interpreter_with_scope_no_error():
    """Verify kind='append' with a scope works without prior error."""
    channel = PyChannel(name="test")

    got = []

    @channel.build.command()
    async def foo() -> int:
        got.append(1)
        return 123

    shell = new_ctml_shell("test_shell")
    shell.main_channel.import_channels(channel)

    async with shell:
        # Round 1: scope with command
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<_>hello<test:foo /></_>")
            interpreter.commit()
            await interpreter.wait_tasks()
            interpreter.raise_exception()
        assert len(got) == 1

        # Round 2: append scope with command
        async with await shell.interpreter(kind="append") as interpreter:
            interpreter.feed("<_>hello<test:foo /></_>")
            interpreter.commit()
            await interpreter.wait_tasks()
            interpreter.raise_exception()
        assert len(got) == 2


@pytest.mark.asyncio
async def test_poisoned_scope_cancels_subsequent_scope():
    """
    Verify the fix: after a parse error poisons a scope, subsequent
    append-interpreter scopes should still work correctly.

    Round 1: parse error inside <_> leaves a poisoned (closed but
    uncommitted) scope in the runtime.

    Round 2 (kind='append'): a new <_> scope is opened. get_active_scope
    cleans up the poisoned scope before returning, so the new scope is
    created without being bound to a dead parent.

    Result: the Round 2 foo command executes successfully.
    """
    channel = PyChannel(name="test")

    got = []

    @channel.build.command()
    async def foo() -> int:
        got.append(1)
        return 123

    shell = new_ctml_shell("test_shell")
    shell.main_channel.import_channels(channel)

    async with shell:
        # Round 1: scope with text triggers enter task delivery via
        # on_delta_token. Then <test:nonexistent /> triggers parse error
        # which cancels the enter task via _set_interpreter_error.
        # The scope is closed but stays in _uncommitted_scopes.
        try:
            async with await shell.interpreter() as interpreter:
                interpreter.feed("<_><test:nonexistent /></_>")
                interpreter.commit()
                await interpreter.wait_tasks()
                interpreter.raise_exception()
        except InterpretError:
            pass

        # Round 2 (append): open a scope with a valid command inside.
        # After fix: get_active_scope cleans up the poisoned scope from
        # Round 1 before returning, so the new scope is NOT bound to a
        # dead parent and foo executes successfully.
        # Use until='all' so scope waits for cross-channel tasks (test:foo).
        async with await shell.interpreter(kind="append") as interpreter:
            interpreter.feed("<_ until='all'>hello<test:foo /></_>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            interpreter.raise_exception()

        # Verify no task was cancelled by poisoned scope
        cancelled = [t for t in tasks.values() if t.cancelled()]
        assert len(cancelled) == 0, (
            f"BUG: {len(cancelled)} tasks cancelled by poisoned scope: "
            f"{[t.caller_name() for t in cancelled]}"
        )


@pytest.mark.asyncio
async def test_healthy_scope_baseline():
    """
    Baseline: without a parse error, scope lifecycle should work normally.
    """
    channel = PyChannel(name="test")

    got = []

    @channel.build.command()
    async def foo() -> int:
        got.append(1)
        return 123

    shell = new_ctml_shell("test_shell")
    shell.main_channel.import_channels(channel)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<_>hello<test:foo /></_>")
            interpreter.commit()
            await interpreter.wait_tasks()
            interpreter.raise_exception()

        assert len(got) == 1
