import asyncio
import time

import pytest

from ghoshell_terminal.card import CardState, CardType
from ghoshell_terminal.channel import build_terminal_channel
from ghoshell_terminal.store import CardStore, Mode

from fakes import FakeSubprocesses, Recorder, chunks


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return CardStore(root=root, outputs_dir=tmp_path / "out")


def _channel(store, processes, rec=None, enabled=None, groundset=None):
    rec = rec or Recorder()
    chan = build_terminal_channel(
        store, processes, surface=rec, signaler=rec, enabled=enabled,
        groundset=groundset,
    )
    return chan, rec


async def _until(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_channel_exposes_the_command_set(store):
    chan, _ = _channel(store, FakeSubprocesses())
    async with chan.bootstrap() as runtime:
        for name in (
            "open", "threads", "exec", "rule", "read", "cards",
            "stop", "stop_all", "cancel",
        ):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_disabled_mode_takes_the_commands_off_the_interface(store):
    gate = [True]
    chan, _ = _channel(store, FakeSubprocesses(), enabled=lambda: gate[0])
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.get_command("exec").meta().available is True

        gate[0] = False
        await runtime.refresh_metas()
        assert runtime.get_command("exec").meta().available is False


@pytest.mark.asyncio
async def test_open_registers_a_thread_and_refuses_an_escape(store, tmp_path):
    chan, _ = _channel(store, FakeSubprocesses())
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command(
            "open", args=("dev", "."), kwargs={"description": "build"}
        )
        assert "dev" in out
        listed = await runtime.execute_command("threads")
        assert "dev" in listed

        with pytest.raises(Exception):
            await runtime.execute_command("open", args=("bad", "../outside"))


@pytest.mark.asyncio
async def test_exec_returns_a_receipt_before_the_human_decides(store):
    processes = FakeSubprocesses(lines=["done\n"])
    chan, rec = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        receipt = await runtime.execute_command(
            "exec", args=("dev", chunks(["ls -la\n"])), kwargs={"desc": "list"}
        )

        assert "awaiting approval" in receipt, "the model must not block on the human"
        assert processes.spawned == [], "nothing runs before a verdict"
        card = store.cards()[-1]
        assert card.state is CardState.AWAITING
        assert card.content == "ls -la\n"
        assert rec.types()[:2] == ["card.head", "card.tail"]


@pytest.mark.asyncio
async def test_accept_runs_it_and_signals_the_outcome(store):
    processes = FakeSubprocesses(lines=["hello\n"], exit_code=0)
    chan, rec = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        await runtime.execute_command("exec", args=("dev", chunks(["echo hi"])))
        card = store.cards()[-1]

        store.settle(card.id, "accept")
        await _until(lambda: card.state is CardState.DONE)

        assert len(processes.spawned) == 1
        assert processes.spawned[0].meta.command == "echo hi"
        assert [f["type"] for f in rec.of("card.output")] == ["card.output"]
        assert rec.of("card.output")[0]["lines"] == ["hello\n"]
        assert len(rec.signals) == 2, "batch notify + completion signal"
        assert "decided" in str(rec.signals[0].messages[0].to_content_string())
        assert "done" in str(rec.signals[1].messages[0].to_content_string())


@pytest.mark.asyncio
async def test_deny_is_terminal_and_runs_nothing(store):
    processes = FakeSubprocesses()
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        await runtime.execute_command("exec", args=("dev", chunks(["rm -rf /"])))
        card = store.cards()[-1]

        store.settle(card.id, "deny")
        await _until(lambda: card.state is CardState.REJECTED)
        assert processes.spawned == []


@pytest.mark.asyncio
async def test_the_real_gate_is_the_mode(store):
    """main.py wires the gate to the store's mode; that wiring is what makes the
    web surface's 'disabled' switch reach the model's interface."""
    chan, _ = _channel(
        store, FakeSubprocesses(), enabled=lambda: store.mode != Mode.DISABLED
    )
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.get_command("exec").meta().available is True

        store.set_mode(Mode.DISABLED)
        await runtime.refresh_metas()
        assert runtime.get_command("exec").meta().available is False


@pytest.mark.asyncio
async def test_an_accepted_rule_auto_approves_a_matching_command(store):
    processes = FakeSubprocesses(lines=["ok\n"])
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        rule_card = store.new_card(CardType.RULE, title="safe reads")
        store.append_content(rule_card.id, r"^ls\b")
        store.activate_rule(rule_card.id)

        receipt = await runtime.execute_command("exec", args=("dev", chunks(["ls -la"])))
        assert "auto-approved" in receipt
        card = store.cards()[-1]
        await _until(lambda: card.state is CardState.DONE)
        assert len(processes.spawned) == 1


@pytest.mark.asyncio
async def test_no_rule_and_not_auto_still_asks(store):
    processes = FakeSubprocesses()
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        receipt = await runtime.execute_command("exec", args=("dev", chunks(["rm -rf /"])))
        assert "awaiting approval" in receipt
        assert processes.spawned == []


@pytest.mark.asyncio
async def test_an_auto_thread_runs_without_asking(store):
    processes = FakeSubprocesses(lines=["ok\n"])
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        store.set_thread_auto("dev", True)
        receipt = await runtime.execute_command("exec", args=("dev", chunks(["ls -la"])))
        assert "auto-approved" in receipt
        card = store.cards()[-1]
        await _until(lambda: card.state is CardState.DONE)
        assert len(processes.spawned) == 1


@pytest.mark.asyncio
async def test_exec_defaults_to_the_root_thread(store):
    processes = FakeSubprocesses()
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command(
            "exec", kwargs={"chunks__": chunks(["ls\n"])}
        )
        assert "awaiting approval" in receipt
        card = store.cards()[-1]
        assert card.thread == "root"
        assert card.cwd == str(store.root)


@pytest.mark.asyncio
async def test_a_rule_proposal_decides_nothing_until_accepted(store):
    chan, _ = _channel(store, FakeSubprocesses())
    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command(
            "rule", args=("reads", r"^cat\b"), kwargs={"description": "read files"}
        )
        assert "awaiting approval" in receipt
        card = store.cards()[-1]
        assert store.match_rule("cat x") is None

        store.settle(card.id, "accept")
        await _until(lambda: store.match_rule("cat x") is not None)
        assert card.state is CardState.DONE


@pytest.mark.asyncio
async def test_a_broken_regex_never_becomes_a_card(store):
    chan, _ = _channel(store, FakeSubprocesses())
    async with chan.bootstrap() as runtime:
        with pytest.raises(Exception):
            await runtime.execute_command("rule", args=("bad", "("))
        assert store.cards() == []


@pytest.mark.asyncio
async def test_read_hands_back_the_file_when_output_is_long(store):
    processes = FakeSubprocesses(lines=["x" * 200 + "\n"] * 60)
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        await runtime.execute_command("exec", args=("dev", chunks(["noisy"])))
        card = store.cards()[-1]
        store.settle(card.id, "accept")
        await _until(lambda: card.state is CardState.DONE)

        report = await runtime.execute_command("read", args=(card.id,))
        assert card.output_file in report
        assert str(card.output_chars) in report


@pytest.mark.asyncio
async def test_read_hands_back_the_text_when_output_is_short(store):
    processes = FakeSubprocesses(lines=["small\n"])
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        await runtime.execute_command("exec", args=("dev", chunks(["echo small"])))
        card = store.cards()[-1]
        store.settle(card.id, "accept")
        await _until(lambda: card.state is CardState.DONE)
        assert await runtime.execute_command("read", args=(card.id,)) is not None
        assert "small" in await runtime.execute_command("read", args=(card.id,))
        assert card.output_file is None, "short output leaves no file behind"
        assert not store.output_path(card.id).exists()


@pytest.mark.asyncio
async def test_cancel_withdraws_a_pending_card(store):
    processes = FakeSubprocesses()
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        await runtime.execute_command("exec", args=("dev", chunks(["sleep 100"])))
        card = store.cards()[-1]

        out = await runtime.execute_command("cancel", args=(card.id,))
        assert "withdrawn" in out
        assert card.state is CardState.CANCELLED
        assert processes.spawned == []


@pytest.mark.asyncio
async def test_stop_ends_a_running_process(store):
    processes = FakeSubprocesses(hold=True)
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        await runtime.execute_command("exec", args=("dev", chunks(["sleep 100"])))
        card = store.cards()[-1]
        store.settle(card.id, "accept")
        await _until(lambda: card.state is CardState.RUNNING)

        await runtime.execute_command("stop", args=(card.id,))
        assert processes.spawned[0].stopped is True
        await _until(lambda: card.state is CardState.ERROR)


@pytest.mark.asyncio
async def test_notice_carries_counts_and_thread_flags(store):
    chan, _ = _channel(store, FakeSubprocesses())
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."), kwargs={"description": "d"})
        await runtime.execute_command("exec", args=("dev", chunks(["ls"])))
        store.set_thread_auto("dev", True)
        await runtime.refresh_metas()

        notices = runtime.self_meta().named_notices
        assert "awaiting: 1" in notices["terminal"]
        assert "auto" in notices["thread_dev"]
        assert "thread_root" in notices, "the default root thread is always listed"


@pytest.mark.asyncio
async def test_a_batch_notify_fires_once_when_the_last_card_settles(store):
    processes = FakeSubprocesses()
    chan, rec = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("dev", "."))
        await runtime.execute_command("exec", args=("dev", chunks(["a"])))
        await runtime.execute_command("exec", args=("dev", chunks(["b"])))
        a, b = store.cards()

        store.settle(a.id, "deny")
        await _until(lambda: a.state is CardState.REJECTED)
        assert rec.signals == [], "one verdict left — no batch knock yet"

        store.settle(b.id, "deny")
        await _until(lambda: b.state is CardState.REJECTED)
        assert len(rec.signals) == 1, "one batch knock for the whole round"
        assert "decided" in rec.signals[0].messages[0].to_content_string()


@pytest.mark.asyncio
async def test_ground_renders_the_thread_cognitive_field(tmp_path):
    from ghoshell_moss.ground import DefaultGroundSet

    root = tmp_path / "root"
    root.mkdir()
    (root / "GROUND.md").write_text("---\nname: project\n---\n\nwelcome field\n")

    store = CardStore(root=root, outputs_dir=tmp_path / "out")
    groundset = DefaultGroundSet(workspace_root=root, materialize=False)

    chan, _ = _channel(store, FakeSubprocesses(), groundset=groundset)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command("ground")
        assert "welcome field" in out


@pytest.mark.asyncio
async def test_ground_reports_when_there_is_no_field(tmp_path):
    from ghoshell_moss.ground import DefaultGroundSet

    root = tmp_path / "root"
    root.mkdir()
    store = CardStore(root=root, outputs_dir=tmp_path / "out")
    groundset = DefaultGroundSet(workspace_root=root, materialize=False)

    chan, _ = _channel(store, FakeSubprocesses(), groundset=groundset)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command("ground")
        assert "no ground" in out
