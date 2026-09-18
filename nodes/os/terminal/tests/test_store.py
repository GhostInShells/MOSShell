import asyncio

import pytest

from ghoshell_terminal.card import CardState, CardType
from ghoshell_terminal.store import CardStore, Mode


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return CardStore(root=root, outputs_dir=tmp_path / "out")


def test_thread_cwd_must_stay_inside_root(store, tmp_path):
    store.open_thread("dev", ".", "build and test")
    assert store.get_thread("dev").cwd == str(store.root)

    with pytest.raises(ValueError, match="escapes the root"):
        store.open_thread("bad", "../elsewhere")
    with pytest.raises(ValueError, match="escapes the root"):
        store.open_thread("bad", str(tmp_path / "sibling"))


def test_thread_names_are_unique(store):
    store.open_thread("dev", ".")
    with pytest.raises(ValueError, match="already exists"):
        store.open_thread("dev", ".")


def test_relative_cwd_resolves_against_root(store):
    (store.root / "sub").mkdir()
    thread = store.open_thread("dev", "sub")
    assert thread.cwd == str((store.root / "sub").resolve())


def test_settle_only_lands_on_a_pending_card(store):
    card = store.new_card(CardType.COMMAND, title="dev")
    store.set_state(card.id, CardState.AWAITING)
    assert store.settle(card.id, "accept") is True

    store.set_state(card.id, CardState.RUNNING)
    assert store.settle(card.id, "accept") is False, "a decided card takes no second verdict"

    store.set_state(card.id, CardState.REJECTED)
    assert store.settle(card.id, "deny") is False


@pytest.mark.asyncio
async def test_settle_wakes_the_waiter(store):
    card = store.new_card(CardType.COMMAND)
    store.set_state(card.id, CardState.AWAITING)
    waiter = store.waiter(card.id)

    store.settle(card.id, "accept")
    assert await asyncio.wait_for(waiter, timeout=1) == "accept"


def test_rejected_and_cancelled_are_distinct_terminal_states(store):
    rejected = store.new_card(CardType.COMMAND)
    store.set_state(rejected.id, CardState.REJECTED)
    cancelled = store.new_card(CardType.COMMAND)
    store.set_state(cancelled.id, CardState.CANCELLED)

    assert rejected.settled and cancelled.settled
    assert rejected.state is not cancelled.state


def test_output_keeps_a_bounded_tail_and_a_complete_file(store):
    card = store.new_card(CardType.COMMAND)
    for i in range(500):
        store.append_output(card.id, [f"line {i}\n"])

    assert len(card.output_tail) <= 400
    assert card.output_tail[-1] == "line 499\n"
    assert card.output_chars == sum(len(f"line {i}\n") for i in range(500))

    written = store.output_text(card.id)
    assert written.endswith("line 499\n")
    from pathlib import Path

    assert Path(card.output_file).read_text().endswith("line 499\n")


def test_rules_only_go_live_once_accepted(store):
    card = store.new_card(CardType.RULE, title="safe reads")
    store.append_content(card.id, r"ls -la")

    assert store.match_rule("ls -la") is None, "proposed rules decide nothing"

    store.activate_rule(card.id)
    assert store.match_rule("ls -la") is card
    assert store.match_rule("rm -rf /") is None
    assert store.match_rule("cd x && ls -la") is card, "matched with re.search, not fullmatch"


def test_a_broken_rule_is_rejected_at_activation(store):
    card = store.new_card(CardType.RULE)
    store.append_content(card.id, "(")
    with pytest.raises(Exception):
        store.activate_rule(card.id)


def test_mode_rejects_unknown_values(store):
    assert store.mode == Mode.APPROVAL
    assert store.set_mode(Mode.AUTO) == Mode.AUTO
    with pytest.raises(ValueError, match="unknown mode"):
        store.set_mode("yolo")


def test_counts_track_pending_and_running(store):
    a = store.new_card(CardType.COMMAND, thread="dev")
    b = store.new_card(CardType.COMMAND, thread="dev")
    c = store.new_card(CardType.COMMAND, thread="ops")
    store.set_state(a.id, CardState.AWAITING)
    store.set_state(b.id, CardState.RUNNING)
    store.set_state(c.id, CardState.DONE)

    assert [x.id for x in store.awaiting()] == [a.id]
    assert [x.id for x in store.running()] == [b.id]
