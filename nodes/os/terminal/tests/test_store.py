import asyncio
import json
from datetime import datetime

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


def test_a_root_thread_always_exists(store):
    root = store.get_thread("root")
    assert root is not None
    assert root.cwd == str(store.root)
    assert root.auto is False


def test_thread_auto_flag(store):
    store.set_thread_auto("root", True)
    assert store.get_thread("root").auto is True
    with pytest.raises(KeyError):
        store.set_thread_auto("nope", True)


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


def test_output_keeps_a_bounded_tail_and_never_writes_a_file(store):
    card = store.new_card(CardType.COMMAND)
    for i in range(500):
        store.append_output(card.id, [f"line {i}\n"])

    assert len(card.output_tail) <= 400
    assert card.output_tail[-1] == "line 499\n"
    assert card.output_chars == sum(len(f"line {i}\n") for i in range(500))
    assert store.output_text(card.id).endswith("line 499\n")

    # The store is not responsible for the file — the channel decides keep/delete.
    assert card.output_file is None
    assert store.output_path(card.id).name == f"card_{card.id}.log"
    assert not store.output_path(card.id).exists()


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
    assert store.set_mode(Mode.DISABLED) == Mode.DISABLED
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


def test_settled_cards_are_written_to_the_audit_log(tmp_path):
    log_dir = tmp_path / "cards"
    store = CardStore(
        root=tmp_path / "root",
        outputs_dir=tmp_path / "out",
        log_dir=log_dir,
    )
    card = store.new_card(CardType.COMMAND, title="dev", thread="dev")
    store.append_content(card.id, "ls -la")
    store.set_state(card.id, CardState.DONE, exit_code=0)

    day = datetime.now().strftime("%Y-%m-%d")
    path = log_dir / f"{day}.jsonl"
    assert path.exists()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == card.id
    assert records[0]["state"] == "done"
    assert records[0]["content"] == "ls -la"


def test_unsettled_cards_are_not_audited(tmp_path):
    log_dir = tmp_path / "cards"
    store = CardStore(
        root=tmp_path / "root",
        outputs_dir=tmp_path / "out",
        log_dir=log_dir,
    )
    card = store.new_card(CardType.COMMAND)
    store.set_state(card.id, CardState.AWAITING)
    store.set_state(card.id, CardState.RUNNING)
    assert not any(log_dir.glob("*.jsonl")), "only settled cards are audited"


def test_memory_cap_evicts_oldest_settled_cards(store):
    cards = [store.new_card(CardType.COMMAND) for _ in range(105)]
    for c in cards:
        store.set_state(c.id, CardState.DONE)

    assert len(store.cards()) == 100
    assert store.get(cards[0].id) is None
    assert store.get(cards[-1].id) is not None


def test_running_card_is_never_evicted(store):
    running = store.new_card(CardType.COMMAND)
    store.set_state(running.id, CardState.RUNNING)
    for _ in range(105):
        c = store.new_card(CardType.COMMAND)
        store.set_state(c.id, CardState.DONE)

    assert store.get(running.id) is not None
    assert store.get(running.id).state is CardState.RUNNING
    assert len(store.cards()) <= 101
