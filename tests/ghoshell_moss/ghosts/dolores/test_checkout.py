"""``checkout`` — a line switch: validation now, identity at turn/end.

What this pins: the command surface carries a receipt and nothing from the destination; the
transition writes both anchors over one session span, with the arrival durable before the
departure; and the ego's identity does not move until the whole record has landed.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ._support import _OfflineConnection


def _manager(tmp_path: Path):
    from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager
    from ghoshell_moss.memento import new_local_memento

    memento = new_local_memento(tmp_path / "owner", writable=True)
    for name in ("main", "idea", "spare"):
        memento.create_branch(name)
    manager = EgoMementoManager(
        connection=_OfflineConnection(), memento=memento, config=EgoMementoConfig()
    )
    return manager, memento


def _ego(manager):
    from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext

    return DoloresEgo(
        launcher=SimpleNamespace(),
        ctx=DoloresEgoContext(
            project_home=Path("."),
            project_name="pytest",
            name="dolores",
            mode="pytest",
            instruction="i",
            facade=None,
        ),
        memento_manager=manager,
    )


class TestCheckoutRequest:
    """What the model can ask for — and what it is refused."""

    def test_the_receipt_carries_no_destination_content(self, tmp_path: Path):
        """The face arrives with the next turn; a command result would land on the wrong side."""
        manager, _ = _manager(tmp_path)
        receipt = _ego(manager).request_checkout("idea", "carry the long task")
        assert "queued" in receipt
        assert "idea" in receipt
        assert "<memento" not in receipt

    def test_a_second_checkout_in_one_turn_clears_the_first(self, tmp_path: Path):
        """One transition per turn: honouring either silently would leave the ego guessing."""
        from ghoshell_moss.ghosts.dolores._ego import CheckoutRejected

        manager, _ = _manager(tmp_path)
        ego = _ego(manager)
        ego.request_checkout("idea", "one")
        with pytest.raises(CheckoutRejected) as error:
            ego.request_checkout("spare", "two")
        assert "already queued" in str(error.value)
        # the first is gone, and the next request is the one that will run
        assert "queued" in ego.request_checkout("spare", "three")

    def test_the_line_it_already_lives_on_is_refused(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego import CheckoutRejected

        manager, _ = _manager(tmp_path)
        with pytest.raises(CheckoutRejected) as error:
            _ego(manager).request_checkout("main", "nowhere to go")
        assert "already on `main`" in str(error.value)

    def test_an_archived_line_cannot_be_entered(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego import CheckoutRejected

        manager, memento = _manager(tmp_path)
        memento.archive_branch("spare")
        with pytest.raises(CheckoutRejected) as error:
            _ego(manager).request_checkout("spare", "into the archive")
        assert "no active line named `spare`" in str(error.value)

    def test_a_checkout_without_a_title_is_refused(self, tmp_path: Path):
        """The title is what both anchors will say; a checkout that says nothing is not a record."""
        from ghoshell_moss.ghosts.dolores._ego import CheckoutRejected

        manager, _ = _manager(tmp_path)
        with pytest.raises(CheckoutRejected) as error:
            _ego(manager).request_checkout("idea", "   ")
        assert "why it leaves" in str(error.value)


class TestCheckoutTransition:
    """The turn/end half — two anchors, one span, identity untouched until both are durable."""

    @pytest.mark.asyncio
    async def test_both_anchors_carry_the_same_session_span(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego import CheckoutRequest, run_checkout
        from ghoshell_moss.memento import CommitMessage

        manager, memento = _manager(tmp_path)
        outcome = await run_checkout(
            manager,
            CheckoutRequest(name="idea", title="carry the long task", note="state of play"),
            session_id="s1",
            start_turn=4,
            end_turn=5,
        )
        assert outcome.surface is not None, outcome.reason

        main_tip = memento.get_branch("main").commits()[-1]
        idea_tip = memento.get_branch("idea").commits()[-1]
        # One event, two records: compare the whole structure, so a field added to the ref is
        # covered without editing this test. `created` is when each record was written — the one
        # thing that legitimately differs between two anchors of the same span.
        from ghoshell_moss.deepseek_harness.v0_2_0 import DshSessionRef

        departure_ref = DshSessionRef(**main_tip.metadata["ref"])
        arrival_ref = DshSessionRef(**idea_tip.metadata["ref"])
        assert departure_ref.model_dump(exclude={"created"}) == arrival_ref.model_dump(
            exclude={"created"}
        )
        assert departure_ref.end_turn == 5

        arrival = CommitMessage.parse(memento.get_branch("idea").notes()[idea_tip.id].message)
        departure = CommitMessage.parse(memento.get_branch("main").notes()[main_tip.id].message)
        assert arrival.op == "arrive"
        assert "main" in arrival.target
        assert departure.op == "depart"
        assert "idea" in departure.target
        assert arrival.content == departure.content == "carry the long task"

    @pytest.mark.asyncio
    async def test_identity_is_the_callers_last_step(self, tmp_path: Path):
        """`run_checkout` never moves the ego: the caller does, after it gets a surface back."""
        from ghoshell_moss.ghosts.dolores._ego import CheckoutRequest, run_checkout

        manager, _ = _manager(tmp_path)
        await run_checkout(
            manager,
            CheckoutRequest(name="idea", title="t"),
            session_id="s1",
            start_turn=4,
            end_turn=5,
        )
        assert manager.active_branch == "main"

    @pytest.mark.asyncio
    async def test_an_empty_span_aborts_before_anything_is_written(self, tmp_path: Path):
        """No completed turn = nothing honest to record; the transition must not invent a span."""
        from ghoshell_moss.ghosts.dolores._ego import CheckoutRequest, run_checkout

        manager, memento = _manager(tmp_path)
        outcome = await run_checkout(
            manager,
            CheckoutRequest(name="idea", title="t"),
            session_id="s1",
            start_turn=5,
            end_turn=5,
        )
        assert outcome.surface is None
        assert "no completed turn" in outcome.reason
        assert memento.get_branch("main").commits() == []
        assert memento.get_branch("idea").commits() == []
