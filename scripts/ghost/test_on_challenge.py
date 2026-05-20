"""Verifies Mindflow.on_challenge callback: register observer → send signal → verify verdict.

Tests the challenge observer contract without going through articulate/action loops.
Uses MockSpeech to avoid real TTS.
"""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock import MockGhostMeta
from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.core.blueprint.mindflow import (
    Mindflow, ChallengeObserver, ChallengeVerdict, InputSignal, Impulse,
)
from ghoshell_moss.contracts.speech import Speech

host = Host()
meta = MockGhostMeta(name="test_challenge")
gr = host.run_ghost(meta)

# Inject mock speech before startup to avoid real TTS
gr.container.set(Speech, MockSpeech())

# Collect callback invocations
calls: list[dict] = []


def challenge_observer(
        challenger: Impulse,
        defender: Impulse | None,
        verdict: ChallengeVerdict,
) -> None:
    calls.append({
        "challenger_id": challenger.id,
        "challenger_source": challenger.source,
        "defender_id": defender.id if defender else None,
        "verdict": verdict,
    })


async def main():
    async with gr:
        # ── 1. mindflow exposed + ABC contract ──
        mf = gr.mindflow
        assert isinstance(mf, Mindflow)
        assert mf.is_running()
        assert hasattr(Mindflow, "on_challenge")
        print(f"mindflow: {type(mf).__name__}, is_running={mf.is_running()}")

        # ── 2. faculties include InputSignalNucleus ──
        names = [f.name() for f in mf.faculties()]
        assert "input" in names, f"expected input nucleus, got {names}"
        print(f"faculties: {names}")

        # ── 3. register observer ──
        mf.on_challenge(challenge_observer)

        # ── 4. send signal → wait for challenge ──
        signal = InputSignal().to_signal(
            "你好",
            description="hello ghost",
        )
        mf.add_signal(signal)
        print(f"signal sent: {signal.id}")

        # Give mindflow time to process: signal → nucleus → impulse → challenge
        await asyncio.sleep(0.5)

        # ── 5. verify callback fired ──
        assert len(calls) >= 1, f"expected at least 1 challenge, got {len(calls)}"
        call = calls[0]

        # First signal — no current attention
        assert call["verdict"] == "initial", f"expected initial, got {call['verdict']}"
        assert call["defender_id"] is None
        assert call["challenger_source"] == "input"
        print(f"challenge: verdict={call['verdict']}, source={call['challenger_source']}")

        # ── 6. clear observer (null-op) ──
        mf.on_challenge(None)

        gr.close()

    print("OK — on_challenge observer verified")


asyncio.run(main())
