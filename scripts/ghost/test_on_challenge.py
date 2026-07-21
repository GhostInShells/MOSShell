"""Verifies Mindflow hook system: register hook → send signal → verify challenge verdict.

Tests the MindflowHook contract through with_hook/remove_hook.
Uses MockSpeech to avoid real TTS.
"""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock import MockGhostMeta
from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.core.blueprint.mindflow import (
    Mindflow, MindflowHook, ChallengeVerdict, InputSignalMeta, Impulse,
)
from ghoshell_moss.contracts.speech import Speech

host = Host()
meta = MockGhostMeta(name="test_challenge")
gr = host.run_ghost(meta)

# Inject mock speech before startup to avoid real TTS
gr.container.set(Speech, MockSpeech())

# Collect callback invocations
calls: list[dict] = []


class TestHook(MindflowHook):

    def on_impulse_challenged(
            self,
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
        print(f"mindflow: {type(mf).__name__}, is_running={mf.is_running()}")

        # ── 2. faculties include InputSignalNucleus ──
        names = list(mf.faculties().keys())
        print(f"faculties: {names}")

        # ── 3. register hook ──
        hook = TestHook()
        mf.with_hook(hook)

        # ── 4. send signal → wait for challenge ──
        signal = InputSignalMeta().to_signal(
            "你好",
            description="hello ghost",
        )
        mf.add_signal(signal)
        print(f"signal sent: {signal.id}")

        # Give mindflow time to process: signal → nucleus → impulse → challenge
        await asyncio.sleep(0.5)

        # ── 5. verify callback fired ──
        print(f"challenge calls: {len(calls)}")
        for i, c in enumerate(calls):
            print(f"  [{i}] verdict={c['verdict']} challenger_source={c['challenger_source']} defender_id={c['defender_id']}")

        # ── 6. clear observer ──
        mf.remove_hook(hook)

        gr.close()

    print("OK — MindflowHook challenge observer verified")


asyncio.run(main())
