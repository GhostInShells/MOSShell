"""Full-link test: input signal → impulse → challenge → articulate → action → output.

Traces the complete three-loop pipeline: sends a signal with MockGhost,
collects all challenge verdicts, output items, and logos stream deltas.
"""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock import MockGhostMeta, MockGhost
from ghoshell_moss.core.blueprint.mindflow import Impulse
from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.contracts.speech import Speech
from ghoshell_moss.core.helpers import ThreadSafeEvent

host = Host()
meta = MockGhostMeta(name="test_full_link")
gr = host.run_ghost(meta)
gr.container.set(Speech, MockSpeech())

# Collectors
output_items: list = []
logos_chunks: list[str] = []
challenge_calls: list[dict] = []

logos_finished = ThreadSafeEvent()
shell_has_any_task = ThreadSafeEvent()


def on_output(item):
    output_items.append(item)


def on_challenge(challenger: Impulse, defender: Impulse | None, verdict: str):
    challenge_calls.append({
        "challenger_source": challenger.to_json(),
        "defender_id": defender.to_json() if defender else None,
        "verdict": verdict,
    })


def _on_any_task_done(task):
    shell_has_any_task.set()


async def collect_logos(session):
    buf = ""
    async for delta in session.get_logos():
        logos_chunks.append(delta)
        buf += delta
        if buf.endswith("\n\n"):
            logos_finished.set()
            buf = ""


async def main():
    async with gr:
        ghost = gr.ghost
        assert isinstance(ghost, MockGhost)
        ghost.set_articulate_responses(["hello from ghost"])

        session = gr.moss.session
        shell = gr.moss.shell
        mindflow = gr.mindflow

        # ── startup checks ──
        assert gr.moss.is_running()
        assert shell.is_running()
        shell.runtime.on_task_done(_on_any_task_done)
        assert mindflow.is_running()
        print(f"shell: running={shell.is_running()}, name={shell.name}")
        print(f"mindflow: {type(mindflow).__name__}, faculties={[name for name in mindflow.faculties().keys()]}")
        print(f"ghost: {ghost.meta.name()}")

        # ── wire observers ──
        session.on_output(on_output)
        mindflow.on_challenge(on_challenge)
        logos_task = asyncio.create_task(collect_logos(session))

        # ── send signal ──
        session.add_input_signal(
            "hello ghost, how are you?",
            description="test input",
        )

        # Wait for logos stream to complete (articulate loop done)
        await asyncio.wait_for(logos_finished.wait(), timeout=5.0)
        await asyncio.sleep(0.0)
        print("logos done, waiting for shell idle...")
        assert shell.is_running()
        task = await shell.wait_any_task()
        print("shell execute task: %s" % task.caller_name())

        # Wait for action loop to finish
        await asyncio.wait_for(shell.wait_until_idle(), timeout=10.0)
        print("shell idle")

        logos_task.cancel()
        try:
            await logos_task
        except asyncio.CancelledError:
            pass

        # ── report ──
        print(f"\n=== challenge ({len(challenge_calls)}) ===")
        for c in challenge_calls:
            print(f"  verdict={c['verdict']}  source={c['challenger_source']}  defender={c['defender_id']}")

        print(f"\n=== output items ({len(output_items)}) ===")
        for o in output_items:
            body = o.messages_string()
            if body:
                print(f"  [{o.role}] {o.log}")
                for line in body.split("\n"):
                    print(f"    | {line}")
            else:
                print(f"  [{o.role}] log={o.log}")

        print(f"\n=== logos ({len(logos_chunks)} chunks, {sum(len(c) for c in logos_chunks)} chars) ===")
        for i, c in enumerate(logos_chunks):
            print(f"  [{i}] {c!r}")

        gr.close()

    print("\nDone — review output above")


asyncio.run(main())
