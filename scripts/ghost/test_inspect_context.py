"""Inspect ghost context after one articulate cycle.

Sends a single input signal, waits for articulate to complete,
then prints ghost.inspect_context() and ghost.inspect_state() as formatted JSON.
"""
import asyncio
import json
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock import MockGhostMeta, MockGhost
from ghoshell_moss.contracts.speech import Speech
from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.core.helpers import ThreadSafeEvent

host = Host()
meta = MockGhostMeta(name="test_inspect_context")
gr = host.run_ghost(meta)
gr.container.set(Speech, MockSpeech())

logos_finished = ThreadSafeEvent()


async def collect_logos(session):
    buf = ""
    async for delta in session.get_logos():
        buf += delta
        if buf.endswith("\n\n"):
            logos_finished.set()
            buf = ""


async def main():
    async with gr:
        ghost = gr.ghost
        assert isinstance(ghost, MockGhost)
        ghost.set_articulate_responses(["hello world\n\n"])

        session = gr.moss.session
        logos_task = asyncio.create_task(collect_logos(session))

        session.add_input_signal("tell me about yourself", description="test")

        await asyncio.wait_for(logos_finished.wait(), timeout=5.0)
        await asyncio.sleep(0.0)

        logos_task.cancel()
        try:
            await logos_task
        except asyncio.CancelledError:
            pass

        print("=== inspect_state ===")
        print(json.dumps(ghost.inspect_state(), indent=2, ensure_ascii=False))

        print("\n=== inspect_context (full moment) ===")
        print(json.dumps(ghost.inspect_context(), indent=2, ensure_ascii=False))

        gr.close()


asyncio.run(main())
