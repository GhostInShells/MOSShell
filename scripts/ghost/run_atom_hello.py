"""Send 'hello' to an Atom ghost and print logos + inspect_context.

Uses ANTHROPIC_MODEL env var for model selection.
"""
import asyncio
import json
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.atom._meta import AtomMeta
from ghoshell_moss.contracts.speech import Speech
from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.core.helpers import ThreadSafeEvent

host = Host()
meta = AtomMeta(
    name="atom_hello",
    soul_content="You are a friendly assistant.",
)
gr = host.run_ghost(meta)
# gr.container.set(Speech, MockSpeech())

logos_finished = ThreadSafeEvent()


async def collect_logos(session):
    buf = ""
    async for delta in session.get_logos():
        print("++++++++++++", delta, end="", flush=True)
        buf += delta
        if buf.endswith("\n\n"):
            logos_finished.set()
            buf = ""


async def main():
    async with gr:
        session = gr.moss.session
        logos_task = asyncio.create_task(collect_logos(session))

        print("=== sending: hello ===\n")
        session.add_input_signal("hello", description="test")

        await asyncio.wait_for(logos_finished.wait(), timeout=30.0)
        await asyncio.sleep(0.1)
        task = await gr.moss.shell.wait_any_task()
        print("++++++++ task", task)

        await gr.moss.shell.wait_until_idle()

        logos_task.cancel()
        try:
            await logos_task
        except asyncio.CancelledError:
            pass

        ghost = gr.ghost
        print("\n\n=== inspect_context (moment sent to model) ===")
        ctx = ghost.inspect_context()
        if ctx:
            print(json.dumps(ctx, indent=2, ensure_ascii=False))
        else:
            print("(empty — Atom may not override inspect_context yet)")

        gr.close()


if __name__ == "__main__":
    asyncio.run(main())
