"""Verify GhostRuntime teardown does not hang (cleanup deadlock check)."""
import asyncio
from ghoshell_moss.host import Host

host = Host()
gr = host.run_ghost("echo")


async def main():
    async with gr:
        ghost = gr.ghost
        soul = ghost.meta.soul_content
        assert len(soul) > 100, f"soul not loaded ({len(soul)} chars)"
        print(f"echo started, soul: {len(soul)} chars")

        session = gr.moss.session
        session.add_input_signal("hello")
        await asyncio.sleep(0.5)

    print("__aexit__ completed — teardown clean")

    # close() is a signal, not the teardown itself
    gr.close()
    print("close() returned")


try:
    asyncio.run(asyncio.wait_for(main(), timeout=15.0))
    print("OK — no deadlock")
except asyncio.TimeoutError:
    print("FAIL: teardown timed out after 15s — deadlock suspected")
