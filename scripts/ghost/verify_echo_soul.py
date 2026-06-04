"""Verify echo ghost soul is loaded from GhostWorkspace.

Starts echo via Host.run_ghost("echo"), enters runtime context,
then checks soul_content on the meta.
"""
import asyncio
from ghoshell_moss.host import Host

host = Host()

ghosts = host.all_ghosts()
assert "echo" in ghosts, f"echo not found in: {list(ghosts.keys())}"
print(f"discovered: {list(ghosts.keys())}")

gr = host.run_ghost("echo")


async def main():
    async with gr:
        ghost = gr.ghost
        soul = ghost.meta.soul_content

        assert len(soul) > 100, f"soul too short ({len(soul)} chars)"
        assert "Echo" in soul, f"missing identity marker"
        assert "并行存在" in soul or "回响" in soul, f"missing key content"

        print(f"soul loaded: {len(soul)} chars")
        print(f"--- first 300 chars ---")
        print(soul[:300])
        print(f"--- last 200 chars ---")
        print(soul[-200:])

        # verify soul is assembled into the full instruction
        full_instruction = ghost.meta.build_instruction_from_ioc(gr.container)
        assert "Echo" in full_instruction, "soul not in full instruction"
        assert "CTML" in full_instruction, "missing CTML instruction"
        print(f"\nfull instruction: {len(full_instruction)} chars (CTML + project + mode + static + soul)")

        gr.close()

    print("\nOK — echo soul verified")


asyncio.run(main())
