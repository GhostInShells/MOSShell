import asyncio
import sys
from pathlib import Path

from ghoshell_moss.transports.script_channel import ScriptChannelHub, ScriptHubConfig, ScriptProxyConfig

# Allow running directly via: python examples/script_channel_hub_demo/main.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def main() -> None:
    scripts_dir = Path(__file__).resolve().parent / "scripts"

    hub = ScriptChannelHub(
        ScriptHubConfig(
            name="hub",
            description="demo hub managing multiple script channels",
            root_dir=str(scripts_dir),
            proxies={
                "math": ScriptProxyConfig(target="math_ops.py", description="math operations"),
                "text": ScriptProxyConfig(target="text_ops.py", description="text operations"),
            },
        )
    ).as_channel()

    async with hub.bootstrap() as runtime:
        # start children (spawn subprocess + handshake)
        start = runtime.get_command("start")
        status = runtime.get_command("status")
        stop = runtime.get_command("stop")
        restart = runtime.get_command("restart")
        assert start is not None
        assert status is not None
        assert stop is not None
        assert restart is not None

        await runtime.refresh_metas()
        print("--- init ---")
        print(runtime.own_meta().description)
        print(await status())

        await start("math")
        await start("text")

        await runtime.refresh_metas()
        print("--- after start ---")
        print(runtime.own_meta().description)
        print(await status())

        add = runtime.get_command("math:add")
        assert add is not None
        print("math.add(1,2)=", await add(1, 2))

        hello = runtime.get_command("text:hello")
        assert hello is not None
        print("text.hello('moss')=", await hello("moss"))

        await restart("text")
        await runtime.refresh_metas()
        print("--- after restart(text) ---")
        print(await status())

        await stop("math")
        await stop("text")

        await runtime.refresh_metas()
        print("--- after stop ---")
        print(runtime.own_meta().description)
        print(await status())


if __name__ == "__main__":
    asyncio.run(main())
