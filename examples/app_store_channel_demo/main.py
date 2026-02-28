import asyncio
import sys
from pathlib import Path

from ghoshell_moss.channels import AppStoreChannel, AppStoreConfig

# Allow running directly via: python examples/app_store_channel_demo/main.py
ROOT = Path(__file__).resolve().parents[0]
print("ROOT=", ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def main() -> None:
    store = AppStoreChannel(
        AppStoreConfig(
            name="app_store",
            description="Demo AppStoreChannel",
            root_dir=str((ROOT / "apps").resolve()),
            rescan_on_refresh=True,
        )
    )

    async with store.bootstrap() as runtime:
        await runtime.wait_connected()
        await runtime.refresh_metas()

        print("--- available apps ---")
        meta = runtime.own_meta()
        if meta and meta.context:
            # meta.context is a list[Message], with `text` content.
            print(meta.context[0].contents[0]["data"]["text"])

        read = runtime.get_command("read")
        open_cmd = runtime.get_command("open")
        close_cmd = runtime.get_command("close")
        assert read is not None
        assert open_cmd is not None
        assert close_cmd is not None

        print("\n--- read explicit_chan ---")
        print(await read(name="explicit_chan"))

        # 1) python-mode
        await open_cmd(name="adder_py")
        add = runtime.get_command("adder_py:add")
        hello = runtime.get_command("adder_py:hello")
        assert add is not None
        assert hello is not None
        print("\n[adder_py] add(1,2)=", await add(a=1, b=2))
        print("[adder_py] hello('moss')=", await hello(name="moss"))
        await close_cmd(name="adder_py")

        # 2) module-mode
        await open_cmd(name="math_mod")
        mul = runtime.get_command("math_mod:mul")
        pow2 = runtime.get_command("math_mod:pow2")
        assert mul is not None
        assert pow2 is not None
        print("\n[math_mod] mul(3,4)=", await mul(a=3, b=4))
        print("[math_mod] pow2(9)=", await pow2(x=9))
        await close_cmd(name="math_mod")

        # 3) channel-mode
        await open_cmd(name="explicit_chan")
        ping = runtime.get_command("explicit_chan:ping")
        echo = runtime.get_command("explicit_chan:echo")
        assert ping is not None
        assert echo is not None
        print("\n[explicit_chan] ping()=", await ping())
        print("[explicit_chan] echo('hi')=", await echo(text="hi"))
        await close_cmd(name="explicit_chan")

        await runtime.refresh_metas()
        print("\n--- after close all ---")
        meta = runtime.own_meta()
        if meta and meta.context:
            print(meta.context[0].contents[0]["data"]["text"])


if __name__ == "__main__":
    asyncio.run(main())
