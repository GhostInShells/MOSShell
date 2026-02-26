import asyncio
import logging
import os
import sys
from pathlib import Path

from ghoshell_moss.transports import script_channel

# Allow running directly via: python examples/script_channel_demo/main.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IS_DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if IS_DEBUG else logging.INFO)


async def main():
    dir = Path(__file__).resolve().parent
    target_script = str((dir / "target_script.py").resolve())

    # 构造一个 proxy channel，并启动子进程 provider。
    proxy = script_channel.channel(
        target_script,
        name="script channel demo",
        description="script channel demo",
        logger=logger,
    )

    async with proxy.bootstrap() as runtime:
        await runtime.wait_connected()

        add = runtime.get_command("add")
        assert add is not None
        print("add(1,2) ->", await add(1, 2))

        hello = runtime.get_command("hello")
        assert hello is not None
        print("hello('moss') ->", await hello("moss"))


if __name__ == "__main__":
    asyncio.run(main())
