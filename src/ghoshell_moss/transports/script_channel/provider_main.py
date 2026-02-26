import argparse
import asyncio
import json
import logging
import sys

from ghoshell_moss.channels.module_channel import ModuleChannel

from .script_channel import ScriptChannelProvider, ScriptProviderConfig, StdioProviderConnection


async def _build_stdio_connection(logger: logging.Logger | None = None) -> StdioProviderConnection:
    loop = asyncio.get_running_loop()

    reader = asyncio.StreamReader()
    read_protocol = asyncio.StreamReaderProtocol(reader)
    # NOTE: use `.buffer` to ensure we always get bytes.
    await loop.connect_read_pipe(lambda: read_protocol, sys.stdin.buffer)

    write_transport, write_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin,
        sys.stdout.buffer,
    )
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)
    return StdioProviderConnection(reader=reader, writer=writer, logger=logger)


async def _arun_provider(config: ScriptProviderConfig) -> None:
    provider_logger = logging.getLogger("ghoshell_moss.script_channel.provider")
    connection = await _build_stdio_connection(provider_logger)
    provider = ScriptChannelProvider(connection=connection)

    channel = ModuleChannel(**config.module_channel_kwargs)

    async with provider.arun(channel):
        await provider.wait_stop()


def run_provider_main(config: ScriptProviderConfig) -> None:
    """Run a stdio-based provider for a module/script.

    This helper is intended for use in a dedicated python process.
    """

    asyncio.run(_arun_provider(config))


def _parse_args(argv: list[str]) -> ScriptProviderConfig:
    parser = argparse.ArgumentParser(description="ghoshell-moss script_channel provider")
    parser.add_argument("--config-json", required=True, help="JSON serialized ScriptProviderConfig")
    args = parser.parse_args(argv)
    raw = json.loads(args.config_json)
    return ScriptProviderConfig(**raw)


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    config = _parse_args(argv)
    run_provider_main(config)


if __name__ == "__main__":
    main()
