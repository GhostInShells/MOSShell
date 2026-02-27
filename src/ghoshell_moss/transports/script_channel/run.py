import ast
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from ghoshell_container import IoCContainer

from .script_channel import ScriptChannelProxy


def channel(
    target_script: str,
    *,
    name: str | None = None,
    description: str = "",
    python_executable: str | None = None,
    stderr: str = "pipe",
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    provider_include: list[str] | None = None,
    provider_channel_file: str | None = None,
    provider_expect_channel_val: str = "__channel__",
    provider_reload_on_bootstrap: bool = False,
    logger: logging.Logger | None = None,
) -> ScriptChannelProxy:
    """Build a :class:`ScriptChannelProxy` for a target python script.

    This is the ergonomic entrypoint similar to ``uvicorn.run(app)`` patterns.
    Internally it starts a child provider process that wraps the target script
    with :class:`~ghoshell_moss.channels.module_channel.ModuleChannel`.

    Notes:
        The stdio transport uses provider stdout as protocol stream.
        Protocol lines are prefixed, so target scripts can print to stdout
        without breaking the channel (non-protocol lines will be ignored).
    """

    target_path = Path(target_script).expanduser()
    if not target_path.is_absolute():
        target_path = (Path.cwd() / target_path).resolve()

    proxy_name = name or target_path.stem

    # Use the built-in launcher to start provider_main in a child process.
    launcher = str((Path(__file__).resolve().parent / "launcher.py").resolve())

    return ScriptChannelProxy(
        name=proxy_name,
        description=description,
        provider_launcher=launcher,
        provider_target=str(target_path),
        python_executable=python_executable,
        stderr=stderr,
        env=env,
        cwd=cwd,
        provider_include=provider_include,
        provider_channel_file=provider_channel_file,
        provider_expect_channel_val=provider_expect_channel_val,
        provider_reload_on_bootstrap=provider_reload_on_bootstrap,
        logger=logger,
    )


def run(
    target_script: str,
    *,
    name: str | None = None,
    description: str = "",
    python_executable: str | None = None,
    stderr: str = "pipe",
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    provider_include: list[str] | None = None,
    provider_channel_file: str | None = None,
    provider_expect_channel_val: str = "__channel__",
    provider_reload_on_bootstrap: bool = False,
    container: Optional[IoCContainer] = None,
    interactive: bool = False,
) -> None:
    """Run a target python script as a child process channel.

    This function blocks the current process until cancelled (Ctrl+C).
    """

    proxy = channel(
        target_script,
        name=name,
        description=description,
        python_executable=python_executable,
        stderr=stderr,
        env=env,
        cwd=cwd,
        provider_include=provider_include,
        provider_channel_file=provider_channel_file,
        provider_expect_channel_val=provider_expect_channel_val,
        provider_reload_on_bootstrap=provider_reload_on_bootstrap,
    )

    async def _readline() -> str:
        # Use to_thread to avoid blocking the event loop.
        return await asyncio.to_thread(input, "moss> ")

    def _print_help() -> None:
        print(
            "\n".join(
                [
                    "Commands:",
                    "  help               Show this help",
                    "  ls                 List available commands",
                    "  quit / exit        Exit",
                    "\nCall syntax:",
                    "  <cmd> key=value [key=value ...]",
                    "  <cmd> {json}",
                    "\nExamples:",
                    "  add a=1 b=2",
                    "  hello name='moss'",
                    '  add {"a": 1, "b": 2}',
                    '  foo {"args": [1,2], "kwargs": {"x": 3}}',
                ]
            )
        )

    def _parse_kv(tokens: list[str]) -> dict:
        kwargs: dict[str, object] = {}
        for token in tokens:
            if "=" not in token:
                raise ValueError(f"invalid token (expect key=value): {token}")
            key, value_str = token.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"invalid token (empty key): {token}")
            value_str = value_str.strip()
            if value_str == "":
                kwargs[key] = ""
                continue
            try:
                kwargs[key] = ast.literal_eval(value_str)
            except Exception:
                # fallback: treat as raw string
                kwargs[key] = value_str
        return kwargs

    async def _main() -> None:
        async with proxy.bootstrap(container=container) as runtime:
            await runtime.wait_connected()

            if not interactive:
                # Block forever until cancelled.
                await asyncio.Event().wait()
                return

            _print_help()

            while True:
                try:
                    line = (await _readline()).strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not line:
                    continue
                if line in ("quit", "exit"):
                    break
                if line == "help":
                    _print_help()
                    continue
                if line == "ls":
                    meta = runtime.own_meta()
                    if meta is None:
                        print("(no meta yet)")
                        continue
                    names = [c.name for c in meta.commands]
                    print("commands:", ", ".join(names) if names else "(none)")
                    continue

                parts = line.split(maxsplit=1)
                cmd_name = parts[0]
                rest = parts[1].strip() if len(parts) > 1 else ""

                cmd = runtime.get_command(cmd_name)
                if cmd is None:
                    print(f"unknown command: {cmd_name}")
                    continue

                args = []
                kwargs = {}
                if rest:
                    if rest.startswith("{"):
                        payload = json.loads(rest)
                        if isinstance(payload, dict) and ("args" in payload or "kwargs" in payload):
                            args = list(payload.get("args") or [])
                            kwargs = dict(payload.get("kwargs") or {})
                        elif isinstance(payload, dict):
                            kwargs = payload
                        else:
                            raise ValueError("json payload must be an object, or {args:[...], kwargs:{...}}")
                    else:
                        kwargs = _parse_kv(rest.split())

                try:
                    result = await cmd(*args, **kwargs)
                    print(result)
                except Exception as exc:
                    print(f"error: {exc}")

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        return
