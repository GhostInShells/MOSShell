import asyncio
import logging
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import frontmatter
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field, ValidationError

from ghoshell_moss.core import Channel, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command, CommandWrapper, PyCommand
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.concepts.runtime import AbsChannelTreeRuntime
from ghoshell_moss.message.utils import new_text_message
from ghoshell_moss.transports.script_channel.script_channel import ScriptChannelProxy

__all__ = [
    "AppStoreChannel",
    "AppStoreConfig",
]


class AppStoreConfig(BaseModel):
    name: str = Field(default="app_store", description="app store channel name")
    description: str = Field(default="Manage apps discovered from CHANNEL.md manifests.")
    root_dir: str = Field(description="apps root directory")

    rescan_on_refresh: bool = Field(
        default=True,
        description="Whether to re-scan root_dir on every meta refresh.",
    )
    env: dict[str, str] = Field(default_factory=dict, description="environment vars passed to child providers")
    cwd_is_app_dir: bool = Field(
        default=True,
        description="When launching children, set cwd to the app directory.",
    )
    python_executable: str | None = Field(
        default=None,
        description="Optional python executable for child processes.",
    )


class _ChannelManifest(BaseModel):
    name: str
    description: str = ""
    main: str
    mode: Literal["python", "module", "channel"] = "module"


@dataclass
class _AppDefinition:
    name: str
    app_dir: Path
    manifest_path: Path
    manifest: _ChannelManifest
    body_markdown: str
    raw_markdown: str
    main_ref: str
    main_args: list[str]


@dataclass
class _RunningApp:
    name: str
    proxy: Channel
    opened_at: float
    definition: _AppDefinition | None


class AppStoreChannel(Channel):
    """A channel that discovers and manages script-backed "apps".

    Each app is represented by a directory containing a ``CHANNEL.md`` file.
    The file uses YAML front matter to declare the child channel's name,
    description, entrypoint (``main``), and mode.

    App definitions are discovered dynamically by scanning ``root_dir``.
    """

    def __init__(self, config: AppStoreConfig):
        self._config = config
        self._uid = uuid()

    def name(self) -> str:
        return self._config.name

    def id(self) -> str:
        return self._uid

    def description(self) -> str:
        return self._config.description

    def bootstrap(self, container: Optional[IoCContainer] = None) -> ChannelRuntime:
        return AppStoreRuntime(channel=self, container=container)


class AppStoreRuntime(AbsChannelTreeRuntime):
    def __init__(self, *, channel: AppStoreChannel, container: IoCContainer | None = None):
        super().__init__(channel=channel, container=container)
        self._logger = logging.getLogger(__name__)

        self._cfg = channel._config
        self._root_dir = Path(self._cfg.root_dir).expanduser().resolve()

        self._apps: dict[str, _AppDefinition] = {}
        self._running: dict[str, _RunningApp] = {}
        self._sub_channels: dict[str, Channel] = {}
        self._apps_lock = asyncio.Lock()

        self._commands: dict[str, Command] = {
            "open": PyCommand(self._open_cmd, chan=self._name, name="open", blocking=True),
            "close": PyCommand(self._close_cmd, chan=self._name, name="close", blocking=True),
            "read": PyCommand(self._read_cmd, chan=self._name, name="read", blocking=True),
        }

        self._scan_apps_sync(force=True)

    def sub_channels(self) -> dict[str, Channel]:
        return self._sub_channels

    def is_connected(self) -> bool:
        # In-process runtime is always connected.
        return True

    async def wait_connected(self) -> None:
        return

    async def on_start_up(self) -> None:
        return

    async def on_close(self) -> None:
        for name in list(self._sub_channels.keys()):
            await self._close_app(name)
        self._running.clear()
        self._sub_channels.clear()

    async def on_running(self) -> None:
        while not self._closing_event.is_set():
            await asyncio.sleep(0.5)

    async def on_idle(self) -> None:
        return

    async def clear_own(self) -> None:
        return

    def default_states(self) -> list:
        return []

    def _is_available(self) -> bool:
        return True

    def own_commands(self, available_only: bool = True) -> dict[str, Command]:
        if not self.is_available():
            return {}
        return {name: self._wrap_origin_command(cmd) for name, cmd in self._commands.items()}

    def get_own_command(self, name: str) -> Optional[Command]:
        return self._wrap_origin_command(self._commands.get(name))

    def _wrap_origin_command(self, command: Command | None) -> Command | None:
        if command is None:
            return None

        async def _run_with_runtime(*args, **kwargs):
            from ghoshell_moss.core.concepts.channel import ChannelCtx

            ctx = ChannelCtx(self)
            async with ctx.in_ctx():
                return await command(*args, **kwargs)

        return CommandWrapper.wrap(command, func=_run_with_runtime)

    async def _generate_own_metas(self, force: bool):
        from ghoshell_moss.core.concepts.channel import ChannelMeta

        if force or self._cfg.rescan_on_refresh:
            async with self._apps_lock:
                self._scan_apps_sync(force=True)

        async with self._apps_lock:
            children = sorted(self._apps.keys())
            lines = [self._format_app_line(name) for name in children]
            context_text = "\n".join(
                [
                    "Available apps (open/close/read):",
                    *(lines or ["(none)"]),
                ]
            )

        meta = ChannelMeta(
            name=self._name,
            description=self.channel.description(),
            channel_id=self.id,
            available=True,
            commands=[cmd.meta() for cmd in self._commands.values()],
            children=children,
            instructions=[],
            context=[new_text_message(context_text, role="system")],
        )
        meta.dynamic = True
        return {"": meta}

    def _format_app_line(self, name: str) -> str:
        definition = self._apps.get(name)
        desc = definition.manifest.description if definition is not None else ""
        state = self._format_state(name)
        return f"- {name}: {desc} | {state}".rstrip()

    def _format_state(self, name: str) -> str:
        running = self._running.get(name)
        if running is None or name not in self._sub_channels:
            return "❌ stopped"

        runtime = self.importlib.get_channel_runtime(running.proxy)
        if runtime is None or not runtime.is_running():
            return "❌ stopped"

        if runtime.is_connected():
            pid = getattr(running.proxy, "connection", None)
            pid_val = None
            start_time = None
            if pid is not None:
                pid_val = getattr(pid, "pid", None)
                start_time = getattr(pid, "start_time", None)
            pid_part = f"PID: {pid_val}" if pid_val is not None else "PID: ?"
            runtime_secs = ""
            if start_time is not None:
                runtime_secs = f", {max(0.0, time.time() - start_time):.1f}s"
            return f"✅ running ({pid_part}{runtime_secs})"

        return "⚠️ connecting"

    def _scan_apps_sync(self, *, force: bool = False) -> None:
        if not force and self._apps:
            return

        apps: dict[str, _AppDefinition] = {}
        if not self._root_dir.exists():
            self._logger.warning("AppStore root_dir not found: %s", self._root_dir)
            self._apps = apps
            return

        for entry in sorted(self._root_dir.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "CHANNEL.md"
            if not manifest_path.exists() or not manifest_path.is_file():
                continue
            try:
                definition = self._load_app_definition(entry, manifest_path)
            except Exception as exc:
                self._logger.warning("Failed to load app manifest %s: %s", manifest_path, exc)
                continue
            if definition.name in apps:
                self._logger.warning(
                    "Duplicate app name %s found in %s and %s; keeping first.",
                    definition.name,
                    apps[definition.name].manifest_path,
                    definition.manifest_path,
                )
                continue
            apps[definition.name] = definition

        self._apps = apps

    @staticmethod
    def _parse_main(main: str) -> tuple[str, list[str]]:
        tokens = shlex.split(main)
        if not tokens:
            raise ValueError("main must not be empty")
        return tokens[0], tokens[1:]

    def _load_app_definition(self, app_dir: Path, manifest_path: Path) -> _AppDefinition:
        raw_markdown = manifest_path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw_markdown)
        try:
            manifest = _ChannelManifest(**post.metadata)
        except ValidationError as exc:
            raise ValueError(f"Invalid CHANNEL.md front matter: {exc}") from exc

        main_ref, main_args = self._parse_main(manifest.main)

        body_markdown = (post.content or "").strip()
        return _AppDefinition(
            name=manifest.name,
            app_dir=app_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            body_markdown=body_markdown,
            raw_markdown=raw_markdown,
            main_ref=main_ref,
            main_args=main_args,
        )

    def _resolve_main_target(self, definition: _AppDefinition) -> tuple[str, list[str]]:
        main_ref = definition.main_ref
        candidate = Path(main_ref).expanduser()
        if not candidate.is_absolute():
            candidate = (definition.app_dir / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            return str(candidate), list(definition.main_args)
        return main_ref, list(definition.main_args)

    def _build_proxy(self, definition: _AppDefinition) -> Channel:
        target, args = self._resolve_main_target(definition)
        cfg = self._cfg

        detail_description = definition.manifest.description
        if definition.body_markdown:
            detail_description = (detail_description + "\n\n" + definition.body_markdown).strip()

        provider_launcher = str(
            (Path(__file__).resolve().parents[1] / "transports" / "script_channel" / "launcher.py").resolve()
        )

        provider_target: str | None = target
        provider_channel_file: str | None = None

        # `python` mode: treat `main` as a python script file and load it as a module
        # inside the provider process (same python interpreter as parent by default).
        if definition.manifest.mode == "python":
            target_path = Path(target)
            if not target_path.exists() or not target_path.is_file():
                raise FileNotFoundError(f"python mode requires a script file, got: {target}")
            provider_target = str(target_path)

        # `channel` mode: `main` points to a python file defining `__channel__`.
        if definition.manifest.mode == "channel":
            target_path = Path(target)
            if not target_path.exists() or not target_path.is_file():
                raise FileNotFoundError(f"channel mode requires a python file, got: {target}")
            provider_target = str(target_path)
            provider_channel_file = str(target_path)

        env = {"PYTHONUNBUFFERED": "1", **dict(cfg.env)}
        if definition.manifest.mode == "module":
            # Make sure the parent directory of `apps/` is importable in the child
            # provider. This is required for `mode=module` when `main` is a dotted
            # path like `apps.xxx.yyy`.
            apps_parent = str(self._root_dir.parent)
            existing_py_path = env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
            parts = [p for p in existing_py_path.split(os.pathsep) if p]
            if apps_parent not in parts:
                parts.insert(0, apps_parent)
            env["PYTHONPATH"] = os.pathsep.join(parts)

        return ScriptChannelProxy(
            name=definition.name,
            description=detail_description,
            provider_launcher=provider_launcher,
            provider_target=provider_target,
            cwd=str(definition.app_dir) if cfg.cwd_is_app_dir else None,
            env=env,
            python_executable=cfg.python_executable,
            args=args,
            provider_channel_file=provider_channel_file,
        )

    async def _open_cmd(self, name: str, timeout: float = 15.0) -> str:
        async with self._apps_lock:
            definition = self._apps.get(name)

        if definition is None:
            raise CommandErrorCode.NOT_FOUND.error(f"app {name} not found")

        await self._close_app(name)

        proxy = self._build_proxy(definition)
        self._sub_channels[name] = proxy
        self._running[name] = _RunningApp(name=name, proxy=proxy, opened_at=time.time(), definition=definition)

        runtime = await self.importlib.get_or_create_channel_runtime(proxy)
        if runtime is None:
            raise CommandErrorCode.NOT_FOUND.error(f"app {name} runtime not available")
        try:
            await asyncio.wait_for(runtime.wait_connected(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._close_app(name)
            raise CommandErrorCode.TIMEOUT.error(f"open app {name} timeout")
        return ""

    async def _close_app(self, name: str) -> None:
        running = self._running.get(name)
        if running is None:
            self._sub_channels.pop(name, None)
            return
        runtime = self.importlib.get_channel_runtime(running.proxy)
        if runtime is not None and runtime.is_running():
            await runtime.close()
        self._sub_channels.pop(name, None)
        self._running.pop(name, None)

    async def _close_cmd(self, name: str, timeout: float = 5.0) -> str:
        try:
            await asyncio.wait_for(self._close_app(name), timeout=timeout)
        except asyncio.TimeoutError:
            raise CommandErrorCode.TIMEOUT.error(f"close app {name} timeout")
        return f"App {name} closed."

    async def _read_cmd(self, name: str) -> str:
        async with self._apps_lock:
            definition = self._apps.get(name)
        if definition is None:
            raise CommandErrorCode.NOT_FOUND.error(f"app {name} not found")

        header_lines = [
            f"# {definition.manifest.name}",
            f"- description: {definition.manifest.description}",
            f"- mode: {definition.manifest.mode}",
            f"- main: {definition.manifest.main}",
            f"- dir: {definition.app_dir}",
        ]
        if definition.body_markdown:
            return "\n".join([*header_lines, "", definition.body_markdown])
        return "\n".join(header_lines)
