import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field

from ghoshell_moss.core import Channel, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command, CommandWrapper, PyCommand
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.concepts.runtime import AbsChannelTreeRuntime

from .run import channel as build_script_channel
from .script_channel import ScriptChannelProxy

__all__ = [
    "ScriptChannelHub",
    "ScriptHubConfig",
    "ScriptProxyConfig",
]


class ScriptProxyConfig(BaseModel):
    target: str = Field(description="python script path to be loaded as a channel")
    description: str = Field(default="", description="the description of the script channel")


class ScriptHubConfig(BaseModel):
    name: str = Field(description="name of the hub")
    description: str = Field(description="description of the hub")
    root_dir: str = Field(description="directory containing target scripts")
    proxies: dict[str, ScriptProxyConfig] = Field(default_factory=dict)


class ScriptChannelHub:
    """A hub that manages multiple script-backed channels.

    Similar behavior to ``ZMQChannelHub``:

    - exposes `start`/`stop`/`restart`/`status` commands
    - keeps configured children visible in metas
    - starts provider subprocess lazily when a child runtime starts
    """

    def __init__(self, config: ScriptHubConfig):
        self._config = config

    def as_channel(self) -> Channel:
        return _ScriptHubChannel(config=self._config)


class _ScriptHubChannel(Channel):
    def __init__(self, *, config: ScriptHubConfig):
        self._config = config
        self._uid = uuid()

    def name(self) -> str:
        return self._config.name

    def id(self) -> str:
        return self._uid

    def description(self) -> str:
        return self._config.description

    def bootstrap(self, container: Optional[IoCContainer] = None) -> ChannelRuntime:
        return _ScriptHubRuntime(channel=self, container=container)


@dataclass
class _ChildEntry:
    name: str
    target_path: str
    description: str
    proxy: ScriptChannelProxy | None = None


class _ScriptHubRuntime(AbsChannelTreeRuntime):
    def __init__(self, *, channel: _ScriptHubChannel, container: IoCContainer | None = None):
        super().__init__(channel=channel, container=container)

        cfg = channel._config
        root = Path(cfg.root_dir).expanduser()
        self._children: dict[str, _ChildEntry] = {
            name: _ChildEntry(
                name=name,
                target_path=str((root / child_cfg.target).resolve()),
                description=child_cfg.description,
            )
            for name, child_cfg in cfg.proxies.items()
        }

        # Important: keep sub-channels empty by default.
        # AbsChannelRuntime.start() will recursively start all sub-channels.
        # ScriptChannelProxy spawns a subprocess in `connection.start()`, so we
        # must mount children lazily (only after `start <name>` is called).
        self._sub_channels: dict[str, Channel] = {}

        self._commands: dict[str, Command] = {
            "start": PyCommand(self._start_child_cmd, chan=self._name, name="start", blocking=True),
            "stop": PyCommand(self._stop_child_cmd, chan=self._name, name="stop", blocking=True),
            "restart": PyCommand(self._restart_child_cmd, chan=self._name, name="restart", blocking=True),
            "status": PyCommand(self._status_cmd, chan=self._name, name="status", blocking=True),
        }

    def sub_channels(self) -> dict[str, Channel]:
        return self._sub_channels

    def is_connected(self) -> bool:
        return self.is_running()

    async def wait_connected(self) -> None:
        return

    async def on_start_up(self) -> None:
        return

    async def on_close(self) -> None:
        # Stop all running children.
        for name in list(self._sub_channels.keys()):
            await self._stop_child(name)
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

        status_lines = [self._format_status_line(name) for name in self._children]
        desc = self.channel.description()
        if status_lines:
            desc = desc + "\n\n" + "\n".join(status_lines)

        meta = ChannelMeta(
            name=self._name,
            description=desc,
            channel_id=self.id,
            available=True,
            commands=[cmd.meta() for cmd in self._commands.values()],
            children=list(self._children.keys()),
            instructions=[],
            context=[],
        )
        meta.dynamic = True
        return {"": meta}

    def _format_status_line(self, name: str) -> str:
        entry = self._children[name]
        proxy = entry.proxy
        if proxy is None or name not in self._sub_channels:
            return f"- {name}: ❌ 未运行"

        runtime = self.importlib.get_channel_runtime(proxy)
        if runtime is None or not runtime.is_running():
            return f"- {name}: ❌ 未运行"

        conn = proxy.connection
        pid = conn.pid
        start_time = conn.start_time
        runtime_secs = ""
        if start_time is not None:
            runtime_secs = f", 运行时间: {max(0.0, time.time() - start_time):.1f}s"
        pid_part = f"PID: {pid}" if pid is not None else "PID: ?"
        if runtime.is_connected():
            return f"- {name}: ✅ 运行中 ({pid_part}{runtime_secs})"
        return f"- {name}: ⚠️  已启动但未连接 ({pid_part}{runtime_secs})"

    def _ensure_proxy(self, name: str) -> ScriptChannelProxy:
        entry = self._children.get(name)
        if entry is None:
            raise CommandErrorCode.NOT_FOUND.error(f"child channel {name} not registered")
        proxy = build_script_channel(
            entry.target_path,
            name=name,
            description=entry.description,
            env={"PYTHONUNBUFFERED": "1"},
        )
        entry.proxy = proxy
        return proxy

    async def _start_child_cmd(self, name: str, timeout: float = 15.0) -> str:
        if name not in self._children:
            raise CommandErrorCode.NOT_FOUND.error(f"child channel {name} not registered")

        # Always create a fresh proxy instance so importlib compiles a new
        # runtime (channel.id is per-instance).
        proxy = self._ensure_proxy(name)
        self._sub_channels[name] = proxy

        runtime = await self.importlib.get_or_create_channel_runtime(proxy)
        if runtime is None:
            raise CommandErrorCode.NOT_FOUND.error(f"child channel {name} runtime not available")

        try:
            await asyncio.wait_for(runtime.wait_connected(), timeout=timeout)
        except asyncio.TimeoutError:
            raise CommandErrorCode.TIMEOUT.error(f"start channel {name} timeout")
        return ""

    async def _stop_child(self, name: str) -> None:
        entry = self._children.get(name)
        if entry is None or entry.proxy is None:
            return
        runtime = self.importlib.get_channel_runtime(entry.proxy)
        if runtime is not None and runtime.is_running():
            await runtime.close()
        # Unmount from tree so it will not be auto-started.
        self._sub_channels.pop(name, None)

    async def _stop_child_cmd(self, name: str, timeout: float = 5.0) -> str:
        if name not in self._children:
            raise CommandErrorCode.NOT_FOUND.error(f"child channel {name} not registered")
        try:
            await asyncio.wait_for(self._stop_child(name), timeout=timeout)
        except asyncio.TimeoutError:
            raise CommandErrorCode.TIMEOUT.error(f"stop channel {name} timeout")
        return f"Channel {name} stopped."

    async def _restart_child_cmd(self, name: str, timeout: float = 15.0) -> str:
        if name not in self._children:
            raise CommandErrorCode.NOT_FOUND.error(f"child channel {name} not registered")
        await self._stop_child_cmd(name)
        await self._start_child_cmd(name, timeout=timeout)
        return ""

    async def _status_cmd(self) -> str:
        return "\n".join([self._format_status_line(name) for name in self._children])
