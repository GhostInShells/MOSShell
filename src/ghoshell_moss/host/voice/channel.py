"""Voice control channel — IoC-driven wrapper around VoiceController.

Provides model-governable commands: start/stop listening, mode switching, config, status.
Uses ChannelInterface OO style — constructed from IoC, commands call the controller.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss.core.blueprint.channel_builder import (
    Channel,
    ChannelInterface,
    CommandUtil,
    MutableChannel,
    new_channel,
)

from ghoshell_moss.host.voice.contracts import VoiceConfig, VoiceController, VoiceMode

if TYPE_CHECKING:
    pass

__all__ = ["VoiceChannel"]


class VoiceChannel(ChannelInterface):
    """Model-facing control surface for voice input. Pulls VoiceController from IoC."""

    def __init__(self, controller: VoiceController):
        self._ctrl = controller

    @classmethod
    def new(cls, container: IoCContainer) -> Self:
        ctrl = container.force_fetch(VoiceController)
        return cls(ctrl)

    def as_channel(self) -> MutableChannel:
        chan = new_channel(name="voice", description="语音输入控制 — 开启/关闭聆听、切换模式、配置开关")

        # ── root commands ──

        @chan.build.command()
        async def start() -> str:
            """开启语音聆听，启动采集与 ASR 管线。"""
            await self._ctrl.start()
            return "voice listening started"

        @chan.build.command()
        async def stop() -> str:
            """关闭语音聆听，停止采集与 ASR。"""
            await self._ctrl.stop()
            return "voice listening stopped"

        @chan.build.command(name="status")
        async def _status() -> str:
            """查看当前语音输入运行时状态。"""
            snap = self._ctrl.snapshot()
            return snap.model_dump_json(indent=2)

        # ── mode sub-channel ──

        mode_chan = new_channel(name="mode", description="语音交互模式：ptt / enter / turn_taking / duplex")

        @mode_chan.build.command()
        async def set(name: str) -> str:
            """设置交互模式。name: ptt / enter / turn_taking / duplex / off"""
            try:
                m = VoiceMode(name)
            except ValueError:
                return f"unknown mode: {name}. supported: {[v.value for v in VoiceMode]}"
            await self._ctrl.set_mode(m)
            return f"voice mode set to {m.value}"

        @mode_chan.build.command()
        async def current() -> str:
            """查询当前交互模式。"""
            snap = self._ctrl.snapshot()
            return snap.mode

        chan.import_channels(mode_chan)

        # ── config sub-channel ──

        config_chan = new_channel(name="config", description="语音配置开关 — 10 项正交维度")

        @config_chan.build.command()
        async def show() -> str:
            """查看当前全部配置。"""
            # reload from controller's current config
            snap = self._ctrl.snapshot()
            return snap.model_dump_json(indent=2)

        @config_chan.build.command()
        async def set(key: str, value: str) -> str:
            """修改单个配置项。key: 开关名，value: 新值。例如 config:set key=barge_in value=false"""
            ctrl = CommandUtil.force_get_contract(VoiceController)
            try:
                new_cfg = VoiceConfig(**{key: _coerce(value)})
            except Exception:
                return f"invalid config key or value: {key}={value}"
            await ctrl.set_config(new_cfg)
            return f"config {key} set to {value}"

        chan.import_channels(config_chan)

        return chan


def _coerce(value: str):
    """Coerce string config value to appropriate Python type."""
    v = value.strip().lower()
    if v in ("true", "false"):
        return v == "true"
    if v.isdigit():
        return int(v)
    return value
