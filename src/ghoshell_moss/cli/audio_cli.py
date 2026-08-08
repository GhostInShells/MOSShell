"""Audio CLI — capability probing for audio capture, playback, TTS, ASR.

CLI 以 cli 身份声明为 matrix node (Matrix.new), 从容器查询音频抽象的注册 provider.
只调 get_provider 反映注册可用性, 不实例化实现 — 不触发重 import, 无副作用.
仅构造容器, 不 join 网络.
"""

from __future__ import annotations

from typing import Type

import typer

from ghoshell_container import INSTANCE

from ghoshell_moss.cli.utils import echo, print_info, print_simple_table
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.asr import ASR
from ghoshell_moss.contracts.speech import Speech, StreamAudioPlayer, TTS

audio_app = typer.Typer(
    help="Audio capability probing — capture, playback, TTS, ASR.",
    no_args_is_help=True,
)

# 核心音频抽象槽位 + 未注册时的原因说明.
_SLOTS: list[tuple[str, Type[INSTANCE], str]] = [
    ("tts", TTS, ""),
    ("speech", Speech, ""),
    ("player", StreamAudioPlayer, ""),
    ("capture", AudioCaptureSource, "in mode HOST layer only — not in project container"),
    ("asr", ASR, "no provider registered"),
]


@audio_app.command("contracts")
def contracts() -> None:
    """List the IoC provider backing each core audio abstraction (no instantiation)."""
    matrix = Matrix.new("audio_cli", category="cli")
    con = matrix.container

    rows = []
    for slot, abstract, note in _SLOTS:
        try:
            provider = con.get_provider(abstract)
        except Exception as e:
            rows.append([slot, abstract.__name__, "—", f"get_provider error: {type(e).__name__} {e}"])
            continue
        if provider is None:
            rows.append([slot, abstract.__name__, "—", note or "no provider registered"])
        else:
            rows.append([slot, abstract.__name__, type(provider).__name__, "OK"])

    print_simple_table(
        data=rows,
        headers=["Slot", "Contract", "Provider", "Status"],
        title="audio contracts",
    )
    echo("")
    print_info(
        "Result is scoped to the active mode (`moss --mode <name>`). "
        "Use `moss manifests providers` for the full provider view."
    )
