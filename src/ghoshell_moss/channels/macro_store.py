"""宏存储 — 程序性记忆基座 | 记忆 | beta

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.macro_store import MacroStoreModule
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule())
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from ghoshell_moss.core.blueprint.states_channel import ChannelModule
from ghoshell_moss.core.concepts.command import Command, PyCommand

__all__ = ["MacroStoreModule"]

_INSTRUCTION = """\
Macro store — save and recall CTML procedures as named macros.

Commands:
  macro(label) — invoke a stored macro. The returned CTML is expanded inline.
    Use this as a shorthand for repeated CTML patterns.
  macro_save(label, description="", text__) — save a CTML procedure.
    Use CDATA to wrap the CTML body:
    <macro_save label="greet">...<![CDATA[<a:say/>]]>...</macro_save>
  macro_read(label) — read the raw CTML of a stored macro without expanding.
  macro_list — list all stored macros with descriptions."""


def _load_store(dir: Path) -> dict[str, dict]:
    index_file = dir / "macros.json"
    if not index_file.exists():
        return {}
    try:
        return json.loads(index_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_store(dir: Path, store: dict[str, dict]) -> None:
    dir.mkdir(parents=True, exist_ok=True)
    index_file = dir / "macros.json"
    tmp = index_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2))
    tmp.replace(index_file)


class MacroStoreModule(ChannelModule):
    """程序性记忆模块 — 将 CTML 序列固化为命名宏，支持存储与召回。

    绑定到 main_channel::

        main.with_module(MacroStoreModule())
        main.with_module(MacroStoreModule(dir=Path("./macros")))
    """

    def __init__(self, dir: Optional[Path] = None):
        self._dir = Path(dir) if dir is not None else None
        self._store: dict[str, dict] = {}
        self._own_commands: dict[str, Command] = {}

    # -- ChannelModule protocol ----------------------------------------------

    def name(self) -> str:
        return "macro"

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands

    async def on_startup(self) -> None:
        if self._dir is not None:
            self._store.update(_load_store(self._dir))
        self._own_commands = {
            "macro": PyCommand(self._macro, name="macro", macro=True),
            "macro_save": PyCommand(self._macro_save, name="macro_save"),
            "macro_read": PyCommand(self._macro_read, name="macro_read", always_observe=True),
            "macro_list": PyCommand(self._macro_list, name="macro_list", always_observe=True),
        }

    async def get_instruction(self) -> str:
        return _INSTRUCTION

    # -- commands ------------------------------------------------------------

    async def _macro(self, label: str) -> str:
        """Invoke a stored macro. Returns CTML that is expanded inline."""
        entry = self._store.get(label)
        if entry is None:
            raise ValueError(f"macro '{label}' not found")
        return entry["ctml"]

    async def _macro_save(self, label: str, description: str = "", *, text__: str) -> str:
        """Save a CTML procedure. text__: CTML content via open-close tag body (use CDATA)."""
        self._store[label] = {"description": description, "ctml": text__}
        self._persist()
        return f"macro '{label}' saved"

    async def _macro_read(self, label: str) -> str:
        """Read the raw CTML of a stored macro without expanding."""
        entry = self._store.get(label)
        if entry is None:
            raise ValueError(f"macro '{label}' not found")
        return entry["ctml"]

    async def _macro_list(self) -> str:
        """List all stored macros with descriptions."""
        if not self._store:
            return "(no macros stored)"
        lines = []
        for label, entry in self._store.items():
            desc = entry.get("description", "")
            lines.append(f"- {label}: {desc}" if desc else f"- {label}")
        return "\n".join(lines)

    # -- persistence ---------------------------------------------------------

    def _persist(self) -> None:
        if self._dir is None:
            return
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _save_store, self._dir, dict(self._store))
