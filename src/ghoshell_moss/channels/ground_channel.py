"""Reflects a GROUND.md cognitive ground into commands and a child-channel tree | integration | beta

Ground 是 Ghost 的目录级认知场: 一个被 GROUND.md 标记的目录就是场
(frontmatter 身份 + body 法 + pins 注视). 本 channel 持有 GroundSet, root 场与
``groundset:`` 字段声明的子场在构造期即物化, 子场挂成 virtual child —— 子 channel
无命令, 身份 (meta) 与帧 (body + pins 内容) 都放 named notice, 由 shell trajectory
逐片段 diff 增量重供. root 的 body 上行到 ghost instruction (render_root 默认
False, 不在这里重复).

**帧是缓存的**: 场帧只在 startup / open / ``refresh`` 三个显式点重渲染, 其余
每一轮 refresh_metas 都只读内存 —— 场里的 exec pin 因此不会随刷新反复起进程.
``refresh`` 同时重读 GROUND.md (法) 与重渲染 (帧), 所以 ghost 自改场文件后调它
即可看到新内容; ``render`` 仍是无状态 peek, 不入缓存.

冷数据 (身份 + pin TOC) 走 notice 而非 instruction: instruction 没有 delta 载体
(shell trajectory 的 diff_facade 只比 states / notice / commands), 放它那里等于
每 epoch 只发一次、之后永不更新.

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.ground_channel import build_project_ground_channel

    main = new_shell_main_channel()
    main.import_channels(build_project_ground_channel())
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import sys
from pathlib import Path

import yaml

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet
from ghoshell_moss.ground._render import ground_meta
from ghoshell_moss.ground._manager import GroundSlot, GroundsManager
from ghoshell_moss.ground._l0 import dump_l0_pins, load_l0
from ghoshell_moss.ground.contract import (
    PIN_LABEL_MAX_LEN,
    ExecArguments,
    ExecPin,
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    Ground,
    GroundSet,
    LawArguments,
    LawPin,
    LsArguments,
    LsPin,
    Pin,
)

__all__ = ["new_ground_channel", "build_ground_channel_factory", "build_project_ground_channel"]

_PIN_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
_KNOWN_VERBS = {"file", "glob", "frontmatter", "ls", "exec", "law"}
_REQUIRED_ARGS = {
    "file": {"path"}, "glob": {"path"}, "frontmatter": {"path"},
    "ls": {"path"}, "exec": {"ref"}, "law": {"filename"},
}


# -- helpers ---------------------------------------------------------------


def _instruction_prose(manager: "GroundsManager | None" = None) -> str:
    """The one-paragraph mechanism note — it explains relations, never lists commands.

    Two shapes: a single rooted set (the old path, one home) and several homes (one ghost
    governing thought / body / environment). The latter describes relations only; content
    lives in each sub-channel's notice.
    """
    if manager is not None:
        lines = [
            "## Ground (cognitive field)",
            "A field is a directory marked by GROUND.md: frontmatter (identity + pins) + body (law).",
            "This ghost governs several homes at once. Each answers a different question, and none "
            "overrides another:",
        ]
        for slot in manager.slots():
            state = "mounted" if slot.mounted else f"absent ({slot.reason})"
            lines.append(f"- `{slot.key}` — {state}")
        lines += [
            "The homes are peers, not a hierarchy: nothing inherits law from a neighbour. A home "
            "without GROUND.md simply has none, and never borrows one from above.",
            "Content lives in each home's own sub-channel (named notice) and is re-rendered only on "
            "startup / open / refresh; every other round reads the cache. `refresh` is what brings "
            "new content; `stats` reports when each field was last rendered and whether its source "
            "changed since.",
            "Heavier read views (render / meta / validate / templates) are deliberately NOT "
            "duplicated here: run them through `ground_run` with `moss ground <verb>` (its cwd is the "
            "home you name), or straight from bash when you have one.",
            "With more than one home mounted, `ground=<label>` is required on every command that "
            "touches a field tree. It is never guessed: a wrong guess would render or rewrite another "
            "home's files and still look like success.",
        ]
        return "\n".join(lines)
    return (
        "## Ground (cognitive field)\n"
        "A field is a directory marked by GROUND.md: frontmatter (identity + pins) + body (law).\n"
        "This channel holds one rooted set: the root field's body is surfaced elsewhere, and only the "
        "fields declared by its ``groundset:`` entry are mounted here as sub-channels. A field's "
        "identity and frame both live in its notice and are re-rendered **only on startup / open / "
        "refresh** (a refresh is what re-reads the law and pays for the frame); every other round "
        "reads the cache and touches no disk. Call `refresh` after editing a field file. Heavier read "
        "views go through `ground_run` with `moss ground <verb>`."
    )


async def _upsert_pin(ground_file: str | Path, pin: Pin) -> str:
    """增改一枚 pin — 纯文件操作, 走契约上的 classmethod, 不借任何驻留场集.

    编辑面只认绝对路径: 多 home 下"相对谁"没有唯一答案, 与其猜一个基点, 不如让
    调用方把地址写全 (场帧的 meta 首行就是 ``cd <abs>``).
    """
    try:
        # 契约上是 classmethod: 走默认实现, 但语义属于 GroundSet 那一层.
        return await asyncio.to_thread(DefaultGroundSet.upsert_pin, ground_file, pin)
    except ValueError as e:
        return f"[ground] {e}"


async def _unpin_at(ground_file: str | Path, label: str) -> str:
    try:
        return await asyncio.to_thread(DefaultGroundSet.unpin_at, ground_file, label)
    except ValueError as e:
        return f"[ground] {e}"


# -- channel builder -------------------------------------------------------


def new_ground_channel(
    groundset: GroundSet | GroundsManager,
    *,
    open_on_start: list[str | Path] | None = None,
    render_root: bool = False,
    instruction: str | None = None,
    edit: bool = False,
    name: str = "ground",
    description: str | None = None,
) -> MutableChannel:
    """组装 ground channel — 持有认知场, 把场挂成 virtual children.

    :param groundset: 一个 ``GroundSet`` (单根, 旧路径), 或一个 ``GroundsManager``
        (若干各自单根的场集, 每个 home 一棵树). manager 模式下: 每个已挂载的 home
        是一个内容子 channel, 该 home ``groundset`` 声明出来的子场挂成它的孙子.
        **命令面仍然只有这一个 channel** —— 树是内容, 不是又一堆控制面.
    :param open_on_start: 启动时额外 open 的场目录列表 (子场声明之外).
    :param render_root: 单根模式下, 是否把 root 场的帧渲染进本 channel 的 notice.
        manager 模式下每层的帧本来就在各自子 channel 里, 此参数无效.
    :param instruction: 覆盖默认 instruction (None = 机制 prose).
    :param edit: 编辑模式的初始开关.
    :param name: CTML 标签名.
    :param description: 覆盖默认描述.
    """
    # 单根兼容层. 复数只住在 manager 里, 不在 GroundSet 里 (D1).
    manager: GroundsManager | None = groundset if isinstance(groundset, GroundsManager) else None
    if manager is not None:
        slots: list[GroundSlot] = manager.mounted()
    else:
        slots = [GroundSlot(key=groundset.root.label, home=groundset.root.root, groundset=groundset)]

    def _set(key: str = "") -> GroundSet:
        """一个 home label → 它的场集.

        只有一个 home 时, 省掉 ``ground`` 是安全的 (没有第二种答案). 有多个 home 时
        省掉它 → 报错并列出全部 label, **不替你挑主槽**: 猜错的事不静默, 否则渲染/
        改写的会是另一个家的文件, 而错误看起来像成功.
        """
        if manager is None:
            if key:
                raise CommandErrorCode.NOT_FOUND.error(
                    f"this ground channel holds a single home; no slot {key!r}"
                )
            return groundset  # type: ignore[return-value]
        mounted = manager.mounted()
        if not key:
            if len(mounted) == 1:
                return mounted[0].groundset  # type: ignore[return-value]
            known = ", ".join(f"{s.key}={s.home}" for s in manager.slots()) or "(none)"
            raise CommandErrorCode.INVALID_USAGE.error(
                "ground= is required when this ghost governs several homes — "
                f"known: {known}"
            )
        try:
            slot = manager.require(key)
        except KeyError as e:
            raise CommandErrorCode.NOT_FOUND.error(str(e)) from None
        if slot.groundset is None:
            raise CommandErrorCode.NOT_AVAILABLE.error(
                f"ground slot {key!r} is not mounted ({slot.reason})"
            )
        return slot.groundset

    def _find_field(label: str, key: str = "") -> Ground | None:
        """按 label 找场. 给了 ground 就只在那一个 home 里找; 否则在所有已挂载 home 里找."""
        if key:
            return _set(key).get(label)
        for slot in slots:
            if slot.groundset is None:
                continue
            found = slot.groundset.get(label)
            if found is not None:
                return found
        return None

    def _field_hint(label: str, key: str = "") -> str:
        """同名场跨 home 时, 把候选指出来 —— 让"没找到"和"找到了好几个"分得开."""
        homes = [
            f"{slot.key}:{label}"
            for slot in slots
            if slot.groundset is not None and slot.groundset.get(label) is not None
        ]
        if len(homes) > 1 and not key:
            return f" — it lives in several homes ({', '.join(homes)}); pass ground=<label>"
        return ""

    children: dict[str, Channel] = {}
    # 场帧缓存 — ground.id -> {"meta", "frame", "at"}. id 而不是 label: 三个 home
    # 各自可能有叫 `skills` 的子场, label 撞名正常, id 不会.
    _cached: dict[str, dict] = {}
    _edit_mode = edit

    if description is None:
        description = (
            "Ground — cognitive fields: directories marked by GROUND.md whose law chain survives "
            "compaction. One content sub-channel per home, one command surface; `refresh` is the "
            "explicit re-render."
        )

    chan = new_channel(name=name, description=description)

    async def _refresh_ground(ground: Ground) -> None:
        """重读法 (GROUND.md) + 重渲染帧. 脏场跳过 load, 保留未落盘改动."""
        import time as _t

        if not ground.dirty:
            await ground.load()
        _cached[ground.id] = {
            "meta": await ground_meta(ground),
            "frame": str(await ground.render()),
            "at": _t.time(),
        }

    def _build_leaf(ground: Ground) -> Channel:
        """一个场的 channel: 只有内容, 没有命令."""
        child = new_channel(name=ground.label, description=f"ground field {ground.label}", uid=ground.id)

        @child.build.named_notices
        async def _leaf_parts() -> dict[str, str | None]:
            cached = _cached.get(ground.id, {})
            return {"meta": cached.get("meta", ""), "frame": cached.get("frame", "")}

        return child

    def _build_slot_tree(slot: GroundSlot) -> Channel:
        """把一个 home 建成一棵子树: 根是它自己的场, 孙子是它 ``groundset`` 声明的子场.

        channel 名用 slot key (``ghost-home`` / ``mode-ground`` / ``project-ground``):
        读的人需要知道的是"这条法属于哪一层", 不是它碰巧住在哪个目录.
        """
        gs = slot.groundset
        grandchildren: dict[str, Channel] = {}
        for ground in gs.active().values():
            if ground is gs.root:
                continue
            grandchildren[ground.label] = _build_leaf(ground)

        node = new_channel(name=slot.key, description=f"ground of {slot.key}", uid=gs.root.id)

        @node.build.named_notices
        async def _slot_parts() -> dict[str, str | None]:
            cached = _cached.get(gs.root.id, {})
            return {"meta": cached.get("meta", ""), "frame": cached.get("frame", "")}

        @node.build.virtual_children
        def _slot_children() -> dict[str, Channel]:
            return dict(grandchildren)

        return node

    async def _open_ground(
        directory: str | Path, label: str | None, doc: str | None, template: str | None, key: str = "",
    ) -> Ground:
        gs = _set(key)
        ground = await gs.open(directory, label=label, doc=doc, template=template)
        if ground is not gs.root:
            children[ground.label] = _build_leaf(ground)
        await _refresh_ground(ground)
        return ground

    @chan.build.startup
    async def _startup() -> None:
        # 每个已挂载的 home 一棵树; 根是 slot key, 孙子是它 set 协议声明的子场.
        for slot in slots:
            children[slot.key] = _build_slot_tree(slot)
            for ground in slot.groundset.active().values():
                await _refresh_ground(ground)
        for d in (open_on_start or []):
            await _open_ground(d, None, None, None)

    @chan.build.instruction
    async def _instruction() -> str:
        if instruction is not None:
            return instruction
        return _instruction_prose(manager)

    @chan.build.notice
    async def _notice() -> str:
        if manager is not None:
            # 缺席是内容: 一行一个 slot, 让"这里没有场"和"没找到"分得开 (D6).
            lines = manager.status_lines()
            for slot in manager.mounted():
                for err in slot.groundset.materialize_errors():
                    lines.append(f"{slot.key}: {err}")
            return "\n".join(lines)
        parts = [f"[ground] {e}" for e in groundset.materialize_errors()]
        if render_root:
            parts.append(_cached.get(groundset.root.id, {}).get("frame", ""))
        return "\n\n".join(parts)

    @chan.build.virtual_children
    def _virtual_children() -> dict[str, Channel]:
        return dict(children)

    @chan.build.command(name="open", always_observe=True)
    async def open(
        directory: str,
        label: str | None = None,
        doc: str | None = None,
        template: str | None = None,
        ground: str = "",
    ) -> str:
        """Open a field and mount it as a sub-channel (identity + frame live in its notice).

        :param directory: The field directory. A relative path resolves against the root of
            the home named by ``ground``.
        :param label: Unique label inside this channel. None = the directory basename.
        :param doc: Explicit GROUND.md path (the law anchor). None = directory/GROUND.md.
        :param template: Template name from .grounds/; its body + pins seed the field.
        :param ground: Which home this belongs to (a slot label from the session notice).
            Required when the ghost governs several homes. Omit it with a single home; if it
            is needed and missing, the call fails and lists every home — it never picks one.
        """
        gs = _set(ground)
        ground_obj = await _open_ground(directory, label, doc, template, ground)
        return (
            f"[ground] opened {ground_obj.label} @ {ground_obj.root}\n\n"
            f"{await ground_meta(ground_obj)}"
        )

    @chan.build.command(name="close", always_observe=False)
    async def close(label: str, ground: str = "") -> str:
        """Close an open field and unmount its sub-channel (dirty fields are written first).

        :param label: The field label (as returned by ``open``).
        :param ground: Which home; required when several are mounted.
        """
        gs = _set(ground)
        found = gs.get(label)
        if label not in children and found is None:
            return f"[ground] no such open ground: {label}"
        await gs.close(label)
        children.pop(label, None)
        if found is not None:
            _cached.pop(found.id, None)
        return f"[ground] closed {label}"

    @chan.build.command(name="refresh", always_observe=False)
    async def refresh(label: str | None = None, ground: str = "") -> str:
        """Re-read a field's law and re-render its frame — an explicit refresh of the field.

        Nothing updates by itself: a refresh is a command, not a clock. After editing a field
        file, this is what makes the change visible.

        :param label: Which field. None = every field in the named home (or in every mounted
            home when ``ground`` is omitted too).
        :param ground: Which home. Naming it removes the ambiguity of a label that exists in
            more than one home.
        """
        if label is None:
            if ground:
                targets = [g for g in _set(ground).active().values()]
            else:
                targets = [g for slot in slots for g in slot.groundset.active().values()]
        else:
            found = _find_field(label, ground)
            if found is None:
                return f"[ground] no such field: {label}{_field_hint(label, ground)}"
            targets = [found]
        for ground_obj in targets:
            await _refresh_ground(ground_obj)
        return f"[ground] refreshed {', '.join(g.label for g in targets)}"

    @chan.build.command(name="stats", always_observe=True)
    async def stats() -> str:
        """When each field was last rendered, and whether its source moved since — cold read.

        Every home is printed with its absolute path: the label says which layer, only the
        path says where the files are. A home's root field has no name of its own — its
        state rides the home's line, because one home carries one identity.

        Times are absolute clock times, not ages: an age changes on every read, which
        turns a cold read hot and implies a freshness that nothing here maintains.
        """
        import time as _t

        def _state(ground) -> str:
            cached = _cached.get(ground.id)
            if not cached:
                return "not rendered yet"
            at = float(cached.get("at", 0) or 0)
            doc = getattr(ground, "doc_path", None) or (ground.root / DEFAULT_L0_FILENAME)
            try:
                moved = bool(Path(doc).is_file() and Path(doc).stat().st_mtime > at)
            except OSError:
                moved = False
            stamp = _t.strftime("%H:%M", _t.localtime(at)) if at else "--:--"
            return f"rendered at {stamp}, source changed: {'yes' if moved else 'no'}"

        if manager is not None:
            homes: list[GroundSlot] = manager.slots()
        else:
            homes = [
                GroundSlot(key=groundset.root.label, home=groundset.root.root, groundset=groundset)
            ]

        lines: list[str] = []
        for slot in homes:
            if slot.groundset is None:
                lines.append(f"{slot.key}: {slot.home} — {slot.status_line()}")
                continue
            root = slot.groundset.root
            lines.append(f"{slot.key}: {slot.home} — {_state(root)}")
            for ground in slot.groundset.active().values():
                if ground is root:
                    continue
                lines.append(f"  {ground.label}: {_state(ground)}")
        return "\n".join(lines) or "[ground] nothing mounted"

    @chan.build.command(name="ground_run", always_observe=True, blocking=False)
    async def ground_run(text__: str, ground: str = "") -> str:
        """Run one ``moss ground`` command line — heavier read views stay in the CLI.

        The working directory is the root of the home named by ``ground``, so relative paths
        point at the right place by construction: ``render .`` renders that home's root field,
        ``templates`` scans that home's own .grounds/.

        :param text__: subcommand + arguments, e.g. ``meta .`` / ``validate .`` /
            ``templates`` / ``verbs`` / ``spec``. Never include ``moss`` or ``ground`` itself.
        :param ground: Which home; required when several are mounted.
        """
        home = _set(ground).root.root
        args = (text__ or "").strip()
        if not args:
            return "[ground] empty command — e.g. `meta .`"
        cmd = [sys.executable, "-m", "ghoshell_moss.cli", "ground", *shlex.split(args)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(home),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
        except Exception as e:  # noqa: BLE001 — 起不来就把原因交回去, 不吞
            return f"[ground] cannot run: {e}"
        text = (out or b"").decode("utf-8", errors="replace").strip()
        errtext = (err or b"").decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return f"[ground] ground {args} → exit {proc.returncode}\n{errtext or text}"
        return text or errtext or f"[ground] ground {args} → no output"

    @chan.build.command(name="edit", always_observe=False)
    async def edit(on: bool = True) -> str:
        """Toggle edit mode — unfolds / folds the pin_* and unpin group.

        :param on: True unfolds the editing group (default), False folds it back to the
            navigation view.
        """
        nonlocal _edit_mode
        _edit_mode = on
        state = "on" if _edit_mode else "off"
        return f"[ground] edit mode {state}"

    @chan.build.command(name="unpin", always_observe=False, available=lambda: _edit_mode)
    async def unpin(ground_file: str, label: str) -> str:
        """Remove one pin from a GROUND.md — pins could be added but never removed before this.

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: The pin label to remove.
        """
        return await _unpin_at(ground_file, label)

    @chan.build.command(name="pin_file", always_observe=False, available=lambda: _edit_mode)
    async def pin_file(
        ground_file: str, label: str, path: str,
        line_range: str | None = None, budget: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """Add or replace a file pin in a GROUND.md (one file watched, optional line range).

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: Unique label; replaces any pin with the same label.
        :param path: The file path; anchor syntax allows ($GROUND|$CWD|$HOME).
        :param line_range: Line range 'N' or 'N-M'.
        :param budget: Cap on rendered characters.
        :param always_show: Keep it expanded even in walk mode.
        :param description: One line on why this is watched.
        """
        pin = FilePin(
            label=label,
            arguments=FileArguments(path=path, range=line_range, budget=budget),
            always_show=always_show,
            description=description,
        )
        return await _upsert_pin(ground_file, pin)

    @chan.build.command(name="pin_glob", always_observe=False, available=lambda: _edit_mode)
    async def pin_glob(
        ground_file: str, label: str, path: str,
        limit: int | None = None, max_depth: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """Add or replace a glob pin in a GROUND.md (a list of matching paths, no content).

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: Unique label; replaces any pin with the same label.
        :param path: The glob (*, **, ? with standard semantics); anchor prefixes allowed.
        :param limit: Cap on matching paths.
        :param max_depth: Cap on ** recursion depth.
        :param always_show: Keep it expanded even in walk mode.
        :param description: One line on why this is watched.
        """
        pin = GlobPin(
            label=label,
            arguments=GlobArguments(path=path, limit=limit, max_depth=max_depth),
            always_show=always_show,
            description=description,
        )
        return await _upsert_pin(ground_file, pin)

    @chan.build.command(name="pin_frontmatter", always_observe=False, available=lambda: _edit_mode)
    async def pin_frontmatter(
        ground_file: str, label: str, path: str,
        keys: list[str] | None = None, budget: int | None = None,
        limit: int | None = None, max_depth: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """Add or replace a frontmatter pin in a GROUND.md (YAML frontmatter, glob-capable).

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: Unique label; replaces any pin with the same label.
        :param path: A file path or a glob pattern.
        :param keys: Only these frontmatter keys. None = the whole block.
        :param budget: Cap on rendered characters.
        :param limit: Cap on matched files in pattern mode.
        :param max_depth: Cap on recursive discovery depth.
        :param always_show: Keep it expanded even in walk mode.
        :param description: One line on why this is watched.
        """
        pin = FrontmatterPin(
            label=label,
            arguments=FrontmatterArguments(path=path, keys=keys, budget=budget, limit=limit, max_depth=max_depth),
            always_show=always_show,
            description=description,
        )
        return await _upsert_pin(ground_file, pin)

    @chan.build.command(name="pin_ls", always_observe=False, available=lambda: _edit_mode)
    async def pin_ls(
        ground_file: str, label: str, path: str,
        depth: int = 2, limit: int | None = None, max_depth: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """Add or replace an ls pin in a GROUND.md (directory structure, no content).

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: Unique label; replaces any pin with the same label.
        :param path: The directory path; anchor syntax allowed.
        :param depth: Traversal depth. Default 2.
        :param limit: Cap on directory entries.
        :param max_depth: Cap on recursion depth (the smaller of the two wins).
        :param always_show: Keep it expanded even in walk mode.
        :param description: One line on why this is watched.
        """
        pin = LsPin(
            label=label,
            arguments=LsArguments(path=path, depth=depth, limit=limit, max_depth=max_depth),
            always_show=always_show,
            description=description,
        )
        return await _upsert_pin(ground_file, pin)

    @chan.build.command(name="pin_exec", always_observe=False, available=lambda: _edit_mode)
    async def pin_exec(
        ground_file: str, label: str, ref: str,
        mode: str = "shebang",
        timeout: float = 10.0, budget: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """Add or replace an exec pin in a GROUND.md (an executable inside the field subtree).

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: Unique label; replaces any pin with the same label.
        :param ref: Path to the executable, relative to the field root. No ../ and no
            absolute paths.
        :param mode: How to run it.
        :param budget: Cap on rendered characters.
        :param timeout: Seconds before the run is abandoned.
        :param always_show: Keep it expanded even in walk mode.
        :param description: One line on why this is watched.
        """
        pin = ExecPin(
            label=label,
            arguments=ExecArguments(ref=ref, mode=mode, timeout=timeout, budget=budget),
            always_show=always_show,
            description=description,
        )
        return await _upsert_pin(ground_file, pin)

    @chan.build.command(name="pin_law", always_observe=False, available=lambda: _edit_mode)
    async def pin_law(
        ground_file: str, label: str, filename: str,
        budget: int | None = None, lines: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """Add or replace a law pin in a GROUND.md (a law file the field stands on).

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: Unique label; replaces any pin with the same label.
        :param filename: The law file, relative to the field root.
        :param budget: Cap on rendered characters.
        :param lines: Cap on rendered lines.
        :param always_show: Keep it expanded even in walk mode.
        :param description: One line on why this is watched.
        """
        pin = LawPin(
            label=label,
            arguments=LawArguments(filename=filename, budget=budget, lines=lines),
            always_show=always_show,
            description=description,
        )
        return await _upsert_pin(ground_file, pin)

    return chan


def build_ground_channel_factory(
    workspace_root: str | Path,
    *,
    open_on_start: list[str | Path] | None = None,
    render_root: bool = False,
    instruction: str | None = None,
    edit: bool = False,
    name: str = "ground",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂 — 建 DefaultGroundSet + ground channel."""
    def factory(container: IoCContainer) -> Channel:
        root = Path(workspace_root).resolve()
        groundset = DefaultGroundSet(workspace_root=root)
        return new_ground_channel(
            groundset,
            open_on_start=open_on_start,
            render_root=render_root,
            instruction=instruction,
            edit=edit,
            name=name,
            description=description,
        )
    return factory


def build_project_ground_channel(
    *,
    open_on_start: list[str | Path] | None = None,
    render_root: bool = False,
    instruction: str | None = None,
    edit: bool = False,
    name: str = "ground",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂 — 解析 MOSS 项目根, 建 root + 物化子场.

    Project 解耦收敛在此层: host 侧无感. root 场 (项目根) 由 GroundSet
    构造期建立, ``groundset:`` 声明的子场挂成 virtual child (meta 在
    instruction, 帧在 notice).
    """
    def factory(container: IoCContainer) -> Channel:
        from ghoshell_moss.core.blueprint.project import Project

        project_root = Path(Project.discover().root).resolve()
        groundset = DefaultGroundSet(workspace_root=project_root)
        return new_ground_channel(
            groundset,
            open_on_start=open_on_start,
            render_root=render_root,
            instruction=instruction,
            edit=edit,
            name=name,
            description=description,
        )
    return factory
