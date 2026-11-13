"""Reflects a GROUND.md cognitive ground into commands and a child-channel tree | integration | beta

Ground 是 Ghost 的目录级认知场: 一个被 GROUND.md 标记的目录就是场
(frontmatter 身份 + body 法 + pins 注视).

**一个 home 一个 channel**. 每个实例独占一个 ``GroundSet``: 它的 root 是身份, 它
``inline:`` 声明的子场是内容. 多 home 不在这一个 channel 里消歧 —— 每个 home 各
挂一个实例, 路径本身就是命名空间, 没有任何 ``ground=`` 参数可供猜错.

**场树按需披露**. 子场默认关闭 (``gate``), 目录经 notice 交付, 只有 ``mount_child``
过的子场才走进 facade —— 一个 ghost 有十几处场时, 常驻的只有目录那几行. 高阶配置
(open/close, stats, pins, CLI) 收在 ``admin`` 子通道里, 不读不出现.

**帧是缓存的**: 场帧只在 startup / open / ``refresh`` 三个显式点重渲染, 其余
每一轮 refresh_metas 都只读内存 —— 场里的 exec pin 因此不会随刷新反复起进程.
``refresh`` 同时重读 GROUND.md (法) 与重渲染 (帧), 所以 ghost 自改场文件后调它
即可看到新内容.

**open 的边界就是这个 home**: 越界的目录直接拒绝. 两个 set 各持一份同目录的 Ground
会立刻分叉 (一边 edit 另一边看不见, close 也各关各的), 所以边界划在 channel 里,
而不是留给调用方自觉.

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
from collections.abc import Callable
from pathlib import Path

import yaml

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet
from ghoshell_moss.ground._render import ground_meta
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
    ContentArguments,
    ContentPin,
    LsArguments,
    LsPin,
    Pin,
)

__all__ = ["new_ground_channel", "build_ground_channel_factory", "build_project_ground_channel"]

_PIN_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
_KNOWN_VERBS = {"file", "glob", "frontmatter", "ls", "exec", "content"}
_REQUIRED_ARGS = {
    "file": {"path"}, "glob": {"path"}, "frontmatter": {"path"},
    "ls": {"path"}, "exec": {"ref"}, "content": {"filename"},
}


# -- helpers ---------------------------------------------------------------


def _instruction_prose() -> str:
    """The one-paragraph mechanism note — it explains relations, never lists commands.

    The command signatures are the declaration: repeating them here costs tokens and drifts.
    """
    return (
        "## Ground (cognitive field)\n"
        "A field is a directory marked by GROUND.md: identity + pins in the frontmatter, content in "
        "the body. Fields are mounted on demand — mount a child to read its frame, and call "
        "`refresh` after editing a file. Administration (open/close, stats, pin editing, the "
        "`moss ground` CLI) lives under the `admin` child."
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
    inline: GroundSet,
    *,
    open_on_start: list[str | Path] | None = None,
    render_root: bool = False,
    instruction: str | None = None,
    edit: bool = False,
    gate: bool = True,
    root_frame: str | None = None,
    on_root_refresh: Callable[[str], None] | None = None,
    name: str = "ground",
    description: str | None = None,
) -> MutableChannel:
    """组装 ground channel — 一个 GroundSet 实例, 一棵场树, 一个命令面.

    一个 home 一个 channel. 多 home 不在这里消歧: 每个 home 各挂一个实例, 路径就是
    命名空间; 没有任何一个实例需要 ``ground=`` 这种参数, 猜错另一家的风险因此不存在.

    :param inline: 本 channel 独占的场集, 同时是 **边界** —— ``open`` 只接受落在
        ``inline.root.root`` 之内的目录, 跨家的场由另一个 channel 持有.
    :param open_on_start: 启动时额外 open 的场目录 (子场声明之外).
    :param render_root: 是否把 root 场的帧渲染进本 channel 的 notice. 内容另有去处的
        home (ghost home 走 instruction) 设 False, 不在两处各放一份.
    :param instruction: 覆盖默认 instruction (None = 机制 prose). 次要的 ground channel
        传 ``""`` 即可整段清掉.
    :param edit: 编辑模式的初始开关.
    :param gate: 子场默认关闭, 由 ``mount_child`` 逐一披露 —— 目录随挂载变化重发,
        未挂的子场连 instruction/notice/interface 都不进 facade.
    :param root_frame: 已经渲染好的 root 帧. 传入则 startup 复用它, 不重渲染
        (exec pin 不该在开机时跑两遍).
    :param on_root_refresh: root 被 ``refresh`` 后拿到新帧的回调 —— 让 instruction 的
        持有者同步更新.
    :param name: CTML 标签名.
    :param description: 覆盖默认描述.
    """
    root = inline.root
    home = root.root.resolve()

    # 场帧缓存 — ground.id -> {"meta", "frame", "at"}. 用 id 不用 label.
    children: dict[str, Channel] = {}
    _cached: dict[str, dict] = {}
    _edit_mode = edit

    if description is None:
        description = (
            "Ground — a cognitive field: a directory marked by GROUND.md whose content survives "
            "compaction. Fields are mounted on demand."
        )

    chan = new_prime_channel(name, description=description, gate=gate)

    async def _refresh_ground(ground: Ground) -> None:
        """重读法 (GROUND.md) + 重渲染帧. 脏场跳过 load, 保留未落盘改动."""
        import time as _t

        if not ground.dirty:
            await ground.load()
        frame = str(await ground.render())
        _cached[ground.id] = {
            "meta": await ground_meta(ground),
            "frame": frame,
            "at": _t.time(),
        }
        if ground is root and on_root_refresh is not None:
            on_root_refresh(frame)

    def _build_leaf(ground: Ground) -> Channel:
        """一个场的 channel: 只有内容 (meta + frame 走 named notice), 没有命令."""
        desc = f"ground field {ground.label}"
        try:
            if ground.convention.description:
                desc = f"{ground.label} — {ground.convention.description}"
        except Exception:  # noqa: BLE001 — 描述缺失不该挡住建场
            pass
        child = new_channel(name=ground.label, description=desc, uid=ground.id)

        @child.build.named_notices
        async def _leaf_parts() -> dict[str, str | None]:
            cached = _cached.get(ground.id, {})
            return {"meta": cached.get("meta", ""), "frame": cached.get("frame", "")}

        return child

    def _inside(directory: str | Path, *, field: str = "directory") -> Path:
        """把路径解析到 home 之内 —— 越界就报错, 不静默开一份指向别处的场.

        两个 set 各持一份同目录的 Ground 会立刻分叉: 一边 edit 另一边看不见, close
        也各关各的. 边界在这里划, 比在调用方约定可靠.
        """
        p = Path(directory)
        if not p.is_absolute():
            p = home / p
        p = p.resolve()
        if not p.is_relative_to(home):
            raise CommandErrorCode.INVALID_USAGE.error(
                f"this ground channel is rooted at {home}; {field} {p} is outside it — "
                f"open it from the channel that owns that home"
            )
        return p

    async def _open_ground(
        directory: str | Path,
        label: str | None,
        doc: str | None,
        template: str | None,
    ) -> Ground:
        if (label or Path(directory).name) == "admin":
            raise CommandErrorCode.INVALID_USAGE.error(
                "'admin' is a reserved label of this channel"
            )
        dir_abs = _inside(directory)
        doc_abs: str | None = None
        if doc is not None:
            doc_abs = str(_inside(doc, field="doc"))
        ground = await inline.open(dir_abs, label=label, doc=doc_abs, template=template)
        if ground is not root:
            children[ground.label] = _build_leaf(ground)
        await _refresh_ground(ground)
        return ground

    # -- admin 子通道: 高阶配置与落盘动作, gate 打开时不读不出现 -------------
    admin = new_channel(
        name="admin",
        description=(
            "Ground administration — open/close fields, stats, pin editing, and the "
            "moss CLI escape. Not part of the reading surface."
        ),
    )

    @chan.build.startup
    async def _startup() -> None:
        for ground in inline.active().values():
            if ground is root:
                if root_frame is not None:
                    import time as _t

                    _cached[root.id] = {
                        "meta": await ground_meta(root),
                        "frame": root_frame,
                        "at": _t.time(),
                    }
                else:
                    await _refresh_ground(root)
                continue
            children[ground.label] = _build_leaf(ground)
            await _refresh_ground(ground)
        for d in (open_on_start or []):
            await _open_ground(d, None, None, None)

    @chan.build.instruction
    async def _instruction() -> str:
        if instruction is not None:
            return instruction
        return _instruction_prose()

    @chan.build.notice
    async def _notice() -> str:
        parts = [f"[ground] {e}" for e in inline.materialize_errors()]
        if render_root:
            parts.append(_cached.get(root.id, {}).get("frame", ""))
        return "\n\n".join(parts)

    @chan.build.virtual_children
    def _virtual_children() -> dict[str, Channel]:
        result: dict[str, Channel] = {"admin": admin}
        result.update(children)
        return result

    @chan.build.command(name="refresh", always_observe=False)
    async def refresh(label: str | None = None) -> str:
        """Re-read a field's content and re-render its frame — an explicit refresh of the field.

        Nothing updates by itself: a refresh is a command, not a clock. After editing a field
        file, this is what makes the change visible.

        :param label: Which field. None = every field in this home.
        """
        if label is None:
            targets = list(inline.active().values())
        else:
            found = inline.get(label)
            if found is None:
                return f"[ground] no such field: {label}"
            targets = [found]
        for ground_obj in targets:
            await _refresh_ground(ground_obj)
        return f"[ground] refreshed {', '.join(g.label for g in targets)}"

    # -- admin: 落盘与诊断 --------------------------------------------------

    @admin.build.command(name="open", always_observe=True)
    async def open(
        directory: str,
        label: str | None = None,
        doc: str | None = None,
        template: str | None = None,
    ) -> str:
        """Open a field and return its identity + frame.

        The frame is returned here, so a one-shot look needs no mounting. The field also joins
        this channel's child catalog; mount its label to keep it on your surface.

        :param directory: The field directory, inside this channel's home. A relative path
            resolves there; a path outside is refused.
        :param label: Unique label inside this channel. None = the directory basename.
        :param doc: Explicit GROUND.md path (the content anchor). None = directory/GROUND.md.
        :param template: Template name from .grounds/; its body + pins seed the field.
        """
        ground_obj = await _open_ground(directory, label, doc, template)
        cached = _cached.get(ground_obj.id, {})
        return (
            f"[ground] opened {ground_obj.label} @ {ground_obj.root}\n\n"
            f"{cached.get('meta', '')}\n\n{cached.get('frame', '')}"
        )

    @admin.build.command(name="close", always_observe=False)
    async def close(label: str) -> str:
        """Close an open field and unmount its sub-channel (dirty fields are written first).

        :param label: The field label (as returned by ``open``).
        """
        found = inline.get(label)
        if found is None:
            return f"[ground] no such open ground: {label}"
        await inline.close(label)
        children.pop(label, None)
        _cached.pop(found.id, None)
        return f"[ground] closed {label}"

    @admin.build.command(name="stats", always_observe=True)
    async def stats() -> str:
        """When each field was last rendered, and whether its source moved since — cold read.

        The home is printed with its absolute path: this channel's label says which layer,
        only the path says where the files are. Times are absolute clock times, not ages.

        The channel's catalog notice already describes the fields; this answers what is stale.
        """
        import time as _t

        def _state(ground: Ground) -> str:
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

        lines = [f"{home} — {_state(root)}"]
        for ground in inline.active().values():
            if ground is root:
                continue
            lines.append(f"  {ground.label}: {_state(ground)}")
        return "\n".join(lines)

    @admin.build.command(name="ground_run", always_observe=True, blocking=False)
    async def ground_run(text__: str) -> str:
        """Run one ``moss ground`` command line — heavier read views stay in the CLI.

        The working directory is this home's root, so relative paths point at the right place
        by construction: ``render .`` renders the root field, ``templates`` scans this home's
        own .grounds/.

        :param text__: subcommand + arguments, e.g. ``meta .`` / ``validate .`` /
            ``templates`` / ``verbs`` / ``spec``. Never include ``moss`` or ``ground`` itself.
        """
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

    @admin.build.command(name="edit", always_observe=False)
    async def edit(on: bool = True) -> str:
        """Toggle edit mode — unfolds / folds the pin_* and unpin group.

        :param on: True unfolds the editing group (default), False folds it back to the
            navigation view.
        """
        nonlocal _edit_mode
        _edit_mode = on
        state = "on" if _edit_mode else "off"
        return f"[ground] edit mode {state}"

    @admin.build.command(name="unpin", always_observe=False, available=lambda: _edit_mode)
    async def unpin(ground_file: str, label: str) -> str:
        """Remove one pin from a GROUND.md.

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: The pin label to remove.
        """
        return await _unpin_at(ground_file, label)

    @admin.build.command(name="pin_file", always_observe=False, available=lambda: _edit_mode)
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

    @admin.build.command(name="pin_glob", always_observe=False, available=lambda: _edit_mode)
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

    @admin.build.command(name="pin_frontmatter", always_observe=False, available=lambda: _edit_mode)
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

    @admin.build.command(name="pin_ls", always_observe=False, available=lambda: _edit_mode)
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

    @admin.build.command(name="pin_exec", always_observe=False, available=lambda: _edit_mode)
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

    @admin.build.command(name="pin_law", always_observe=False, available=lambda: _edit_mode)
    async def pin_law(
        ground_file: str, label: str, filename: str,
        budget: int | None = None, lines: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """Add or replace a content pin in a GROUND.md (a content file the field stands on).

        :param ground_file: Absolute path of the target GROUND.md (or its directory).
        :param label: Unique label; replaces any pin with the same label.
        :param filename: The content file, relative to the field root.
        :param budget: Cap on rendered characters.
        :param lines: Cap on rendered lines.
        :param always_show: Keep it expanded even in walk mode.
        :param description: One line on why this is watched.
        """
        pin = ContentPin(
            label=label,
            arguments=ContentArguments(filename=filename, budget=budget, lines=lines),
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
    gate: bool = True,
    name: str = "ground",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂 — 建 DefaultGroundSet + ground channel."""
    def factory(container: IoCContainer) -> Channel:
        root = Path(workspace_root).resolve()
        inline = DefaultGroundSet(workspace_root=root)
        return new_ground_channel(
            inline,
            open_on_start=open_on_start,
            render_root=render_root,
            instruction=instruction,
            edit=edit,
            gate=gate,
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
    gate: bool = True,
    name: str = "ground",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂 — 解析 MOSS 项目根, 建 root + 物化子场."""
    def factory(container: IoCContainer) -> Channel:
        from ghoshell_moss.core.blueprint.project import Project

        project_root = Path(Project.discover().root).resolve()
        inline = DefaultGroundSet(workspace_root=project_root)
        return new_ground_channel(
            inline,
            open_on_start=open_on_start,
            render_root=render_root,
            instruction=instruction,
            edit=edit,
            gate=gate,
            name=name,
            description=description,
        )
    return factory
