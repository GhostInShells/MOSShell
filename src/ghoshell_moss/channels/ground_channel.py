"""Ground channel — 认知场的运行时落点 | 集成 | beta

Ground 是 Ghost 的目录级认知场: 一个被 GROUND.md 标记的目录就是场
(frontmatter 身份 + body 法 + pins 注视). 本 channel 持有 GroundSet, 用
``open``/``close`` 把场挂成 virtual child —— 子 channel 无命令, instruction 只放
meta (身份 + pin TOC), 帧 (body + pins 内容) 放 notice (每 refresh 重算, 由
shell trajectory diff 增量重供). 法链进父 channel 的 static, 跨 compact 存活.

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
from pathlib import Path

import yaml

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet
from ghoshell_moss.ground._chain import collect_chain
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


def _resolve_address(value: str | Path, root: Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _find_ancestor_ground(start: Path) -> Path | None:
    """从 start 的父目录向上找最近的 GROUND.md, 边界 $HOME."""
    start = start.resolve()
    home = Path(os.environ.get("HOME", "/")).resolve()
    current = start.parent
    while True:
        if (current / DEFAULT_L0_FILENAME).is_file():
            return current
        if current == home or current == current.parent:
            return None
        current = current.parent


async def _render_dir(dir_path: str | Path, workspace: Path) -> str:
    """渲染一个目录的 ground 帧 — ground-root 或 walk 模式 (无状态 peek, mirror ``moss ground render``).

    只读: 用一次性 DefaultGroundSet, open → render → close; 未 dirty 不触发 sediment.
    """
    path = _resolve_address(dir_path, workspace)
    if not path.is_dir():
        return f"[ground] not a directory: {path}"

    if (path / DEFAULT_L0_FILENAME).is_file():
        async with DefaultGroundSet(workspace_root=workspace) as gs:
            ground = await gs.open(path)
            return str(await ground.render())

    ground_root = _find_ancestor_ground(path)
    if ground_root is None:
        return f"[ground] no GROUND.md from {path} up to $HOME — run `moss ground init`"
    doc_path = ground_root / DEFAULT_L0_FILENAME
    async with DefaultGroundSet(workspace_root=workspace) as gs:
        ground = await gs.open(path, doc=doc_path)
        return str(await ground.render(cwd=path))


async def _meta_dir(dir_path: str | Path, workspace: Path) -> str:
    """无状态 peek 场 meta — mirror ``moss ground meta``."""
    path = _resolve_address(dir_path, workspace)
    if (path / DEFAULT_L0_FILENAME).is_file():
        ground_root = path
    else:
        ground_root = _find_ancestor_ground(path)
        if ground_root is None:
            return f"[ground] no GROUND.md from {path} up to $HOME"
    doc_path = ground_root / DEFAULT_L0_FILENAME
    async with DefaultGroundSet(workspace_root=workspace) as gs:
        ground = await gs.open(path, doc=doc_path)
        return await _ground_meta(ground)


async def _ground_meta(ground: Ground) -> str:
    """场 meta — 身份 (cd / $id) + pin TOC + 法链计数."""
    from ghoshell_moss.ground._render import render_meta

    chain = await ground.chain_text()
    return render_meta(
        root=ground.root,
        doc_path=ground.doc_path,
        chain=chain,
        pins=ground.pins(),
        id_=ground.convention.id,
        label=ground.label,
    )


def _instruction_prose() -> str:
    """Ground 机制一句话 — 不列命令 (命令签名由 interface 自动反射)."""
    return (
        "## Ground (认知场)\n"
        "场 = 一个被 GROUND.md 标记的目录: frontmatter (身份 + pins) + body (法).\n"
        "本场 body (法) 已写入本条 static —— 它跨 compact 存活, 是你在会话中长期保持的 "
        "稳定认知. 场之间不合并 body (每个场渲染自己的根). 用 ``open`` 把一个场挂成子 "
        "channel, 其 meta 在子 instruction, 帧在子 notice (每 refresh diff 重供); pin 内容不预置."
    )


def _upsert_pin(ground_file: str | Path, pin: Pin, workspace: Path) -> str:
    """把 pin 增改到指定 GROUND.md (同 label 覆盖). 文件不存在 → 报告, 不创建.

    同步 IO, 由调用方卸载到线程池. 只重写 frontmatter pins, 不动 body.
    """
    gpath = _resolve_address(ground_file, workspace)
    if gpath.is_dir():
        gpath = gpath / DEFAULT_L0_FILENAME
    if not gpath.is_file():
        return f"[ground] no such GROUND.md: {gpath}"

    contents = load_l0(gpath.parent, gpath.name)
    pins = [p for p in contents.pins if p.label != pin.label]
    pins.insert(0, pin)
    dump_l0_pins(gpath.parent, pins, gpath.name)
    return f"[ground] pinned {pin.verb}:{pin.label}"


def _validate_text(text: str) -> str:
    """校验一份 GROUND.md 文本 → 诊断 (errors + warnings). 不 raise, 不写盘."""
    errors: list[str] = []
    warnings: list[str] = []

    fm_match = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm_match:
        try:
            fm_data = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            errors.append(f"frontmatter YAML: {e}")
            return _fmt_diagnostics(errors, warnings)
    else:
        fm_data = {}
        if text.lstrip().startswith("---"):
            warnings.append(
                "file starts with '---' but no closing '---' found — frontmatter "
                "may be unclosed; all pins will be ignored"
            )

    pins_data = fm_data.get("pins")
    if pins_data is None:
        return _fmt_diagnostics(errors, warnings)

    if not isinstance(pins_data, list):
        errors.append("frontmatter 'pins': must be a list")
        return _fmt_diagnostics(errors, warnings)

    seen_labels: set[str] = set()
    for i, entry in enumerate(pins_data):
        if not isinstance(entry, dict):
            errors.append(f"pin[{i}]: not a mapping")
            continue
        idx = f"pin[{i}]"

        verb = entry.get("verb")
        if not verb:
            errors.append(f"{idx}: missing 'verb' field")
            continue
        if verb not in _KNOWN_VERBS:
            warnings.append(f"{idx}: unknown verb '{verb}' — will be skipped on load")

        label = entry.get("label")
        if not label:
            errors.append(f"{idx}: missing 'label' field")
        elif not isinstance(label, str) or not _PIN_LABEL_RE.match(label):
            errors.append(f"{idx}: label '{label}' does not match [a-zA-Z_][a-zA-Z0-9_-]*")
        elif len(label) > PIN_LABEL_MAX_LEN:
            errors.append(f"{idx}: label '{label}' exceeds {PIN_LABEL_MAX_LEN} char limit")
        else:
            if label in seen_labels:
                warnings.append(f"{idx}: duplicate label '{label}'")
            seen_labels.add(label)

        args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        for field in _REQUIRED_ARGS.get(verb, set()):
            if field not in args:
                errors.append(f"{idx} ({verb}): missing required argument '{field}'")

        if verb == "file" and "range" in args:
            rv = args["range"]
            if isinstance(rv, int):
                if rv < 1:
                    errors.append(f"{idx}: range {rv} must be >= 1 (1-indexed)")
            elif isinstance(rv, str):
                m = re.match(r"^(\d+)(?:-(\d+))?$", rv)
                if m is None:
                    errors.append(f"{idx}: 'range' '{rv}' does not match pattern N or N-M")
                elif m.group(2) and int(m.group(1)) > int(m.group(2)):
                    errors.append(f"{idx}: range start {m.group(1)} > end {m.group(2)}")
            else:
                errors.append(f"{idx}: 'range' must be str or int, got {type(rv).__name__}")

    return _fmt_diagnostics(errors, warnings)


def _fmt_diagnostics(errors: list[str], warnings: list[str]) -> str:
    if not errors and not warnings:
        return "[ground] GROUND.md is valid"
    lines = [f"[WARN] {w}" for w in warnings] + [f"[ERROR] {e}" for e in errors]
    status = "missing" if errors else f"valid ({len(warnings)} warning(s))"
    lines.append(f"[ground] {status}")
    return "\n".join(lines)


def _templates_as_text(dirpath: str | Path | None, workspace: Path) -> str:
    target = _resolve_address(dirpath, workspace) if dirpath else workspace
    gs = DefaultGroundSet(workspace_root=target, logger=logging.getLogger("moss"))
    tmpls = gs.templates()
    if not tmpls:
        return "[ground] no templates found (.grounds/ empty or missing)"
    return "\n".join(f"- {t.name}  ({t.source})  {t.path}" for t in tmpls)


# -- channel builder -------------------------------------------------------


def new_ground_channel(
    groundset: GroundSet,
    *,
    workspace_root: str | Path | None = None,
    open_on_start: list[str | Path] | None = None,
    edit: bool = False,
    name: str = "ground",
    description: str | None = None,
) -> MutableChannel:
    """组装 ground channel — 持有 GroundSet, open/close 挂 virtual children.

    :param groundset: 注入的 GroundSet 实例 (channel 持有, 承载 open/close 生命周期).
    :param workspace_root: 相对路径解析基点. None = cwd.
    :param open_on_start: 启动时自动 open 的场目录列表.
    :param edit: 编辑模式的初始开关. True = pin_*/spec/validate/templates 默认展开,
        False = 折叠 (edit 命令运行时切换).
    :param name: CTML 标签名.
    :param description: 覆盖默认描述.
    """
    workspace = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
    children: dict[str, Channel] = {}
    # 编辑模式闸门: pin_*/spec/validate/templates 折叠在其后, edit 命令开关.
    # command 级 available 在每次 meta refresh 重算, 折叠/展开随 _edit_mode 变化.
    _edit_mode = edit

    if description is None:
        description = (
            "Ground — 认知场: GROUND.md 标记的目录, 法链跨 compact 存活. "
            "open/close 挂场为子 channel, render 无状态 peek, pin_*/spec/validate/templates."
        )

    chan = new_channel(name=name, description=description)

    def _build_child(ground: Ground) -> Channel:
        child = new_channel(name=ground.label, description=f"ground field {ground.label}")

        @child.build.instruction
        async def _child_instruction() -> str:
            return await _ground_meta(ground)

        @child.build.notice
        async def _child_help() -> str:
            return str(await ground.render())

        return child

    async def _open_ground(directory: str | Path, label: str | None, doc: str | None, template: str | None) -> Ground:
        ground = await groundset.open(directory, label=label, doc=doc, template=template)
        children[ground.label] = _build_child(ground)
        return ground

    @chan.build.startup
    async def _startup() -> None:
        for d in (open_on_start or []):
            await _open_ground(d, None, None, None)

    @chan.build.instruction
    async def _instruction() -> str:
        parts = [_instruction_prose()]
        law = await asyncio.to_thread(collect_chain, workspace)
        if law:
            parts.append("### 法\n\n" + law)
        return "\n\n".join(parts)

    @chan.build.virtual_children
    def _virtual_children() -> dict[str, Channel]:
        return dict(children)

    @chan.build.command(name="open", always_observe=True)
    async def open(directory: str, label: str | None = None, doc: str | None = None, template: str | None = None) -> str:
        """打开一个场, 挂成子 channel (meta 进子 instruction, 帧进子 notice).

        :param directory: 场目录 (相对 root 或绝对).
        :param label: 本 channel 内唯一标识. None = 目录 basename.
        :param doc: 显式 GROUND.md 路径 (法锚点). None = directory/GROUND.md.
        :param template: .grounds/ 模板名, 用模板 body + pins 初始化.
        """
        ground = await _open_ground(directory, label, doc, template)
        return f"[ground] opened {ground.label} @ {ground.root}\n\n{await _ground_meta(ground)}"

    @chan.build.command(name="close", always_observe=False)
    async def close(label: str) -> str:
        """关闭一个已打开的场, 撤下其子 channel (dirty 时落盘).

        :param label: 场标识 (见 open 返回值).
        """
        if label not in children:
            return f"[ground] no such open ground: {label}"
        await groundset.close(label)
        children.pop(label, None)
        return f"[ground] closed {label}"

    @chan.build.command(name="render", always_observe=True)
    async def render(directory: str, meta: bool = False) -> str:
        """无状态 peek — 渲染一个目录的 ground 帧 (body + pins 内容), 不挂子 channel.

        :param directory: 目录路径 (相对 root 或绝对). 场根或场内子目录皆可.
        :param meta: 是否在帧前附身份 + pin TOC + 法链计数. 默认 False.
        """
        frame = await _render_dir(directory, workspace)
        if not meta or frame.startswith("[ground]"):
            return frame
        return await _meta_dir(directory, workspace) + "\n\n---\n\n" + frame

    @chan.build.command(name="edit", always_observe=False)
    async def edit(on: bool = True) -> str:
        """开关编辑模式 — 展开/折叠 pin_* / spec / validate / templates 这一组命令.

        :param on: True 展开编辑命令组 (默认), False 折叠回导航视图.
        """
        nonlocal _edit_mode
        _edit_mode = on
        state = "on" if _edit_mode else "off"
        return f"[ground] edit mode {state}"

    @chan.build.command(name="pin_file", always_observe=False, available=lambda: _edit_mode)
    async def pin_file(
        ground_file: str, label: str, path: str,
        line_range: str | None = None, budget: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """在指定 GROUND.md 增改一枚 file pin (单文件注视, 可选行区间).

        :param ground_file: 目标 GROUND.md 文件路径 (或所在目录). 相对 root 或绝对.
        :param label: 唯一标识, 覆盖同 label 旧 pin.
        :param path: 文件路径, 锚点语法允许 ($GROUND|$CWD|$HOME).
        :param line_range: 行区间 'N' 或 'N-M'.
        :param budget: 内容字符上限.
        :param always_show: walk 模式也不折叠, 永远展开.
        :param description: 一句话说明.
        """
        pin = FilePin(
            label=label,
            arguments=FileArguments(path=path, range=line_range, budget=budget),
            always_show=always_show,
            description=description,
        )
        return await asyncio.to_thread(_upsert_pin, ground_file, pin, workspace)

    @chan.build.command(name="pin_glob", always_observe=False, available=lambda: _edit_mode)
    async def pin_glob(
        ground_file: str, label: str, path: str,
        limit: int | None = None, max_depth: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """在指定 GROUND.md 增改一枚 glob pin (匹配路径清单, 不出内容).

        :param ground_file: 目标 GROUND.md 文件路径 (或所在目录).
        :param label: 唯一标识, 覆盖同 label 旧 pin.
        :param path: glob 路径 (*, **, ? 标准语义), 锚点前缀允许.
        :param limit: 命中路径数上限.
        :param max_depth: ** 递归深度上限.
        :param always_show: walk 模式也不折叠, 永远展开.
        :param description: 一句话说明.
        """
        pin = GlobPin(
            label=label,
            arguments=GlobArguments(path=path, limit=limit, max_depth=max_depth),
            always_show=always_show,
            description=description,
        )
        return await asyncio.to_thread(_upsert_pin, ground_file, pin, workspace)

    @chan.build.command(name="pin_frontmatter", always_observe=False, available=lambda: _edit_mode)
    async def pin_frontmatter(
        ground_file: str, label: str, path: str,
        keys: list[str] | None = None, budget: int | None = None,
        limit: int | None = None, max_depth: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """在指定 GROUND.md 增改一枚 frontmatter pin (YAML frontmatter, pattern 可多文件).

        :param ground_file: 目标 GROUND.md 文件路径 (或所在目录).
        :param label: 唯一标识, 覆盖同 label 旧 pin.
        :param path: 文件路径或 glob pattern.
        :param keys: 只提取指定 frontmatter key. None = 全块.
        :param budget: 内容字符上限.
        :param limit: pattern 模式命中文件数上限.
        :param max_depth: 递归发现深度上限.
        :param always_show: walk 模式也不折叠, 永远展开.
        :param description: 一句话说明.
        """
        pin = FrontmatterPin(
            label=label,
            arguments=FrontmatterArguments(path=path, keys=keys, budget=budget, limit=limit, max_depth=max_depth),
            always_show=always_show,
            description=description,
        )
        return await asyncio.to_thread(_upsert_pin, ground_file, pin, workspace)

    @chan.build.command(name="pin_ls", always_observe=False, available=lambda: _edit_mode)
    async def pin_ls(
        ground_file: str, label: str, path: str,
        depth: int = 2, limit: int | None = None, max_depth: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """在指定 GROUND.md 增改一枚 ls pin (目录树结构视图, 不出内容).

        :param ground_file: 目标 GROUND.md 文件路径 (或所在目录).
        :param label: 唯一标识, 覆盖同 label 旧 pin.
        :param path: 目录路径, 锚点语法允许.
        :param depth: 遍历深度. 默认 2.
        :param limit: 目录条目数上限.
        :param max_depth: 递归深度上限 (与 depth 取较小者).
        :param always_show: walk 模式也不折叠, 永远展开.
        :param description: 一句话说明.
        """
        pin = LsPin(
            label=label,
            arguments=LsArguments(path=path, depth=depth, limit=limit, max_depth=max_depth),
            always_show=always_show,
            description=description,
        )
        return await asyncio.to_thread(_upsert_pin, ground_file, pin, workspace)

    @chan.build.command(name="pin_exec", always_observe=False, available=lambda: _edit_mode)
    async def pin_exec(
        ground_file: str, label: str, ref: str,
        mode: str = "shebang",
        timeout: float = 10.0, budget: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """在指定 GROUND.md 增改一枚 exec pin (场根子树内可执行文件).

        :param ground_file: 目标 GROUND.md 文件路径 (或所在目录).
        :param label: 唯一标识, 覆盖同 label 旧 pin.
        :param ref: 场根子树内的可执行文件相对路径. 不允许 ../ 或绝对路径.
        :param mode: 解释器模式 shebang/python/shell. 非 shebang 不要求 +x.
        :param timeout: 秒. 默认 10, 上限 60.
        :param budget: stdout 字符上限.
        :param always_show: walk 模式也不折叠, 永远展开.
        :param description: 一句话说明.
        """
        pin = ExecPin(
            label=label,
            arguments=ExecArguments(ref=ref, mode=mode, timeout=timeout, budget=budget),
            always_show=always_show,
            description=description,
        )
        return await asyncio.to_thread(_upsert_pin, ground_file, pin, workspace)

    @chan.build.command(name="pin_law", always_observe=False, available=lambda: _edit_mode)
    async def pin_law(
        ground_file: str, label: str, filename: str,
        budget: int | None = None, lines: int | None = None,
        always_show: bool = False, description: str = "",
    ) -> str:
        """在指定 GROUND.md 增改一枚 law pin (外来约定文件法链).

        :param ground_file: 目标 GROUND.md 文件路径 (或所在目录).
        :param label: 唯一标识, 覆盖同 label 旧 pin.
        :param filename: 约定文件名 (CLAUDE.md, AGENT.md...). 从 cwd 向上逐层收集.
        :param budget: 总字符上限.
        :param lines: 总行数上限.
        :param always_show: walk 模式也不折叠, 永远展开.
        :param description: 一句话说明.
        """
        pin = LawPin(
            label=label,
            arguments=LawArguments(filename=filename, budget=budget, lines=lines),
            always_show=always_show,
            description=description,
        )
        return await asyncio.to_thread(_upsert_pin, ground_file, pin, workspace)

    @chan.build.command(name="spec", always_observe=True, available=lambda: _edit_mode)
    async def spec() -> str:
        """GROUND.md 格式规范 (SPECIFICATION.md 全文)."""
        import ghoshell_moss.ground

        spec_path = Path(ghoshell_moss.ground.__file__).parent / "SPECIFICATION.md"
        if not spec_path.is_file():
            return "[ground] SPECIFICATION.md not found"
        return spec_path.read_text(encoding="utf-8")

    @chan.build.command(name="validate", always_observe=True, available=lambda: _edit_mode)
    async def validate(filepath: str) -> str:
        """校验一份 GROUND.md 的格式与 pin 定义, 返回诊断结果.

        :param filepath: 目标 GROUND.md 文件路径 (或所在目录).
        """
        gpath = _resolve_address(filepath, workspace)
        if gpath.is_dir():
            gpath = gpath / DEFAULT_L0_FILENAME
        if not gpath.is_file():
            return f"[ground] no such GROUND.md: {gpath}"
        return _validate_text(gpath.read_text(encoding="utf-8"))

    @chan.build.command(name="templates", always_observe=True, available=lambda: _edit_mode)
    async def templates(dirpath: str | None = None) -> str:
        """列出可用的 ground 模板 (.grounds/ 发现) 与发现源.

        :param dirpath: 项目目录 (扫描其 .grounds/). None = 场根.
        """
        return _templates_as_text(dirpath, workspace)

    return chan


def build_ground_channel_factory(
    workspace_root: str | Path,
    *,
    open_on_start: list[str | Path] | None = None,
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
            workspace_root=root,
            open_on_start=open_on_start,
            edit=edit,
            name=name,
            description=description,
        )
    return factory


def build_project_ground_channel(
    *,
    open_on_start: list[str | Path] | None = None,
    edit: bool = False,
    name: str = "ground",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂 — 解析 MOSS 项目根, 默认启动时 open 项目根场.

    Project 解耦收敛在此层: host 侧无感. 法链进 static, 项目根场挂成
    virtual child (meta 在 instruction, 帧在 notice).
    """
    def factory(container: IoCContainer) -> Channel:
        from ghoshell_moss.core.blueprint.project import Project

        project_root = Path(Project.discover().root).resolve()
        on_start = open_on_start if open_on_start is not None else [project_root]
        groundset = DefaultGroundSet(workspace_root=project_root)
        return new_ground_channel(
            groundset,
            workspace_root=project_root,
            open_on_start=on_start,
            edit=edit,
            name=name,
            description=description,
        )
    return factory
