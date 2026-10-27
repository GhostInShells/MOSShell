"""DefaultGroundSet — GroundSet ABC 的进程内实现.

每个 GroundSet 实例有独立 label 空间 — 多实例, 非单例.
不同 channel 各自创建自己的 GroundSet, label 冲突天然隔离.

构造期即建 root (锚点场) 并按 root 的 ``groundset`` 字段物化子场 —
GroundSet 本就有启动代价, 子场展开是它的一部分 (一层, 不递归).
materialize=False 可关掉物化 (一次性 peek 场景)。

Template discovery: 扫描 $HOME/.grounds/ → $CWD/.grounds/, 合并为模板清单.
同名模板项目属地优先.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from ghoshell_moss.ground._edit import (
    find_ancestor_ground,
    unpin_at,
    upsert_pin,
    validate_text,
)
from ghoshell_moss.ground._ground import DefaultGround
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, load_l0
from ghoshell_moss.ground._render import ground_meta
from ghoshell_moss.ground.contract import Ground, GroundSet, Pin, TemplateInfo

__all__ = ["DefaultGroundSet"]

_TEMPLATE_DIR = ".grounds"


def _abs_dir(directory: str | Path) -> Path | None:
    """相对路径按 cwd 收成绝对路径; 不是目录返回 None."""
    p = Path(directory).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    return p if p.is_dir() else None


def scan_templates(base: Path) -> list[TemplateInfo]:
    """扫 ``$HOME/.grounds/`` → ``<base>/.grounds/`` 的模板清单, 同名属地优先."""
    seen: dict[str, TemplateInfo] = {}
    home = os.environ.get("HOME")
    if home:
        _collect_templates(Path(home) / _TEMPLATE_DIR, "user", seen)
    _collect_templates(Path(base) / _TEMPLATE_DIR, "project", seen)
    return sorted(seen.values(), key=lambda t: t.name)


def _collect_templates(
    root: Path, source: str, seen: dict[str, TemplateInfo]
) -> None:
    if not root.is_dir():
        return
    for md_file in sorted(root.rglob("*.md")):
        if not md_file.is_file():
            continue
        # 跳过 GROUND.md (实例, 不是模板)
        if md_file.name == DEFAULT_L0_FILENAME:
            continue
        try:
            rel = md_file.relative_to(root)
        except ValueError:
            continue
        name = str(rel.with_suffix("")).replace("\\", "/")
        seen[name] = _read_template_info(name, source, md_file)


def _read_template_info(name: str, source: str, path: Path) -> TemplateInfo:
    description = ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return TemplateInfo(name=name, source=source, path=path)

    import re

    fm_match = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm_match:
        import yaml

        try:
            fm_data = yaml.safe_load(fm_match.group(1)) or {}
            if isinstance(fm_data, dict) and "description" in fm_data:
                description = str(fm_data["description"])
        except yaml.YAMLError:
            pass
    if not description:
        body_match = re.search(r"\S.*", text)
        if body_match:
            description = body_match.group(0)[:120]

    return TemplateInfo(name=name, source=source, path=path, description=description)


class DefaultGroundSet(GroundSet):
    """GroundSet ABC 的默认实现.

    - workspace_root: 相对路径解析基点, 同时是锚点场 (root) 的目录.
    - _active: label → Ground 映射 (含 root).
    - _label_by_path: abspath → label, 幂等 open 的快速查表.
    - _templates: 模板清单, __init__ 时扫描.
    """

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        root_label: str | None = None,
        logger: logging.Logger | None = None,
        materialize: bool = True,
    ) -> None:
        self._workspace_root = (
            workspace_root.resolve() if workspace_root else Path.cwd().resolve()
        )
        self._logger = logger or logging.getLogger("moss")
        self._active: dict[str, Ground] = {}
        self._label_by_path: dict[str, str] = {}
        self._templates: list[TemplateInfo] = []
        self._materialize_errors: list[str] = []
        self._scan_templates()
        # 构造期建 root — 建场只读一个小 GROUND.md, 与模板扫描的同步 IO 同量级.
        # root_label: 给根场一个身份. 缺省是目录名 (单根场景里没有第二个名字可用);
        # 多 home 场景由装配者传 slot key —— 一个 home 只该有一个身份, 目录名是地址.
        self._root = self._construct(self._workspace_root, label=root_label)
        if materialize:
            self._materialize_children()

    # -- root -------------------------------------------------------------

    @property
    def root(self) -> Ground:
        return self._root

    def materialize_errors(self) -> list[str]:
        """物化子场时的可见诊断 (路径不是场 / 打开失败). 消费方用于呈现, 空 = 全部就位."""
        return list(self._materialize_errors)

    # -- open/close -------------------------------------------------------

    async def open(
        self,
        dir: str | Path,
        *,
        label: str | None = None,
        doc: str | Path | None = None,
        template: str | None = None,
        override: bool = False,
    ) -> Ground:
        dir_abs = self._resolve_dir(dir)
        return await asyncio.to_thread(
            self._construct, dir_abs, label=label, doc=doc, template=template, override=override,
        )

    async def close(self, label: str) -> None:
        if label not in self._active:
            raise KeyError(label)
        ground = self._active[label]
        # 只在内存有未落盘变更时写盘 — 只读消费 (render/meta/observe)
        # 永不改写 GROUND.md. root 同样参与 sediment (锚点场的编辑要落盘).
        if ground.dirty:
            await ground.sediment()
        del self._active[label]
        for path_key, mapped in list(self._label_by_path.items()):
            if mapped == label:
                del self._label_by_path[path_key]
                break

    # -- 查询 -------------------------------------------------------------

    def active(self) -> dict[str, Ground]:
        return dict(self._active)

    def get(self, label: str) -> Ground | None:
        return self._active.get(label)

    def templates(self) -> list[TemplateInfo]:
        return list(self._templates)

    # -- 无驻留态操作 (契约 @classmethod 的实现) ---------------------------
    #
    # 这些操作不读也不改 self 的驻留状态, 所以是 classmethod. 它们的价值不在于
    # "能答", 而在于答的时候不必 new 一个锚在别处的临时场集 —— 那是多 home 下
    # 最隐蔽的一处错锚: 渲染 project 的目录, 模板却发现的是 ghost 家的 .grounds.

    @classmethod
    async def peek_dir(
        cls,
        directory: str | Path,
        *,
        template_base: str | Path | None = None,
    ) -> str:
        path = _abs_dir(directory)
        if path is None:
            return f"[ground] not a directory: {directory}"
        base = Path(template_base).resolve() if template_base else path

        async with cls(workspace_root=base, materialize=False) as gs:
            if (path / DEFAULT_L0_FILENAME).is_file():
                ground = await gs.open(path)
                return str(await ground.render())
            ground_root = find_ancestor_ground(path)
            if ground_root is None:
                return f"[ground] no GROUND.md from {path} up to $HOME — run `moss ground init`"
            ground = await gs.open(path, doc=ground_root / DEFAULT_L0_FILENAME)
            return str(await ground.render(cwd=path))

    @classmethod
    async def peek_meta(cls, directory: str | Path) -> str:
        path = _abs_dir(directory)
        if path is None:
            return f"[ground] not a directory: {directory}"
        if (path / DEFAULT_L0_FILENAME).is_file():
            ground_root = path
        else:
            ground_root = find_ancestor_ground(path)
            if ground_root is None:
                return f"[ground] no GROUND.md from {path} up to $HOME"

        async with cls(workspace_root=ground_root, materialize=False) as gs:
            ground = await gs.open(path, doc=ground_root / DEFAULT_L0_FILENAME)
            return await ground_meta(ground)

    @classmethod
    def templates_at(cls, directory: str | Path | None = None) -> list[TemplateInfo]:
        base = Path(directory).resolve() if directory else Path.cwd().resolve()
        return scan_templates(base)

    @classmethod
    def validate_text(cls, text: str) -> str:
        return validate_text(text)

    @classmethod
    def upsert_pin(cls, ground_file: str | Path, pin: Pin) -> str:
        return upsert_pin(ground_file, pin)

    @classmethod
    def unpin_at(cls, ground_file: str | Path, label: str) -> str:
        return unpin_at(ground_file, label)

    # -- 生命周期 ---------------------------------------------------------

    async def __aenter__(self) -> "DefaultGroundSet":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        for label in list(self._active.keys()):
            try:
                await self.close(label)
            except Exception:
                self._logger.exception("failed to close ground %r during __aexit__", label)

    # -- 建场核心 (同步) --------------------------------------------------

    def _resolve_dir(self, dir: str | Path) -> Path:
        p = Path(dir)
        if not p.is_absolute():
            p = self._workspace_root / p
        return p.resolve()

    def _construct(
        self,
        dir_abs: Path,
        *,
        label: str | None = None,
        doc: str | Path | None = None,
        template: str | None = None,
        override: bool = False,
    ) -> Ground:
        """同步建场 + 注册. 同目录幂等 (按 dir.resolve()).

        构造期物化与 async open 共用这一条路径; async 侧只做线程卸载.
        """
        key = str(dir_abs)
        if key in self._label_by_path:
            return self._active[self._label_by_path[key]]

        doc_path = Path(doc).resolve() if doc else dir_abs / DEFAULT_L0_FILENAME

        # template: 找到模板, 复制 body + pins; 找不到显式报错, 不静默降级
        template_body = ""
        template_pins: list = []
        if template is not None:
            tmpl = self._find_template(template)
            if tmpl is None:
                avail = ", ".join(t.name for t in self._templates) or "none"
                raise KeyError(
                    f"template {template!r} not found in .grounds/ (available: {avail})"
                )
            contents = load_l0(tmpl.path.parent, filename=tmpl.path.name)
            template_body = contents.body
            template_pins = contents.pins

        # 从 doc (法锚) 加载 convention — doc 缺省 = dir/GROUND.md,
        # doc≠dir 时法来自别处 (场内移动/便携法单元, SPEC §7.1)
        contents = load_l0(doc_path.parent, doc_path.name)
        convention = contents.convention

        # 模板的 body/pins 与本地 GROUND.md 合并: 本地优先.
        # override=True (预览): 模板定义全权接管, 忽略现有 GROUND.md.
        if override:
            body = template_body
            pins = template_pins
        else:
            body = contents.body or template_body
            pins = contents.pins if contents.pins else template_pins

        # label 分配
        base = label if label else dir_abs.name
        final_label = base
        suffix = 2
        while final_label in self._active:
            final_label = f"{base}-{suffix}"
            suffix += 1

        ground = DefaultGround(
            label=final_label,
            root=dir_abs,
            doc_path=doc_path,
            convention=convention,
            workspace_root=self._workspace_root,
        )
        # 手动注入 body/pins 而不是走 load() — load 会覆盖模板内容
        ground._body = body
        for p in pins:
            ground._pins[p.label] = p
        # 模板注入的 pins 尚未落盘 → 标 dirty, 让 close 触发 sediment.
        # 无模板时 open 只是加载既有 GROUND.md, 保持 clean.
        if template_pins and pins is template_pins:
            ground._dirty = True
        self._active[final_label] = ground
        self._label_by_path[key] = final_label
        return ground

    # -- 子场物化 ---------------------------------------------------------

    def _materialize_children(self) -> None:
        """按 root 的 ``groundset`` 字段展开子场 (一层, 不递归).

        字段值是相对 root 目录 ($GROUND) 的路径. 路径不是场或打开失败 →
        记入 materialize_errors 并告警, 不抛 — 一个坏条目不该拖垮整个场集.
        """
        declared = self._root.convention.groundset
        if not declared:
            return
        root_dir = self._root.root
        for entry in declared:
            rel = str(entry).strip()
            if not rel:
                continue
            child_dir = (root_dir / rel).resolve()
            if child_dir == root_dir:
                self._note_error(f"groundset entry {rel!r} points at the root itself")
                continue
            if not (child_dir / DEFAULT_L0_FILENAME).is_file():
                self._note_error(
                    f"groundset entry {rel!r} is not a ground "
                    f"(no {DEFAULT_L0_FILENAME} under {child_dir})"
                )
                continue
            try:
                self._construct(child_dir)
            except Exception as e:
                self._note_error(f"groundset entry {rel!r} failed to open: {e}")

    def _note_error(self, message: str) -> None:
        self._materialize_errors.append(message)
        self._logger.warning("[ground] %s", message)

    # -- template discovery -----------------------------------------------

    def _scan_templates(self) -> None:
        self._templates = scan_templates(self._workspace_root)

    def _find_template(self, name: str) -> TemplateInfo | None:
        for t in self._templates:
            if t.name == name:
                return t
        return None
