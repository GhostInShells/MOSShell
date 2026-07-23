"""DefaultGroundSet — GroundSet ABC 的进程内实现.

每个 GroundSet 实例有独立 label 空间 — 多实例, 非单例.
不同 channel 各自创建自己的 GroundSet, label 冲突天然隔离.

Template discovery (K63): 扫描 $CWD/.grounds/ → $HOME/.grounds/ →
ghost 携带路径, 合并为模板清单. 同名模板项目属地优先.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ghoshell_moss.ground._ground import DefaultGround
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, load_l0
from ghoshell_moss.ground.contract import Ground, GroundSet, TemplateInfo

__all__ = ["DefaultGroundSet"]

_TEMPLATE_DIR = ".grounds"


class DefaultGroundSet(GroundSet):
    """GroundSet ABC 的默认实现.

    - workspace_root: 相对路径解析基点.
    - _active: label → Ground 映射.
    - _label_by_path: abspath → label, 幂等 open 的快速查表.
    - _templates: 模板清单, __init__ 时扫描.
    """

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        ghost_templates_dir: Path | None = None,
    ) -> None:
        self._workspace_root = (
            workspace_root.resolve() if workspace_root else Path.cwd().resolve()
        )
        self._active: dict[str, Ground] = {}
        self._label_by_path: dict[str, str] = {}
        self._templates: list[TemplateInfo] = []
        self._scan_templates(ghost_templates_dir)

    # -- open/close -------------------------------------------------------

    async def open(
        self,
        dir: str | Path,
        *,
        label: str | None = None,
        doc: str | Path | None = None,
        template: str | None = None,
    ) -> Ground:
        dir_path = Path(dir)
        if not dir_path.is_absolute():
            dir_path = self._workspace_root / dir_path
        dir_abs = dir_path.resolve()

        # 幂等
        key = str(dir_abs)
        if key in self._label_by_path:
            return self._active[self._label_by_path[key]]

        # doc 路径
        doc_path = Path(doc).resolve() if doc else dir_abs / DEFAULT_L0_FILENAME

        # template: 找到模板, 复制 body + pins
        template_body = ""
        template_pins: list = []
        if template is not None:
            tmpl = self._find_template(template)
            if tmpl is not None:
                contents = load_l0(tmpl.path.parent, filename=tmpl.path.name)
                template_body = contents.body
                template_pins = contents.pins

        # 从 GROUND.md 加载 convention (没有 GROUND.md 时为空)
        contents = await asyncio.to_thread(load_l0, dir_abs)
        convention = contents.convention

        # 模板的 body/pins 与本地 GROUND.md 合并: 本地优先
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
        self._active[final_label] = ground
        self._label_by_path[key] = final_label
        return ground

    async def close(self, label: str) -> None:
        if label not in self._active:
            raise KeyError(label)
        ground = self._active[label]
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

    # -- 生命周期 ---------------------------------------------------------

    async def __aenter__(self) -> "DefaultGroundSet":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        for label in list(self._active.keys()):
            try:
                await self.close(label)
            except Exception:
                pass

    # -- template discovery -----------------------------------------------

    def _scan_templates(self, ghost_templates_dir: Path | None) -> None:
        seen: dict[str, TemplateInfo] = {}

        # 1. $HOME/.grounds/ — 最低优先级
        home = os.environ.get("HOME")
        if home:
            self._collect_templates(Path(home) / _TEMPLATE_DIR, "user", seen)

        # 2. $CWD/.grounds/ — 项目属地, 覆盖 user
        self._collect_templates(
            self._workspace_root / _TEMPLATE_DIR, "project", seen
        )

        # 3. ghost 携带 — 最高优先级
        if ghost_templates_dir is not None:
            self._collect_templates(ghost_templates_dir, "ghost", seen)

        self._templates = sorted(seen.values(), key=lambda t: t.name)

    def _collect_templates(
        self, root: Path, source: str, seen: dict[str, TemplateInfo]
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
            info = self._read_template_info(name, source, md_file)
            seen[name] = info  # 同名覆盖 = 高优先级 source 胜

    @staticmethod
    def _read_template_info(
        name: str, source: str, path: Path
    ) -> TemplateInfo:
        description = ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return TemplateInfo(name=name, source=source, path=path)

        # 尝试从前 frontmatter 或 body 首行提取 description
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

        return TemplateInfo(
            name=name, source=source, path=path, description=description
        )

    def _find_template(self, name: str) -> TemplateInfo | None:
        for t in self._templates:
            if t.name == name:
                return t
        return None
