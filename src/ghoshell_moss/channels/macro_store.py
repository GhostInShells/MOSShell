"""宏存储 — 文件态与会话态的程序性记忆基座 | 记忆 | beta

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.macro_store import MacroStoreModule
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule(root="./macros"))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import aiofiles
import frontmatter

from ghoshell_moss.core import ChannelCtx, MOSShell
from ghoshell_moss.core.blueprint.states_channel import ChannelModule
from ghoshell_moss.core.concepts.command import Command, PyCommand

__all__ = [
    "MacroStoreModule",
    "MICRO_SUFFIX",
    "FM_NAME",
    "FM_DESCRIPTION",
    "CDATA_START",
    "CDATA_END",
]

MICRO_SUFFIX = ".ctml_micro.md"
"""文件名后缀 — 本模块认识的 CTML 过程文件, 也是文件态与会话态唯一的语法判别特征."""

FM_NAME = "name"
FM_DESCRIPTION = "description"

_RAW_CDATA_START = "<![CDATA["
_RAW_CDATA_END = "]]>"

CDATA_START = "MACRO_CDATA_START"
CDATA_END = "MACRO_CDATA_END"
"""正文里嵌套 CDATA 的占位符.

`macro_save` 的 `text__` 在 CTML 流里已经由 CDATA 包裹, 正文里再出现真实 CDATA 就是
CDATA 套 CDATA, XML 层不成立. 所以模型在正文里写占位符表达嵌套, 占位符同时是存储形态;
展开前还原成真实 CDATA (见 _to_ctml).
"""


def _to_storage(ctml: str) -> str:
    """CTML 形态 → 存储形态: 真实 CDATA 标记归一化成占位符 (幂等)."""
    if _RAW_CDATA_START not in ctml and _RAW_CDATA_END not in ctml:
        return ctml
    return ctml.replace(_RAW_CDATA_START, CDATA_START).replace(_RAW_CDATA_END, CDATA_END)


def _to_ctml(stored: str) -> str:
    """存储形态 → CTML 形态: 占位符还原成真实 CDATA, 供解释器解析."""
    if CDATA_START not in stored and CDATA_END not in stored:
        return stored
    return stored.replace(CDATA_START, _RAW_CDATA_START).replace(CDATA_END, _RAW_CDATA_END)


@dataclass
class _Macro:
    """一条已固化的 CTML 过程 (会话态)."""

    name: str
    description: str
    body: str  # 存储形态


def _dump_micro(name: str, description: str, body: str) -> str:
    meta: dict[str, str] = {FM_NAME: name}
    if description:
        meta[FM_DESCRIPTION] = description
    text = frontmatter.dumps(frontmatter.Post(body, **meta))
    return text if text.endswith("\n") else text + "\n"


def _parse_micro(text: str, path: str) -> _Macro:
    """解析 micro 文件正文. frontmatter 缺 name 视为非法 (不参与目录服务)."""
    post = frontmatter.loads(text)
    name = str(post.metadata.get(FM_NAME) or "").strip()
    description = str(post.metadata.get(FM_DESCRIPTION) or "").strip()
    body = post.content.strip()
    if not name:
        raise ValueError(f"{path}: frontmatter missing required `{FM_NAME}`")
    if not body:
        raise ValueError(f"{path}: empty ctml body")
    return _Macro(name=name, description=description, body=body)


class MacroStoreModule(ChannelModule):
    """程序性记忆模块 — CTML 过程的两态存储.

    - **文件态**: ``root`` 下的 ``*.ctml_micro.md``, frontmatter 自解释, 正文即 CTML.
    - **会话态**: 内存 label, 无 root 时的降级形态.

    ``root`` 只做越权边界, 不做存储布局; root 未配置 (或目录不存在) 时, 文件态命令
    整体不可用 (``macro_load`` / ``micro``), 其余命令退化为纯会话态.
    """

    def __init__(self, root: Path | str | None = None):
        self._root: Path | None = Path(root).expanduser().resolve() if root is not None else None
        self._macros: dict[str, _Macro] = {}
        self._own_commands: dict[str, Command] = {}

    @classmethod
    def new_from_moss_project(cls) -> "MacroStoreModule":
        """从已 seal 的 MOSS 环境发现 project path, 以其为权限边界实例化.

        供 default mode 的 HOST channel 接线; 须在上游已 seal 环境后调用
        (``Environment.discover(bootstrap=False)`` 否则抛 EnvironmentNotSealedError).
        """
        from ghoshell_moss.core.blueprint.environment import Environment

        return cls(root=Environment.discover(bootstrap=False).project_path)

    # -- ChannelModule protocol ----------------------------------------------

    def name(self) -> str:
        return "macro"

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands

    async def on_startup(self) -> None:
        io_available = lambda: self._root is not None and self._root.is_dir()  # noqa: E731
        self._own_commands = {
            "macro": PyCommand(self._macro, name="macro", macro=True),
            "macro_save": PyCommand(self._macro_save, name="macro_save"),
            "macro_load": PyCommand(self._macro_load, name="macro_load", available=io_available),
            "macro_read": PyCommand(self._macro_read, name="macro_read", always_observe=True),
            "macro_forget": PyCommand(self._macro_forget, name="macro_forget"),
            "macro_list": PyCommand(self._macro_list, name="macro_list", always_observe=True),
            "micro": PyCommand(self._micro, name="micro", available=io_available, always_observe=True),
        }

    async def get_instruction(self) -> str:
        if self._root is None:
            return (
                "Macro store — session-scoped CTML procedures (no file root configured).\n"
                "`macro_save` registers a label; `<macro ref=\"x\"/>` expands it in place."
            )
        return (
            f"Macro store — CTML procedures as files and session labels.\n"
            f"Micro files are `*{MICRO_SUFFIX}`: YAML frontmatter with `{FM_NAME}` + `{FM_DESCRIPTION}`,\n"
            f"then the raw CTML body (plain text, no wrapper).\n"
            f"File paths resolve relative to the permission root {self._root} —\n"
            f"absolute paths must stay under it. Session labels come from `macro_save` or `macro_load`.\n"
            f"Nested CDATA inside a body: write {CDATA_START} / {CDATA_END}."
        )

    async def get_named_notices(self) -> dict[str, str]:
        """label 目录作为 named notice — 模型无需 round trip 就知道有哪些宏可用.

        空时返回空串: 渲染层把空片段当静默信号, 不渲染也不宣告变更.
        """
        result = {"macros": self._label_catalog()}
        return result

    # -- commands ------------------------------------------------------------

    async def _macro(self, ref: str, is_file: bool = False) -> str:
        """Invoke a stored CTML procedure; the returned CTML expands at the call site.
        is_file: treat ref as a micro file path under root instead of a session label.
        """
        if is_file:
            return _to_ctml((await self._read_micro(ref)).body)
        macro = self._macros.get(ref)
        if macro is None:
            raise ValueError(f"macro '{ref}' not found")
        return _to_ctml(macro.body)

    async def _macro_save(
            self,
            label: str,
            *,
            text__: str = "",
            description: str = "",
            file: str | None = None,
    ) -> str:
        """Save a CTML procedure as a session label; with `file`, write it as a micro file.
        `text__`: the CTML body (open-close tag body, CDATA-wrapped by CTML syntax).
        """
        body = _to_storage(text__)
        if not body.strip():
            raise ValueError(f"macro '{label}': empty ctml body")
        await self._validate(_to_ctml(body))

        if file is None:
            overwritten = label in self._macros
            self._macros[label] = _Macro(label, description, body)
            verb = "overwritten" if overwritten else "saved"
            return f"macro '{label}' {verb}"

        target = self._resolve_micro_path(file)
        await self._write_micro(target, label, description, body)
        return f"written to {self._relative(target)} (not loaded; use macro_load to register it)"

    async def _macro_load(self, path: str, label: str = "") -> str:
        """Validate a micro file and register it as a session label.
        label: defaults to the file's frontmatter name.
        """
        macro = await self._read_micro(path)
        await self._validate(_to_ctml(macro.body))
        name = label or macro.name
        overwritten = name in self._macros
        self._macros[name] = _Macro(name, macro.description, macro.body)
        verb = "overwritten" if overwritten else "loaded"
        return f"macro '{name}' {verb} from {path}"

    async def _macro_read(self, label: str) -> str:
        """Read a stored macro's raw CTML body without expanding it."""
        macro = self._macros.get(label)
        if macro is None:
            raise ValueError(f"macro '{label}' not found")
        return _to_storage(macro.body)

    async def _macro_forget(self, label: str) -> str:
        """Forget a session label."""
        if label not in self._macros:
            raise ValueError(f"macro '{label}' not found")
        del self._macros[label]
        return f"macro '{label}' forgotten"

    async def _macro_list(self) -> str:
        """List session labels with descriptions."""
        catalog = self._label_catalog()
        return catalog if catalog else "(no macros stored)"

    def _label_catalog(self) -> str:
        """label 目录的文本形态; 空 store 返回空串 (供 named notice 与 list 共用)."""
        lines = [
            f"- {name}: {macro.description}" if macro.description else f"- {name}"
            for name, macro in self._macros.items()
        ]
        return "\n".join(lines)

    async def _micro(self, file: str = ".", recursive: bool = False) -> str:
        """List micro files under a root-relative path, with their frontmatter descriptions."""
        base = self._resolve_micro_path(file)
        if not base.is_dir():
            raise ValueError(f"micro path '{file}' is not a directory under root")
        pattern = f"**/*{MICRO_SUFFIX}" if recursive else f"*{MICRO_SUFFIX}"
        lines = []
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            rel = self._relative(path)
            try:
                macro = _parse_micro(await self._read_text(path), rel)
            except ValueError as e:
                lines.append(f"- {rel} (invalid: {e})")
                continue
            lines.append(f"- {rel}: {macro.description}" if macro.description else f"- {rel}")
        if not lines:
            return f"(no micro files under {self._relative(base)})"
        return "\n".join(lines)

    # -- storage -------------------------------------------------------------

    def _resolve_micro_path(self, path: str) -> Path:
        """越权边界的唯一入口: 解析并保证结果落在 root 内."""
        root = self._root
        if root is None:
            raise ValueError("no macro root configured; file commands are unavailable")
        candidate = Path(path).expanduser()
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if target != root and not target.is_relative_to(root):
            raise ValueError(f"path '{path}' escapes the macro root")
        return target

    def _relative(self, path: Path) -> str:
        root = self._root
        if root is None:
            return path.name
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    async def _read_text(self, path: Path) -> str:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    async def _read_micro(self, path: str) -> _Macro:
        target = self._resolve_micro_path(path)
        if not target.is_file():
            raise ValueError(f"micro file '{path}' not found")
        return _parse_micro(await self._read_text(target), self._relative(target))

    async def _write_micro(self, target: Path, name: str, description: str, body: str) -> None:
        if not target.name.endswith(MICRO_SUFFIX):
            raise ValueError(f"micro file path must end with {MICRO_SUFFIX}: '{self._relative(target)}'")
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(_dump_micro(name, description, body))

    # -- validation ----------------------------------------------------------

    # store 自身的校验命令不得出现在宏正文里 — 它们会递归触发校验, 破坏写/读边界.
    _SAVE_COMMANDS = {"macro_save", "macro_load"}

    async def _validate(self, ctml: str) -> None:
        """真机解析校验 — text → task, 复用真实展开路径.

        dry_run 不展开宏 (run_macro=False), 所以校验只覆盖到宏命令本身; 宏嵌套的
        正确性由真实展开路径的 depth cap 兜底。但 `macro_save` / `macro_load` 会递归
        触发校验, 是 store 的写边界, 不允许出现在正文里。
        """
        shell = ChannelCtx.get_contract(MOSShell)
        current = ChannelCtx.task()
        if current is None:
            raise ValueError("macro validation must run inside a command")
        async for task in shell.parse_text_to_tasks(ctml):
            if task.chan == current.chan and task.meta.name in self._SAVE_COMMANDS:
                raise ValueError(
                    f"macro body must not nest `{task.meta.name}`; "
                    f"save the inner macro separately and invoke it via `macro`"
                )
