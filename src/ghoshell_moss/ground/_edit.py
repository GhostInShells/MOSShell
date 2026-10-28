"""Ground 的编辑面 — 纯文件 / 纯文本操作, 不依赖任何驻留中的场集.

pin 的增删与 GROUND.md 的校验只碰一个文件: 给路径, 改 frontmatter, 返回诊断.
它们既不需要一个 ``GroundSet`` 实例, 也不需要"哪个家"这个坐标系 —— 之前之所以
看起来需要, 是因为 channel 拿一个共享的 ``workspace`` 变量当相对路径基点, 而
那正是多 home 下最隐蔽的一处错锚 (渲染/改写别家的目录时, 锚还在主槽上).

契约上的对应物是 ``GroundSet`` 的 @classmethod (见 contract.py): 声明在契约上
是为了让任何实现都配套提供同一套, 消费方不必知道具体实现如何定位文件。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, dump_l0_pins, load_l0
from ghoshell_moss.ground.contract import PIN_LABEL_MAX_LEN, Pin

__all__ = [
    "find_ancestor_ground",
    "resolve_ground_file",
    "validate_text",
    "upsert_pin",
    "unpin_at",
]

_PIN_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")

_KNOWN_VERBS = {"file", "glob", "frontmatter", "ls", "exec", "law"}

_REQUIRED_ARGS: dict[str, set[str]] = {
    "file": {"path"},
    "glob": {"path"},
    "frontmatter": {"path"},
    "ls": {"path"},
    "exec": {"ref"},
    "law": {"filename"},
}


# -- address ---------------------------------------------------------------


def find_ancestor_ground(start: Path) -> Path | None:
    """从 start 的父目录向上找最近的 GROUND.md, 边界 $HOME."""
    start = start.resolve()
    home = Path(os.environ.get("HOME", "/")).resolve()
    current = start.parent
    while True:
        if (current / DEFAULT_L0_FILENAME).is_file():
            return current
        if current in (home, current.parent):
            return None
        current = current.parent


def resolve_ground_file(ground_file: str | Path) -> Path:
    """把传入的地址收成一份 GROUND.md 的绝对路径.

    编辑面只认绝对路径 —— 多 home 下"相对谁"没有唯一答案, 与其猜一个基点,
    不如让调用方把地址写全 (场帧的 meta 首行就是 ``cd <abs>``, 地址本来就在手上).

    Raises:
        ValueError: 传了相对路径.
    """
    raw = Path(ground_file).expanduser()
    if not raw.is_absolute():
        raise ValueError(
            f"ground file must be an absolute path, got {str(ground_file)!r}"
        )
    if raw.is_dir():
        raw = raw / DEFAULT_L0_FILENAME
    return raw.resolve()


# -- validate --------------------------------------------------------------


def validate_text(text: str) -> str:
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


# -- pins ------------------------------------------------------------------


def upsert_pin(ground_file: str | Path, pin: Pin) -> str:
    """把 pin 增改到指定 GROUND.md (同 label 覆盖). 文件不存在 → 报告, 不创建.

    只重写 frontmatter pins, 不动 body.
    """
    gpath = resolve_ground_file(ground_file)
    if not gpath.is_file():
        return f"[ground] no such GROUND.md: {gpath}"

    try:
        contents = load_l0(gpath.parent, gpath.name)
    except Exception as e:  # noqa: BLE001 — 一份坏文件不该让编辑命令崩掉
        return f"[ground] cannot read {gpath}: {e}"

    pins = [p for p in contents.pins if p.label != pin.label]
    pins.insert(0, pin)
    dump_l0_pins(gpath.parent, pins, gpath.name)
    return f"[ground] pinned {pin.verb}:{pin.label}"


def unpin_at(ground_file: str | Path, label: str) -> str:
    """撤掉指定 GROUND.md 里的一枚 pin. label 不存在 → 报告, 不写盘.

    pin 是场里最重的东西 (它决定每轮看见什么), 只能加不能撤是单向门 ——
    这条是那道门的反向那一半。
    """
    gpath = resolve_ground_file(ground_file)
    if not gpath.is_file():
        return f"[ground] no such GROUND.md: {gpath}"

    try:
        contents = load_l0(gpath.parent, gpath.name)
    except Exception as e:  # noqa: BLE001
        return f"[ground] cannot read {gpath}: {e}"

    remaining = [p for p in contents.pins if p.label != label]
    if len(remaining) == len(contents.pins):
        known = ", ".join(p.label for p in contents.pins) or "none"
        return f"[ground] no pin {label!r} in {gpath} (pins: {known})"

    dump_l0_pins(gpath.parent, remaining, gpath.name)
    return f"[ground] unpinned {label} from {gpath}"
