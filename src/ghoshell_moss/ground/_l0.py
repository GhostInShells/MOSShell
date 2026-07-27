"""L0 file — GROUND.md 读写.

结构 (SPEC §2 revised):
    ---
    <frontmatter YAML — $id, label, pins, ... >
    ---
    <body markdown — 纯粹的人/模型叙事, 无机器段>

pins 是 frontmatter 的一部分, 不是独立的 markdown section.
seen_* (PinShadow) 不进盘 — SPEC §7.2 的铁律.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ghoshell_moss.ground.contract import (
    _VERB_CLASSES,
    ExecArguments,
    FileArguments,
    FrontmatterArguments,
    GlobArguments,
    GroundConvention,
    LsArguments,
    Pin,
)

__all__ = [
    "DEFAULT_L0_FILENAME",
    "L0Contents",
    "load_l0",
    "dump_l0_pins",
]

DEFAULT_L0_FILENAME = "GROUND.md"

# -- regex ----------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---[ \t]*\n(?P<body>.*)",
    re.DOTALL,
)

# K55: arguments models per verb — used for deserialization
_ARG_CLASSES: dict[str, type] = {
    "file": FileArguments,
    "glob": GlobArguments,
    "frontmatter": FrontmatterArguments,
    "ls": LsArguments,
    "exec": ExecArguments,
}


# -- L0Contents -----------------------------------------------------------


@dataclass
class L0Contents:
    """一份 GROUND.md 解析后的视图."""

    convention: GroundConvention
    body: str
    pins: list[Pin]

    @classmethod
    def empty(cls) -> "L0Contents":
        return cls(convention=GroundConvention(), body="", pins=[])


# -- load -----------------------------------------------------------------


def load_l0(root: Path, filename: str = DEFAULT_L0_FILENAME) -> L0Contents:
    """从场 root 加载 GROUND.md. 文件不存在 → empty().

    Raises:
        yaml.YAMLError: YAML 语法错误.
        pydantic.ValidationError: frontmatter schema 不匹配.
    """
    path = root / filename
    if not path.is_file():
        return L0Contents.empty()

    text = path.read_text(encoding="utf-8")

    # frontmatter — pins 是其中的一个 key
    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match is not None:
        fm_data = yaml.safe_load(fm_match.group("yaml")) or {}
        raw_pins = fm_data.pop("pins", None) or []
        convention = GroundConvention(**fm_data)
        body = fm_match.group("body")
    else:
        convention = GroundConvention()
        body = text
        raw_pins = []

    pins = _deserialize_pins(raw_pins)
    return L0Contents(convention=convention, body=body, pins=pins)


def _deserialize_pins(raw: list[dict]) -> list[Pin]:
    """K55 envelope → Pin subclass dispatch.

    Each item: {verb, label, arguments: {...}, description?}.
    Unknown verbs are skipped (SPEC §4.2).
    """
    result: list[Pin] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        verb = item.get("verb", "")
        cls = _VERB_CLASSES.get(verb)
        if cls is None:
            continue

        args_data = item.get("arguments") or {}
        args_cls = _ARG_CLASSES.get(verb)
        if args_cls is not None:
            arguments = args_cls(**args_data)
        else:
            arguments = args_data

        result.append(cls(
            label=item["label"],
            arguments=arguments,
            description=item.get("description", ""),
        ))
    return result


# -- dump -----------------------------------------------------------------


def dump_l0_pins(
    root: Path,
    pins: list[Pin],
    filename: str = DEFAULT_L0_FILENAME,
    *,
    body: str | None = None,
) -> None:
    """把 pin 集写回 GROUND.md 的 frontmatter ``pins`` key.

    - 文件不存在: 创建, 写 frontmatter + pins + optional body.
    - 文件存在: 保留 frontmatter 其他 key + body, 原地替换 pins.
    - 永远不写 seen_* 观察态 (SPEC §7.2).
    """
    path = root / filename
    serialized = [_serialize_pin(p) for p in pins]

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = yaml.safe_dump(
            {"pins": serialized} if serialized else {},
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).rstrip()
        if body:
            body_block = body if body.startswith("\n") else f"\n{body}"
        else:
            body_block = "\n"
        path.write_text(f"---\n{fm}\n---\n{body_block}", encoding="utf-8")
        return

    text = path.read_text(encoding="utf-8")
    fm_match = _FRONTMATTER_RE.match(text)

    if fm_match is not None:
        # parse existing frontmatter, replace pins key
        fm_data = yaml.safe_load(fm_match.group("yaml")) or {}
        if serialized:
            fm_data["pins"] = serialized
        else:
            fm_data.pop("pins", None)
        fm_yaml = yaml.safe_dump(
            fm_data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).rstrip()
        body = fm_match.group("body")
        new_text = f"---\n{fm_yaml}\n---\n{body}"
    else:
        # no frontmatter — create one
        fm_yaml = yaml.safe_dump(
            {"pins": serialized} if serialized else {},
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).rstrip()
        new_text = f"---\n{fm_yaml}\n---\n\n{text}"

    path.write_text(new_text, encoding="utf-8")


def _serialize_pin(pin: Pin) -> dict:
    """Pin → K55 envelope dict: {verb, label, arguments: {...}, description?}.

    verb always first.  arguments excludes None/default values.
    description omitted when empty.
    """
    out: dict[str, object] = {}
    out["verb"] = type(pin).model_fields["verb"].default
    out["label"] = pin.label
    out["arguments"] = pin.arguments.model_dump(
        exclude_none=True, exclude_defaults=True
    )
    if pin.description:
        out["description"] = pin.description
    return out
