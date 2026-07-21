"""L0 file — GROUND.md 读写.

三段结构 (SPEC §2):
    ---
    <frontmatter YAML>
    ---
    <body markdown>
    ## ground:pins
    <bare YAML list, K55 envelope: {label, verb, arguments, description}>

seen_* (PinShadow) 不进盘 — SPEC §7.2 的铁律.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ghoshell_moss.ground.contract import (
    _VERB_CLASSES,
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    GroundConvention,
    LsArguments,
    LsPin,
    Pin,
)

__all__ = [
    "DEFAULT_L0_FILENAME",
    "PIN_SECTION_HEADING",
    "L0Contents",
    "load_l0",
    "dump_l0_pins",
]

DEFAULT_L0_FILENAME = "GROUND.md"
PIN_SECTION_HEADING = "## ground:pins"

# -- regex ----------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---[ \t]*\n(?P<rest>.*)",
    re.DOTALL,
)

_PIN_SECTION_RE = re.compile(
    r"^## ground:pins[ \t]*\n(?P<body>.*?)(?=^#+[ \t]+|\Z)",
    re.DOTALL | re.MULTILINE,
)

# K55: arguments models per verb — used for deserialization
_ARG_CLASSES: dict[str, type] = {
    "file": FileArguments,
    "glob": GlobArguments,
    "frontmatter": FrontmatterArguments,
    "ls": LsArguments,
}


# -- L0Contents -----------------------------------------------------------


@dataclass
class L0Contents:
    """一份 GROUND.md 解析后的三段视图."""

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

    # frontmatter
    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match is not None:
        fm_data = yaml.safe_load(fm_match.group("yaml")) or {}
        convention = GroundConvention(**fm_data)
        rest = fm_match.group("rest")
    else:
        convention = GroundConvention()
        rest = text

    # pins — bare YAML under ## ground:pins (K55: no fenced code block)
    pins: list[Pin] = []
    pin_match = _PIN_SECTION_RE.search(rest)
    if pin_match is not None:
        yaml_text = pin_match.group("body").strip()
        if yaml_text:
            raw_list = yaml.safe_load(yaml_text) or []
            pins = _deserialize_pins(raw_list)
        body = rest[: pin_match.start()] + rest[pin_match.end() :]
    else:
        body = rest

    return L0Contents(convention=convention, body=body, pins=pins)


def _deserialize_pins(raw: list[dict]) -> list[Pin]:
    """K55 envelope → Pin subclass dispatch.

    Each item: {label, verb, arguments: {...}, description?}.
    ``arguments`` is passed to the verb's arguments model constructor;
    unknown verbs are skipped (SPEC §4.2).
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
            # unknown verb — preserve raw dict as-is
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
) -> None:
    """把 pin 集写回 GROUND.md 的 ``## ground:pins`` 段 (K55 envelope).

    - 文件不存在: 创建, 只写 pin 段.
    - 文件存在, 无 pin 段: 追加.
    - 文件存在, 有 pin 段: in-place 替换.
    - 保留 frontmatter 与 body — 只动 pin 段 (K20).
    - 裸 YAML, 无 fenced code block (K55).

    seen_* 永远不落盘 — Pin 模型不含观察影子.
    """
    path = root / filename
    new_section = _render_pin_section(pins)

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_section, encoding="utf-8")
        return

    text = path.read_text(encoding="utf-8")
    pin_match = _PIN_SECTION_RE.search(text)
    if pin_match is not None:
        new_text = text[: pin_match.start()] + new_section + text[pin_match.end() :]
    else:
        prefix = text.rstrip("\n")
        sep = "\n\n" if prefix else ""
        new_text = prefix + sep + new_section

    path.write_text(new_text, encoding="utf-8")


def _render_pin_section(pins: list[Pin]) -> str:
    """渲染 pin 段: ``## ground:pins`` + 裸 YAML (K55 envelope)."""
    if pins:
        data = [_serialize_pin(p) for p in pins]
        yaml_text = yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    else:
        yaml_text = "[]\n"

    return f"{PIN_SECTION_HEADING}\n{yaml_text}\n"


def _serialize_pin(pin: Pin) -> dict:
    """Pin → K55 envelope: {label, verb, arguments: {...}, description?}.

    verb always first.  arguments excludes None/default values.
    description omitted when empty.
    """
    out: dict[str, object] = {}
    # verb — always present, always first
    out["verb"] = type(pin).model_fields["verb"].default
    # label
    out["label"] = pin.label
    # arguments — exclude None/defaults within
    out["arguments"] = pin.arguments.model_dump(
        exclude_none=True, exclude_defaults=True
    )
    # description — only when non-empty
    if pin.description:
        out["description"] = pin.description
    return out
