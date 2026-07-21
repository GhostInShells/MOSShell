"""L0 file — GROUND.md 读写.

三段结构 (SPEC §2):
    ---
    <frontmatter YAML>
    ---
    <body markdown>
    ## ground:pins
    <YAML list, discriminated union by ``kind``>

seen_* (PinShadow) 不进盘 — SPEC §7.2 的铁律.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ghoshell_moss.ground.contract import (
    FilePin,
    FrontmatterPin,
    GlobPin,
    GroundConvention,
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

# kind → Pin subclass dispatch
_PIN_CLASSES: dict[str, type[Pin]] = {
    "file": FilePin,
    "glob": GlobPin,
    "frontmatter": FrontmatterPin,
    "ls": LsPin,
}

# --- regex ----------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---[ \t]*\n(?P<rest>.*)",
    re.DOTALL,
)

_PIN_SECTION_RE = re.compile(
    r"^## ground:pins[ \t]*\n(?P<body>.*?)(?=^#+[ \t]+|\Z)",
    re.DOTALL | re.MULTILINE,
)

_PIN_YAML_BLOCK_RE = re.compile(
    r"```ya?ml[ \t]*\n(?P<yaml>.*?)\n```",
    re.DOTALL,
)


# --- L0Contents -----------------------------------------------------------


@dataclass
class L0Contents:
    """一份 GROUND.md 解析后的三段视图."""

    convention: GroundConvention
    body: str
    pins: list[Pin]

    @classmethod
    def empty(cls) -> "L0Contents":
        return cls(convention=GroundConvention(), body="", pins=[])


# --- load -----------------------------------------------------------------


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

    # pins
    pins: list[Pin] = []
    pin_match = _PIN_SECTION_RE.search(rest)
    if pin_match is not None:
        yaml_match = _PIN_YAML_BLOCK_RE.search(pin_match.group("body"))
        if yaml_match is not None:
            raw_list = yaml.safe_load(yaml_match.group("yaml")) or []
            pins = _deserialize_pins(raw_list)
        # 剥离 pin 段 body
        body = rest[: pin_match.start()] + rest[pin_match.end() :]
    else:
        body = rest

    return L0Contents(convention=convention, body=body, pins=pins)


def _deserialize_pins(raw: list[dict]) -> list[Pin]:
    """kind → class dispatch. 未知 kind 跳过 (SPEC §4.2 保留)."""
    result: list[Pin] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind", "")
        cls = _PIN_CLASSES.get(kind)
        if cls is None:
            continue
        # 移除 kind 后传给对应类 constructor
        fields = {k: v for k, v in item.items() if k != "kind"}
        result.append(cls(**fields))
    return result


# --- dump -----------------------------------------------------------------


def dump_l0_pins(
    root: Path,
    pins: list[Pin],
    filename: str = DEFAULT_L0_FILENAME,
) -> None:
    """把 pin 集写回 GROUND.md 的 ``## ground:pins`` 段.

    - 文件不存在: 创建, 只写 pin 段.
    - 文件存在, 无 pin 段: 追加.
    - 文件存在, 有 pin 段: in-place 替换.
    - 保留 frontmatter 与 body — 只动 pin 段 (K20).

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
    """渲染 pin 段: ``## ground:pins`` + yaml code block."""
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

    return (
        f"{PIN_SECTION_HEADING}\n\n"
        "<!-- Managed by `moss ground`. Do not hand-edit unless you know "
        "what you are doing. -->\n\n"
        f"```yaml\n{yaml_text}```\n\n"
    )


def _serialize_pin(pin: Pin) -> dict:
    """Pin → dict, 注入 kind discriminator."""
    data = pin.model_dump(exclude_none=False)
    # Ensure kind is present for deserialization
    if "kind" not in data:
        data["kind"] = type(pin).model_fields["kind"].default
    return data
