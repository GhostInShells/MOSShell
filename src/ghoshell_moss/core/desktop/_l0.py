"""L0 file — Ground 的持久化载体 (DESKTOP.md).

L0 是场目录里一份 markdown 文件, 载体结构 (K15):

    ---
    <GroundConvention YAML>   ← frontmatter, MOSS 消费
    ---

    <free-form body>          ← 法 / 治理 / 先例, 模型消费

    ## desktop:pins           ← sediment 目标段, moss 独占

    ```yaml
    - addr: ...
      ...
    ```

frontmatter 缺 → GroundConvention() 默认. body 缺 → 空. pin 段缺 → 空 pin
集. sediment 只重写 pin 段, 保留 frontmatter 与 body 的其余部分, 避免 git
diff 噪音 (K20).

文件名 `DESKTOP.md` (K22 决策 2026-07-13): 骑 CLAUDE.md / AGENTS.md /
SKILL.md 命名先验; K8 原始候选之一, 无预训练风险.

sync IO: 本模块函数全部 sync — DefaultGround.load / sediment 在其调用点
用 asyncio.to_thread 卸载. 测试因此可以走同步路径, 不用 asyncio.run 包壳.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ghoshell_moss.contracts.desktop import GroundConvention, Pin

__all__ = [
    "DEFAULT_L0_FILENAME",
    "PIN_SECTION_HEADING",
    "L0Contents",
    "load_l0",
    "dump_l0_pins",
]

DEFAULT_L0_FILENAME = "DESKTOP.md"
PIN_SECTION_HEADING = "## desktop:pins"

# YAML frontmatter: 文件以 `---\n` 开头, 到下一个独占的 `---` 行为止
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---[ \t]*\n(?P<rest>.*)",
    re.DOTALL,
)

# pin 段: `## desktop:pins` heading 到下一个任意级 heading 或 EOF
_PIN_SECTION_RE = re.compile(
    r"^## desktop:pins[ \t]*\n"
    r"(?P<body>.*?)"
    r"(?=^#+[ \t]+|\Z)",
    re.DOTALL | re.MULTILINE,
)

# pin 段里的 yaml 代码块
_PIN_YAML_BLOCK_RE = re.compile(
    r"```yaml[ \t]*\n(?P<yaml>.*?)```",
    re.DOTALL,
)


@dataclass
class L0Contents:
    """一份 L0 文件解析后的三段视图.

    `empty()` 用在 "root 里没有 L0 文件" 的情况: convention 用默认值,
    body 空, pins 空.
    """
    convention: GroundConvention
    body: str  # frontmatter 之后, 剥去 pin 段之后的正文
    pins: list[Pin]

    @classmethod
    def empty(cls) -> "L0Contents":
        return cls(convention=GroundConvention(), body="", pins=[])


def load_l0(root: Path, filename: str = DEFAULT_L0_FILENAME) -> L0Contents:
    """从场 root 加载 L0. 文件不存在返回 empty.

    Raises:
        yaml.YAMLError: frontmatter 或 pin 段 YAML 语法错误.
        pydantic.ValidationError: frontmatter 字段不符合 GroundConvention.
    """
    path = root / filename
    if not path.is_file():
        return L0Contents.empty()

    text = path.read_text(encoding="utf-8")

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match is not None:
        fm_data = yaml.safe_load(fm_match.group("yaml")) or {}
        convention = GroundConvention(**fm_data)
        body_and_pins = fm_match.group("rest")
    else:
        convention = GroundConvention()
        body_and_pins = text

    pins: list[Pin] = []
    pin_match = _PIN_SECTION_RE.search(body_and_pins)
    if pin_match is not None:
        yaml_match = _PIN_YAML_BLOCK_RE.search(pin_match.group("body"))
        if yaml_match is not None:
            raw = yaml.safe_load(yaml_match.group("yaml")) or []
            pins = [Pin(**d) for d in raw]
        # 从 body 中整段剥离 pin 段 — 它是 moss 独占的
        body = body_and_pins[: pin_match.start()] + body_and_pins[pin_match.end() :]
    else:
        body = body_and_pins

    return L0Contents(convention=convention, body=body, pins=pins)


def dump_l0_pins(
    root: Path,
    pins: list[Pin],
    filename: str = DEFAULT_L0_FILENAME,
) -> None:
    """把 pin 集写回 L0 文件的 pin 段, 保留 frontmatter 与 body 其余部分.

    行为:
    - 文件不存在: 创建, 只写 pin 段.
    - 文件存在, 无 pin 段: 追加 pin 段到 body 末尾.
    - 文件存在, 有 pin 段: in-place 替换 pin 段.

    幂等 (等价 pin 集重写产生等价文件).
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
        # 追加, 前面留一个空行分隔
        prefix = text.rstrip("\n")
        sep = "\n\n" if prefix else ""
        new_text = prefix + sep + new_section

    path.write_text(new_text, encoding="utf-8")


def _render_pin_section(pins: list[Pin]) -> str:
    """渲染 pin 段的固定形态."""
    pin_data = [p.model_dump(exclude_none=False) for p in pins]
    if pin_data:
        yaml_text = yaml.safe_dump(
            pin_data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    else:
        yaml_text = "[]\n"
    return (
        f"{PIN_SECTION_HEADING}\n\n"
        "<!-- Managed by `moss desktop`. Do not hand-edit unless you know "
        "what you are doing. -->\n\n"
        f"```yaml\n{yaml_text}```\n\n"
    )
