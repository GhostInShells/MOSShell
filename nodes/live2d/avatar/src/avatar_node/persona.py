"""形象人设与 idle 配置 — `avatars/<name>/AVATAR.md`.

模型资产描述"它长什么样", 人设描述"它是谁、该怎么说话", 这里再补一条: idle 怎么跑.
三者都是形象的一部分, 但分发性质相反: 模型资产受 Live2D 条款约束不入库 (见 INSTALL.md),
人设与配置是文本, **入库**.

格式沿用仓库主流约定 (frontmatter markdown, 同 NODE.md / HOST.md):

    ---
    name: Hiyori
    description: 开朗的少女, 动作轻快          # 一句冷人设
    voice: 可爱女生                            # 推荐音色 (TTS tone 名)
    groups:                                    # 可选: 覆盖某 group 子 channel 的 instruction
      face:
        instruction: "头部角度 angle_x/angle_y/angle_z, cheek 脸红"
    idle:                                      # 可选: 待机配置
      delay: 3.0                               # 空闲多久才进待机 (秒)
      loop:                                    # 全身待机: 一个循环动作组
        group: Idle
        index: 0
      parts:                                   # 部件级 idle (SDK 原生)
        blink: true
        breath: true
    ---
    正文是可选的更完整人设, 不进 instruction, 留给 channel.py 作者/人类.

文件缺失 → 返回 None, 形象照常可用 (人设是增强, 不是前置条件). 缺字段一律用默认值.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter

PERSONA_FILE = "AVATAR.md"

DEFAULT_IDLE_DELAY = 3.0


@dataclass(frozen=True)
class IdleConfig:
    delay: float = DEFAULT_IDLE_DELAY
    loop_group: str | None = None  # 全身待机动作组; None = 自动挑
    loop_index: int = 0
    blink: bool = True  # SDK 眨眼 (eye 组 idle)
    breath: bool = True  # SDK 呼吸 (body 组 idle)


@dataclass(frozen=True)
class Persona:
    name: str  # 形象自称 (缺省回退到套件目录名)
    description: str  # 一句话人设, 进 instruction
    voice: str  # 推荐音色 (TTS 的 tone 名), 进 instruction
    prose: str  # 正文人设, 不进 instruction
    group_instructions: dict[str, str] = field(default_factory=dict)  # group slug → instruction
    idle: IdleConfig = field(default_factory=IdleConfig)

    def is_empty(self) -> bool:
        return not (
            self.description
            or self.voice
            or self.prose
            or self.group_instructions
            or self.idle.loop_group is not None
        )


def _clean(value: object) -> str:
    return str(value or "").strip()


def _parse_idle(raw: Any) -> IdleConfig:
    if not isinstance(raw, dict):
        return IdleConfig()
    loop = raw.get("loop") if isinstance(raw.get("loop"), dict) else {}
    parts = raw.get("parts") if isinstance(raw.get("parts"), dict) else {}
    try:
        delay = float(raw.get("delay", DEFAULT_IDLE_DELAY))
    except (TypeError, ValueError):
        delay = DEFAULT_IDLE_DELAY
    return IdleConfig(
        delay=delay,
        loop_group=_clean(loop.get("group")) or None,
        loop_index=int(loop.get("index", 0)),
        blink=bool(parts.get("blink", True)),
        breath=bool(parts.get("breath", True)),
    )


def _parse_group_instructions(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for slug, entry in raw.items():
        if isinstance(entry, dict):
            text = _clean(entry.get("instruction"))
            if text:
                out[str(slug)] = text
    return out


def load(persona_file: Path, fallback_name: str) -> Persona | None:
    """解析一个套件的人设文件. 不存在返回 None."""
    if not persona_file.is_file():
        return None
    post = frontmatter.loads(persona_file.read_text(encoding="utf-8"))
    persona = Persona(
        name=_clean(post.get("name")) or fallback_name,
        description=_clean(post.get("description")),
        voice=_clean(post.get("voice")),
        prose=post.content.strip(),
        group_instructions=_parse_group_instructions(post.get("groups")),
        idle=_parse_idle(post.get("idle")),
    )
    return None if persona.is_empty() else persona
