"""套件发现 — `avatars/<name>/` 是形象的自包含单元.

发现规则 (两级, 显式接管, 不合并 —— KD1):

    avatars/<name>/
    ├── channel.py   有 → 它就是这个形象的命令面 (模型作者完全控制)
    └── model/       本地模型资产 (不入库), 里面必须有一个 *.model3.json

没有 `channel.py` → 回落到 `mapper.build_auto_channel`, 按模型元数据自动映射一份示范面.

`channel.py` 的约定只有一个入口:

    async def build(avatar: Avatar) -> PrimeChannel: ...

`avatar` 是驱动给形象作者的事件面 (`avatar_node.Avatar`). 作者只依赖这一个对象.
返回必须是 PrimeChannel (用 ``new_prime_channel`` 构建), 因为驱动要 ``with_module`` 挂动画轨迹.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from ghoshell_moss.core.blueprint.states_channel import PrimeChannel

from .avatar import Avatar
from .cubism import ModelSpec, find_model_json, parse
from .persona import PERSONA_FILE, Persona
from .persona import load as load_persona

AVATAR_DIR = "avatars"
MODEL_DIR = "model"
CHANNEL_FILE = "channel.py"


class AvatarNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class AvatarKit:
    name: str
    path: Path
    model_dir: Path
    channel_file: Path | None
    model_json: Path
    persona: Persona | None

    @property
    def is_explicit(self) -> bool:
        return self.channel_file is not None


def discover(root: Path) -> dict[str, AvatarKit]:
    """枚举 root/avatars/*/ 下所有**可用**套件 (有 channel.py 或 model/*.model3.json).

    不可用的目录直接跳过 —— 缺模型资产的套件不该让 node 起不来, 只是它不出现.
    """
    avatars_root = Path(root) / AVATAR_DIR
    kits: dict[str, AvatarKit] = {}
    if not avatars_root.is_dir():
        return kits
    for entry in sorted(avatars_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        model_dir = entry / MODEL_DIR
        if not model_dir.is_dir():
            continue
        model_json = find_model_json(model_dir, prefer=entry.name)
        if model_json is None:
            continue
        channel_file = entry / CHANNEL_FILE
        kits[entry.name] = AvatarKit(
            name=entry.name,
            path=entry,
            model_dir=model_dir,
            channel_file=channel_file if channel_file.is_file() else None,
            model_json=model_json,
            persona=load_persona(entry / PERSONA_FILE, fallback_name=entry.name),
        )
    return kits


def select(root: Path, name: str | None) -> AvatarKit:
    """按名字选套件. name 为空时取发现的第一个; 找不到时抛 AvatarNotFoundError 并列出可用项."""
    kits = discover(root)
    if not kits:
        raise AvatarNotFoundError(f"没有可用的形象套件。请在 {Path(root) / AVATAR_DIR}/<name>/model/ 放入模型资产。")
    if name is None:
        return next(iter(kits.values()))
    if name not in kits:
        available = ", ".join(kits)
        raise AvatarNotFoundError(f"未知形象 {name!r}。可用: {available}")
    return kits[name]


def load_spec(kit: AvatarKit) -> ModelSpec:
    return parse(kit.model_json)


def model_url(kit: AvatarKit) -> str:
    """页面取模型的 URL —— 相对 bridge 挂载的 /model/ 前缀."""
    return f"/model/{kit.model_json.name}"


async def load_channel(kit: AvatarKit, avatar: Avatar) -> PrimeChannel:
    """取这套形象的命令面: 显式 `channel.py` 优先, 否则自动映射.

    两种路径都返回同一棵 PrimeChannel 后, 再挂上动画轨迹模块 (animations.py), 见
    ``setup_animations``。显式 `build(avatar)` 也必须返回 PrimeChannel (用
    ``new_prime_channel`` 构建), 否则无法挂 module。
    """
    if kit.channel_file is None:
        from .mapper import build_auto_channel

        channel: PrimeChannel = build_auto_channel(avatar)
    else:
        module = _import_from_path(kit.channel_file)
        build = getattr(module, "build", None)
        if build is None:
            raise AttributeError(f"{kit.channel_file} 必须定义 `async def build(avatar) -> PrimeChannel`")
        channel = build(avatar)
        if hasattr(channel, "__await__"):
            channel = await channel

    from .animations import ANIMATIONS_FILE, setup_animations

    setup_animations(channel, avatar, kit.path / ANIMATIONS_FILE)
    return channel


def _import_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(f"avatar_kit_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {path} 加载形象套件")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
