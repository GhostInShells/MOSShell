"""自动映射 — 没有 `channel.py` 时, 从模型自带元数据推一份示范命令面.

**这是示范路径, 不是主路径.** 真正的形象由模型作者写 `avatars/<name>/channel.py`
(用 channel builder), 自动映射只在套件没写 channel 时兜底, 用来证明
"把一个标准 Cubism 包丢进来就能驱动"这条约定成立.

映射规则 (KD2):
  cdi3 的参数按 GroupId 分组 → 每个组一个子 channel, 组内每个参数一条命令
  model3 的 Motions          → 每个动作组一条命令
  model3 的 Expressions      → 每个表情一条命令
  Groups.LipSync / EyeBlink  → 记录下来, 供形象作者绑定 (本模块不主动驱动)
"""

from __future__ import annotations

import re

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from .avatar import Avatar
from .cubism import Group, Param
from .lexicon import param_doc, param_ident

_SNAKE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, fallback: str) -> str:
    s = _SNAKE.sub("_", (text or "").lower()).strip("_")
    if not s:
        return fallback
    return f"p{s}" if s[0].isdigit() else s


def build_auto_channel(avatar: Avatar) -> MutableChannel:
    """把 ModelSpec 映射成一条可驱动的 channel 树."""
    spec = avatar.spec
    root = new_channel(name=avatar.name, description=f"{avatar.name} — 自动映射的 Live2D 形象")

    @root.build.instruction
    def _instruction() -> str:
        groups = ", ".join(g.slug for g in spec.groups)
        lines = [
            f"你驱动的是一个 Live2D 形象 `{avatar.name}`。命令由模型包自带的元数据自动映射而来。",
            f"参数按模型作者的分组组织成子 channel: {groups}。",
            "参数命令取模型原值 (通常 -1 到 1, 头/身体角度类约 -30 到 30), 页面会按模型实际上下界夹取。",
            "动作与表情命令在 `motions` / `expressions` 子 channel 下, 播完自动回待机。",
        ]
        if avatar.client_count == 0:
            lines.append("当前没有页面连着 —— 你的命令会被记录, 但没人看见。")
        return "\n".join(lines)

    for group in spec.groups:
        root.import_channels(_group_channel(avatar, group))

    if spec.motions:
        root.import_channels(_motions_channel(avatar))
    if spec.expressions:
        root.import_channels(_expressions_channel(avatar))

    if avatar.backdrop_names:
        _register_backdrop(root, avatar, avatar.backdrop_names)

    @root.build.command()
    async def reset() -> str:
        """把表情与所有参数复位回模型默认。"""
        avatar.reset()
        return "已复位"

    @root.build.command()
    async def status() -> str:
        """报告当前被命令过的参数 (驱动的真相, 不是页面回读)。"""
        state = avatar.state()
        if not state:
            return "尚未下达任何参数命令"
        items = ", ".join(f"{k}={v:g}" for k, v in sorted(state.items()))
        page = "页面已连接" if avatar.client_count else "无页面连接"
        return f"{page}; 参数: {items}"

    @root.build.command()
    async def zoom(factor: float = 1.0) -> str:
        """缩放形象。factor=1.0 为默认大小, 大于 1 放大, 小于 1 缩小。"""
        avatar.zoom(factor)
        return f"缩放 {factor:g}"

    @root.build.command()
    async def move(x: float = 0.0, y: float = 0.0) -> str:
        """平移形象。x/y 取值 -1 到 1, (0,0)=视口中心, (-1,-1)=左上角。"""
        avatar.move(x, y)
        return f"移到 ({x:g}, {y:g})"

    _idle_listing = ", ".join(spec.motions) if spec.motions else ""

    @root.build.command(doc=f"设定待机动作 (空闲时循环播放)。可用动作组: {_idle_listing}。")
    async def set_idle(group: str, index: int = 0) -> str:
        avatar.set_idle(group, index)
        return f"待机设为 {group}[{index}]"

    @root.build.command()
    async def lip_sync(on: bool = True) -> str:
        """开关自动唇动。on=False 时模型手动控嘴优先; on=True 恢复跟随说侧采样。"""
        avatar.set_lip_sync(on)
        return f"自动唇动 {'开' if on else '关'}"

    @root.build.notice
    def _notice() -> str:
        page = "有页面正在看着你。" if avatar.client_count else "没有页面连接 —— 你说的话没人看见。"
        recent = avatar.interactions()
        if recent:
            tail = " · ".join(recent)
            return f"{page} 最近交互: {tail}"
        return page

    @root.build.idle
    async def _idle() -> None:
        # 空闲时回到待机动作; 没有动作组时什么都不做.
        await avatar.idle()

    from .lipsync import run_lip_sync

    @root.build.running
    async def _lip_sync() -> None:
        # 连续订阅说侧采样驱动唇形 (跨进程 topic 桥). 无唇形参数时内部直接返回.
        await run_lip_sync(avatar)

    return root


# --------------------------------------------------------------------------- 背板


def _register_backdrop(chan: MutableChannel, avatar: Avatar, names: tuple[str, ...]) -> None:
    listing = ", ".join(names)

    async def _set_backdrop(name: str) -> str:
        avatar.set_backdrop(f"/backdrop/{name}")
        return f"背板已换成 {name}"

    _set_backdrop.__name__ = "set_backdrop"
    _set_backdrop.__doc__ = f"换页面背板图。可选: {listing}。"
    chan.build.command()(_set_backdrop)


# --------------------------------------------------------------------------- 分组


def _group_channel(avatar: Avatar, group: Group) -> MutableChannel:
    chan = new_channel(name=group.slug, description=f"{group.label or group.slug} 分组参数")
    for param, ident in zip(group.params, group.idents()):
        _register_param(chan, avatar, param, ident)
    return chan


def _register_param(chan: MutableChannel, avatar: Avatar, param: Param, ident: str) -> None:
    doc = f"{param.label}。{param_doc(param.id)}" if param.label else param_doc(param.id)

    async def _set(value: float) -> None:
        avatar.param(param.id, value, manual=True)

    _set.__name__ = ident
    _set.__doc__ = doc
    chan.build.command()(_set)


# --------------------------------------------------------------------------- 动作


def _motions_channel(avatar: Avatar) -> MutableChannel:
    chan = new_channel(name="motions", description="播放模型自带动作")
    for group_name, names in avatar.spec.motions.items():
        _register_motion(chan, avatar, group_name, names)
    return chan


def _register_motion(chan: MutableChannel, avatar: Avatar, group_name: str, names: tuple[str, ...]) -> None:
    ident = _slug(group_name, "motion")
    listing = " / ".join(f"{i}={n}" for i, n in enumerate(names))

    async def _play(index: int = 0, loop: bool = False) -> None:
        avatar.motion(group_name, index, loop=loop)

    _play.__name__ = ident
    _play.__doc__ = f"播放动作「{group_name}」, 可选: {listing}。loop=True 时循环播放。"
    chan.build.command()(_play)


# --------------------------------------------------------------------------- 表情


def _expressions_channel(avatar: Avatar) -> MutableChannel:
    chan = new_channel(name="expressions", description="切换模型自带表情")
    for name in avatar.spec.expressions:
        _register_expression(chan, avatar, name)
    return chan


def _register_expression(chan: MutableChannel, avatar: Avatar, name: str) -> None:
    ident = _slug(param_ident(name), "expression")

    async def _apply() -> None:
        avatar.expression(name)

    _apply.__name__ = ident
    _apply.__doc__ = f"切换到表情「{name}」。"
    chan.build.command()(_apply)
