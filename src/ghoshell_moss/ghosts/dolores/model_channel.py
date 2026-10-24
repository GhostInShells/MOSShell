"""Dolores 的模型器官 —— ghost.model | 集成 | alpha

让 ghost 看见自己用的是什么脑子, 并且能换一颗. 三段, 各有各的层:

- 热 ``named notice`` (挂在 ghost 根 channel 上): 当前 provider/model + 档位. 变才重发, 没变
  零成本. 数据源是 ``request/header`` 的观测 —— 也就是"我实际在跑什么", 不是目录默认值
  (目录默认值是"新会话会从哪开始", 两件事, 混用会让界面上的改动看起来没生效).
- 冷 ``list``: 现读 dsh 的模型目录 (provider → model → 该模型自带的档位表). 走 ego 已经持有
  的那个 dsh session, 不新开连接、不要新凭据.
- 写 ``select``: 只**声明**, 不写 dsh. 声明进 ego 的待采用状态, 下一个真正带上它的思考帧把它
  交给插件, 插件在 turn 边界写一次官方 ``selectForNextRequest``.

为什么写只留一处: 这个器官曾经的长辈 (插件里那份私有 ModelSelection) 和 python 侧的
``selectModel`` 各写一半, 结果同一个档位有两个权威, 界面改了不生效、ghost 声明了又改掉人类的
全局默认. 现在的形状是 —— 声明可以有多个入口 (CTML 命令 / moss_reasoning 工具), **写**只有
插件那一处; 全局默认归人类, ghost 一次也不碰.

面向模型的文本 (instruction / 命令 docstring 与返回值) 一律英文 —— 与 dolores 的提示词体系
一致; 中文只留给开发者注释.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

__all__ = ["ModelOrgan", "new_model_channel", "identity_text"]


@dataclass(frozen=True)
class ModelOrgan:
    """ghost.model 的后端三件事 — 读目录 / 读现状 / 声明下一个请求.

    全部由 ghost runtime 注入: 读目录要 ego 那个 dsh session, 读现状要 ego 的观测,
    声明要写进 ego 的待采用状态. 这个 channel 自己不持有任何权威.
    """

    read_catalog: Callable[[], Awaitable[Any]]
    """dsh 的模型目录 (``session/modelCatalog``) —— 每次调用现读."""

    read_identity: Callable[[], "tuple[str, str, str] | None"]
    """最近一次 request/header 观测到的 (provider, model, effort). None = 还没跑过请求."""

    declare: Callable[[str, str, str], str]
    """声明下一个请求的模型 (+ 可选档位); 返回一行收据. 不写 dsh."""


def identity_text(identity: "tuple[str, str, str] | None") -> str:
    """温 notice 的一句话: 我实际在跑哪个模型、哪一档. 没观测过就如实说没有.

    不带标签前缀 —— 它整体就是 named fragment ``model`` 的内容.
    """
    if identity is None:
        return "not observed yet (no request from this session)"
    provider, model, effort = identity
    line = f"{provider}/{model}"
    return line + (f" · thinking {effort}" if effort else " · thinking (model default)")


def _catalog_text(catalog: Any) -> str:
    """目录 → 模型可读的文本. 按 provider 分组, 每个模型带它自己的档位表."""
    groups = getattr(catalog, "groups", []) or []
    default = getattr(catalog, "default", None)
    routable = set(getattr(catalog, "routableProviders", []) or [])
    failures = getattr(catalog, "failures", []) or []
    if not groups and not failures:
        return "[model] the catalog is empty — no provider is mounted"
    lines = ["[model] dsh catalog (live):"]
    for group in groups:
        mark = "" if group.id in routable else "  (not routable right now)"
        lines.append(f"  {group.id}{mark}")
        for model in group.models:
            ids = [effort.id for effort in (model.reasoning.efforts if model.reasoning else []) if effort.id]
            depth = ""
            if ids:
                depth = "  depths " + "/".join(ids)
                fallback = getattr(model.reasoning, "defaultEffort", None)
                if fallback:
                    depth += f" (default {fallback})"
            detail = f" — {model.description}" if model.description else ""
            lines.append(f"    {model.id}{detail}{depth}")
    if failures:
        lines.append("  unavailable:")
        lines.extend(f"    {failure.id}: {failure.message}" for failure in failures)
    if default is not None and getattr(default, "model", ""):
        current = f"  catalog default: {default.provider}/{default.model}"
        if getattr(default, "reasoningEffort", None):
            current += f" @ {default.reasoningEffort}"
        lines.append(current)
    lines.append(
        "  (the catalog default is where *new* sessions start; what you are running is the"
        " `model` notice on the ghost channel)"
    )
    return "\n".join(lines)


def _instruction_text() -> str:
    return "\n".join([
        "## Model (your own brain)",
        "",
        "- `list` is the cold read of dsh's live catalog: providers, models, and the thinking",
        "  depths each model offers. Call it before you think about switching — never guess an id.",
        "- The warm notice on the ghost channel says what you are running *now*; it is re-sent only",
        "  when it changes, so you do not have to ask.",
        "- `select` declares, it does not switch: the declaration rides the next thinking frame and",
        "  takes effect at that turn's boundary — one request later.",
        "- The write path is single. The plugin writes it once into dsh's own selection at the turn",
        "  boundary; the global default stays the human's, never yours.",
        "- Your eyes are a property of the model. A model without image input turns them off",
        "  silently, and the catalog does not report modality — do not trade sight for speed",
        "  without knowing which model you are going to.",
    ])


def new_model_channel(
    organ: ModelOrgan,
    *,
    name: str = "model",
    description: str | None = None,
) -> MutableChannel:
    """组装模型器官 —— ghost 的"我用的哪颗脑子"控制面.

    :param organ: 后端三件事 (读目录 / 读现状 / 声明), 由 ghost runtime 注入.
    :param name: CTML 标签名 (``ghost.model``).
    :param description: 覆盖默认描述.
    """
    if description is None:
        description = (
            "Your own model: `list` reads dsh's live catalog (providers, models, each model's"
            " thinking depths), `select` declares the model for the next request. What you are"
            " running now rides the ghost channel's `model` notice."
        )
    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        return _instruction_text()

    @chan.build.command(name="list", always_observe=True)
    async def list_models() -> str:
        """Read the dsh model catalog — providers, models, and each model's thinking depths.

        A live call into the dsh session you already hold, so it answers "what is mounted right
        now", including providers that failed to load. What *you* run is not in here — that is
        the `model` notice on the ghost channel.
        """
        try:
            catalog = await organ.read_catalog()
        except Exception as error:  # noqa: BLE001 — 读失败要说人话, 不是抛栈
            return f"[model] catalog unavailable: {error}"
        return _catalog_text(catalog)

    @chan.build.command(name="select", always_observe=True)
    async def select(provider: str, model: str, effort: str = "") -> str:
        """Declare the model for the next request — effective at the next turn boundary.

        Nothing is written here: the declaration rides the next thinking frame, and the plugin
        writes it once through dsh's official selection API. That is the only write path, so the
        human's global default is never touched.

        :param provider: provider id from `list`, e.g. `deepseek-official`.
        :param model: model id from `list`, e.g. `deepseek-flash`.
        :param effort: depth to travel with the switch (off/low/high/max). Empty = keep the depth
            you are on. A depth the new model does not offer is dsh's problem, not a silent drop.
        """
        try:
            receipt = organ.declare(provider, model, effort)
        except ValueError as error:
            return f"[model] {error}"
        return f"[model] {receipt} — applies to the next request"

    return chan
