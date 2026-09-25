"""Cubism 模型包解析 — model3.json + cdi3.json → ModelSpec.

不依赖任何 Live2D SDK: 两个文件都是 JSON, 是 Cubism 模型包的公开元数据.

    model3.json  FileReferences.Moc / Textures / Physics / Expressions / Motions,
                 Groups (声明的唇形/眨眼绑定), HitAreas
    cdi3.json    Parameters (每个参数的 Id / Name / GroupId),
                 ParameterGroups (GroupId → Name)

解析出的 ModelSpec 是**驱动的能力真相**: 有哪些参数, 怎么分组, 有哪些动作/表情.
页面渲染时用模型自己的 moc3 拿真实上下界; 这里只做结构与命名.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .lexicon import group_slug, param_ident

# Cubism Editor 的旋转变形器会自动生成一批中间参数 (Param_Angle_Rotation_<n>_ArtMesh<id>),
# 对驱动是纯噪声. 这是**共享规则** (KD3): 所有模型里都是噪声, 不是 per-model 配置.
_NOISE_RE = re.compile(r"^Param_Angle_Rotation_\d+_ArtMesh\d+$")


def is_noise(param_id: str) -> bool:
    return bool(_NOISE_RE.match(param_id))


@dataclass(frozen=True)
class Param:
    id: str
    label: str  # cdi3 Name, 模型作者的语言
    group_id: str

    @property
    def ident(self) -> str:
        """命令名. 同一分组内保证唯一 (见 Group.unique_idents)."""
        return param_ident(self.id, self.label)


@dataclass(frozen=True)
class Group:
    id: str
    label: str  # cdi3 ParameterGroups Name
    slug: str  # 英文 channel 名
    params: tuple[Param, ...]

    def idents(self) -> list[str]:
        """参数命令名, 组内去重 (碰撞时加序号)."""
        seen: dict[str, int] = {}
        out: list[str] = []
        for p in self.params:
            base = p.ident
            if base in seen:
                seen[base] += 1
                out.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                out.append(base)
        return out


@dataclass(frozen=True)
class ModelSpec:
    model_json: Path
    groups: tuple[Group, ...]
    motions: dict[str, tuple[str, ...]]  # 动作组名 → 动作名列表
    expressions: tuple[str, ...]
    hit_areas: tuple[str, ...]
    lip_sync: tuple[str, ...]  # model3 Groups.LipSync 声明的参数 id
    eye_blink: tuple[str, ...]  # model3 Groups.EyeBlink 声明的参数 id
    noise: tuple[str, ...] = field(default=())  # 被过滤掉的参数, 仅用于报告
    motion_durations: dict[tuple[str, int], float] = field(default_factory=dict)  # (组名, 序号) → 秒

    @property
    def params(self) -> tuple[Param, ...]:
        return tuple(p for g in self.groups for p in g.params)

    def motion_duration(self, group: str, index: int = 0) -> float:
        """一个动作的单圈时长 (motion3.json 的 Meta.Duration). 未知时回退 3.0s."""
        return self.motion_durations.get((group, index), 3.0)

    def find_group(self, slug: str) -> Group | None:
        for g in self.groups:
            if g.slug == slug:
                return g
        return None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_model_json(model_dir: Path, prefer: str | None = None) -> Path | None:
    """在模型目录里找入口 model3.json.

    优先 `<prefer>.model3.json`; 否则任意一个 `*.model3.json`.
    找不到返回 None —— 调用方据此报告"这套 avatar 不可用".
    """
    if prefer:
        candidate = model_dir / f"{prefer}.model3.json"
        if candidate.is_file():
            return candidate
    found = sorted(model_dir.glob("*.model3.json"))
    return found[0] if found else None


def parse(model_json: Path) -> ModelSpec:
    """解析模型包的 model3.json (+ 同级 cdi3.json), 得到能力描述."""
    model = _load_json(model_json)
    refs = model.get("FileReferences", {}) or {}

    motions: dict[str, tuple[str, ...]] = {}
    motion_durations: dict[tuple[str, int], float] = {}
    for group_name, entries in (refs.get("Motions") or {}).items():
        names = tuple(e.get("Name") or Path(e.get("File", "")).stem for e in entries)
        motions[group_name] = names
        for idx, e in enumerate(entries):
            duration = _read_motion_duration(model_json, e.get("File", ""))
            if duration > 0:
                motion_durations[(group_name, idx)] = duration

    expressions = tuple(e.get("Name") or Path(e.get("File", "")).stem for e in (refs.get("Expressions") or []))

    hit_areas = tuple(h.get("Name", "") for h in (model.get("HitAreas") or []) if h.get("Name"))

    lip_sync: tuple[str, ...] = ()
    eye_blink: tuple[str, ...] = ()
    for g in model.get("Groups") or []:
        ids = tuple(g.get("Ids") or ())
        if g.get("Name") == "LipSync":
            lip_sync = ids
        elif g.get("Name") == "EyeBlink":
            eye_blink = ids

    groups, noise = _parse_cdi(model_json)

    return ModelSpec(
        model_json=model_json,
        groups=groups,
        motions=motions,
        expressions=expressions,
        hit_areas=hit_areas,
        lip_sync=lip_sync,
        eye_blink=eye_blink,
        noise=noise,
        motion_durations=motion_durations,
    )


def _read_motion_duration(model_json: Path, file_ref: str) -> float:
    """读一个 motion3.json 的单圈时长 (Meta.Duration). 读不到/缺字段返回 0."""
    if not file_ref:
        return 0.0
    try:
        data = _load_json(model_json.parent / file_ref)
    except (OSError, ValueError):
        return 0.0
    try:
        return float(data.get("Meta", {}).get("Duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_cdi(model_json: Path) -> tuple[tuple[Group, ...], tuple[str, ...]]:
    """cdi3.json → 分组. cdi3 缺失时返回空分组 (调用方走扁平降级)."""
    cdi_path = model_json.with_suffix("").with_name(model_json.name.replace(".model3.json", ".cdi3.json"))
    if not cdi_path.is_file():
        # Cubism 的 DisplayInfo 字段可能指向别处的 cdi3; 也接受同级任意 cdi3.
        candidates = sorted(model_json.parent.glob("*.cdi3.json"))
        if not candidates:
            return (), ()
        cdi_path = candidates[0]

    cdi = _load_json(cdi_path)
    group_labels = {g["Id"]: g.get("Name", "") for g in cdi.get("ParameterGroups") or []}

    buckets: dict[str, list[Param]] = {}
    order: list[str] = []
    noise: list[str] = []
    for raw in cdi.get("Parameters") or []:
        pid = raw.get("Id")
        if not pid:
            continue
        if is_noise(pid):
            noise.append(pid)
            continue
        gid = raw.get("GroupId") or ""
        if gid not in buckets:
            buckets[gid] = []
            order.append(gid)
        buckets[gid].append(Param(id=pid, label=raw.get("Name", ""), group_id=gid))

    groups: list[Group] = []
    used: set[str] = set()
    for gid in order:
        label = group_labels.get(gid, "")
        slug = group_slug(label, fallback=group_slug(gid, fallback="group"))
        if slug in used:
            n = 1
            while f"{slug}_{n}" in used:
                n += 1
            slug = f"{slug}_{n}"
        used.add(slug)
        groups.append(Group(id=gid, label=label, slug=slug, params=tuple(buckets[gid])))

    return tuple(groups), tuple(noise)
