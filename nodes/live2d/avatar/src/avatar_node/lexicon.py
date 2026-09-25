"""共享词表 — 跨模型通用的命名与量纲规则.

Cubism 模型包自带 cdi3.json, 它把参数按 `GroupId` 分组, 并给出两样东西:
  - 每个参数的 `Name` (模型作者的语言: 日文 / 中文)
  - 每个分组的 `Name` (同上)

这些名字对模型驱动是噪声 —— 同一件事在不同模型里叫 `目` / `眼` / 眼睛. 本模块提供:

1. `group_slug()` — 把分组名译成稳定的英文 channel 名 (左/右自动加后缀).
2. `param_ident()` — 把参数 id 归一成 python 命令名.
3. `PARAM_HINTS` — 标准 Cubism 参数的量纲提示 (给命令 docstring 用).

全部是**共享规则**, 不是 per-model 配置 (KD3): 它们描述的是 Cubism 参数词汇本身,
对所有标准模型成立. 未知名字回退到原名, 不会失败.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 分组名 → 英文 channel 名
#
# 匹配是"子串命中", 因为模型作者会把方位/部件拼进去 (如 揺れ　サイドアップ左,
# 左辫子, 右辫子). 表按长度降序试探, 保证 目玉 先于 目 命中.
# ---------------------------------------------------------------------------

_GROUP_WORDS: tuple[tuple[str, str], ...] = (
    # 颜面
    ("目玉", "eyeballs"),
    ("眼球", "eyeballs"),
    ("眉毛", "brows"),
    ("眉", "brows"),
    ("眼睛", "eyes"),
    ("目", "eyes"),
    ("眼", "eyes"),
    ("嘴", "mouth"),
    ("口", "mouth"),
    ("鼻子", "nose"),
    ("鼻", "nose"),
    ("脸", "face"),
    ("顔", "face"),
    ("面", "face"),
    # 躯干与四肢
    ("身体", "body"),
    ("胴体", "body"),
    ("胴", "body"),
    ("体", "body"),
    ("手臂", "arms"),
    ("腕", "arms"),
    ("肘", "elbow"),
    ("手", "hands"),
    ("腿", "legs"),
    ("脚", "legs"),
    # 装饰与摆动
    ("辫子", "pigtail"),
    ("领带", "necktie"),
    ("头发", "hair"),
    ("髪", "hair"),
    ("摇动", "sway"),
    ("揺れ", "sway"),
    ("揺", "sway"),
    # 语义
    ("感情", "emotion"),
    ("表情", "expression"),
)

_DIRECTION_WORDS: tuple[tuple[str, str], ...] = (
    ("左", "left"),
    ("右", "right"),
    ("中", "center"),
    ("上", "upper"),
    ("下", "lower"),
    ("前", "front"),
    ("后", "back"),
)

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _ascii_slug(raw: str) -> str:
    return _SLUG_STRIP.sub("_", raw.lower()).strip("_")


def group_slug(label: str, fallback: str = "group") -> str:
    """分组名 → 英文 channel 名. 命中共享词表; 未命中时回退到 ascii 化, 再回退 fallback."""
    text = label or ""
    words: list[str] = []
    for needle, slug in _GROUP_WORDS:
        if needle in text:
            words.append(slug)
            break
    for needle, slug in _DIRECTION_WORDS:
        if needle in text:
            words.append(slug)
            break
    if words:
        return "_".join(words)
    slug = _ascii_slug(text)
    return slug or fallback


# ---------------------------------------------------------------------------
# 参数 id → python 命令名
# ---------------------------------------------------------------------------

_PARAM_PREFIX = re.compile(r"^(PARAM|Param)_?", re.IGNORECASE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NUMERIC_IDENT = re.compile(r"^p?\d+$")


def param_ident(param_id: str, label: str = "") -> str:
    """`ParamMouthOpenY` → `mouth_open_y`; `PARAM_EYE_L_OPEN` → `eye_l_open`.

    模型作者常给人看的参数起 `Param27` 这类无信息 id, 把语义留在 cdi3 的 `Name` 里
    (`大智慧`, `侧发摇动`). 遇到这种纯数字 id 时改从 `label` 译名; 译不出再回退到 id.
    """
    name = _PARAM_PREFIX.sub("", param_id)
    name = _CAMEL_BOUNDARY.sub("_", name)
    slug = _ascii_slug(name)
    if not slug:
        return "param"
    if slug[0].isdigit():
        slug = f"p{slug}"
    if _NUMERIC_IDENT.match(slug) and label:
        return ident_from_label(label) or slug
    return slug


# ---------------------------------------------------------------------------
# 参数名 (模型作者的语言) → 英文标识符
#
# 只处理 id 无信息时的那批参数. 分词方式: 在每个位置优先匹配最长的词表项, 匹配不到
# 的 ascii 片段原样收下. 译不出任何词时返回空串, 调用方回退.
# ---------------------------------------------------------------------------

_PARAM_WORDS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("目玉", "eyeball"), ("眼球", "eyeball"),
            ("眉毛", "brow"), ("眉", "brow"),
            ("眼睛", "eye"), ("眼", "eye"), ("目", "eye"),
            ("嘴巴", "mouth"), ("嘴", "mouth"), ("口", "mouth"),
            ("鼻子", "nose"), ("鼻", "nose"),
            ("耳朵", "ear"), ("耳", "ear"),
            ("刘海", "hair_front"), ("侧发", "hair_side"), ("头发", "hair"), ("髪", "hair"),
            ("裙子", "skirt"), ("裙", "skirt"),
            ("背手", "hand_back"),
            ("飘带", "ribbon"), ("领巾", "necktie"), ("领带", "necktie"), ("围巾", "scarf"),
            ("手臂", "arm"), ("胳膊", "arm"), ("腕", "arm"), ("臂", "arm"),
            ("手", "hand"), ("肘", "elbow"),
            ("足", "leg"), ("腿", "leg"), ("脚", "leg"),
            ("身体", "body"), ("胴体", "body"), ("腰", "body"), ("体", "body"),
            ("照れ", "blush"), ("脸红", "blush"), ("害羞", "shy"),
            ("眼镜", "glasses"), ("吃惊", "surprised"), ("惊", "surprised"),
            ("怒", "angry"), ("喜", "joy"), ("笑", "smile"),
            ("泣", "cry"), ("哭", "cry"),
            ("怖", "fear"), ("恐", "fear"),
            ("泪", "tear"),
            ("开闭", "open"), ("開閉", "open"), ("开合", "open"),
            ("摇动", "sway"), ("揺れ", "sway"), ("揺", "sway"), ("飘", "sway"),
            ("上下", "y"), ("左右", "x"), ("角度", "angle"), ("回転", "angle"), ("旋转", "angle"),
            ("呼吸", "breath"),
            ("左", "l"), ("右", "r"),
            ("上", "upper"), ("下", "lower"), ("前", "front"), ("后", "back"), ("中", "center"),
        ),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
)

_ASCII_RUN = re.compile(r"[A-Za-z0-9]+")


def ident_from_label(label: str) -> str:
    """`侧发摇动` → `hair_side_sway`; `鼻ちょうちんON` → `nose_on`. 译不出则返回 ''."""
    text = label or ""
    words: list[str] = []
    i = 0
    while i < len(text):
        matched = False
        for needle, slug in _PARAM_WORDS:
            if text.startswith(needle, i):
                words.append(slug)
                i += len(needle)
                matched = True
                break
        if matched:
            continue
        run = _ASCII_RUN.match(text, i)
        if run:
            words.append(run.group().lower())
            i = run.end()
            continue
        i += 1  # 词表外的单字 (如 ち) 跳过
    seen: list[str] = []
    for w in words:
        if w and w not in seen:
            seen.append(w)
    return "_".join(seen)


# ---------------------------------------------------------------------------
# 标准 Cubism 参数的量纲提示
#
# 只影响命令 docstring 里写给模型看的取值范围, 不参与任何校验 —— 真正的上下界由
# 模型自身的 moc3 决定, 页面侧按 Core 的 min/max 夹取. 未收录的参数按 (-1, 1).
# ---------------------------------------------------------------------------

PARAM_HINTS: dict[str, tuple[float, float, str]] = {
    "ParamAngleX": (-30, 30, "头部左右转 (负=左, 正=右)"),
    "ParamAngleY": (-30, 30, "头部上下转 (负=下, 正=上)"),
    "ParamAngleZ": (-30, 30, "头部侧倾"),
    "ParamEyeLOpen": (0, 1, "左眼睁开程度"),
    "ParamEyeROpen": (0, 1, "右眼睁开程度"),
    "ParamEyeBallX": (-1, 1, "眼球左右"),
    "ParamEyeBallY": (-1, 1, "眼球上下"),
    "ParamEyeBallForm": (-1, 1, "眼球形状"),
    "ParamBrowLY": (-1, 1, "左眉上下"),
    "ParamBrowRY": (-1, 1, "右眉上下"),
    "ParamBrowLX": (-1, 1, "左眉左右"),
    "ParamBrowRX": (-1, 1, "右眉左右"),
    "ParamBrowLAngle": (-1, 1, "左眉角度"),
    "ParamBrowRAngle": (-1, 1, "右眉角度"),
    "ParamBrowLForm": (-1, 1, "左眉形状"),
    "ParamBrowRForm": (-1, 1, "右眉形状"),
    "ParamMouthOpenY": (0, 1, "嘴巴张开程度"),
    "ParamMouthForm": (-1, 1, "嘴形 (负=撇嘴, 正=笑)"),
    "ParamBodyAngleX": (-10, 10, "身体左右倾"),
    "ParamBodyAngleY": (-10, 10, "身体上下倾"),
    "ParamBodyAngleZ": (-10, 10, "身体侧转"),
    "ParamBreath": (0, 1, "呼吸起伏"),
    "ParamArmLA": (-1, 1, "左臂抬起 (A 组)"),
    "ParamArmRA": (-1, 1, "右臂抬起 (A 组)"),
    "ParamArmLB": (-1, 1, "左臂 (B 组)"),
    "ParamArmRB": (-1, 1, "右臂 (B 组)"),
    "ParamHandL": (0, 1, "左手张合"),
    "ParamHandR": (0, 1, "右手张合"),
    "ParamHairFront": (-1, 1, "前发摆动"),
    "ParamHairSide": (-1, 1, "侧发摆动"),
    "ParamHairBack": (-1, 1, "后发摆动"),
    "ParamHairAhoge": (-1, 1, "呆毛"),
    "ParamTere": (0, 1, "害羞"),
    "ParamTear": (0, 1, "眼泪"),
    "ParamScarf": (-1, 1, "围巾摆动"),
}

DEFAULT_PARAM_RANGE: tuple[float, float] = (-1, 1)


def _canonical(param_id: str) -> str:
    """参数 id 归一化: 去掉 Param/PARAM 前缀、下划线、大小写差异.

    Cubism 同一个参数在不同模型里写 `ParamAngleX` 或 `PARAM_ANGLE_X` —— 量纲提示
    应该对两者都命中, 所以查表前先归一.
    """
    return _PARAM_PREFIX.sub("", param_id).replace("_", "").lower()


_PARAM_HINTS_CANONICAL: dict[str, tuple[float, float, str]] = {
    _canonical(k): v for k, v in PARAM_HINTS.items()
}


def param_hint(param_id: str) -> tuple[float, float, str]:
    """返回 (min, max, 说明). 未收录时给通用范围与空说明."""
    hint = _PARAM_HINTS_CANONICAL.get(_canonical(param_id))
    if hint is not None:
        return hint
    lo, hi = DEFAULT_PARAM_RANGE
    # 名字里带 Open/呼吸 一类通常是非负的; 保守起见不猜, 交给页面夹取.
    return lo, hi, ""


def param_doc(param_id: str) -> str:
    """给单个参数命令生成 docstring 的一行. 模型看的就是这句."""
    lo, hi, note = param_hint(param_id)
    range_text = f"取值 {lo:g} 到 {hi:g}"
    return f"{note}。{range_text}" if note else range_text
