import asyncio
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path

from ghoshell_common.contracts import YamlConfig, WorkspaceConfigs, DefaultFileStorage
from pydantic import Field

from framework.helpers.layout_snapshot import LayoutSnapshot
from framework.runtime.event_generator import build
from moss_in_reflex.course_manager import CourseManager
from moss_in_reflex.lecture_brain import LectureBrain

# =========================== Logger ===========================
logger = logging.getLogger("moss")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [moss] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.propagate = False


# =========================== Config & Layouts ===========================

class Config(YamlConfig):
    relative_path = "config.yaml"
    layouts: list[str] = Field(default_factory=list, description="List of layout names")


CONFIG = WorkspaceConfigs(
    DefaultFileStorage(dir_=str(Path(__file__).parent.absolute()))
).get_or_create(Config())


def load_layouts() -> list:
    """从 layouts.toml 动态加载 Layout 类。"""
    layouts = []
    for module_path in CONFIG.layouts:
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        layouts.append(cls)
        logger.info(f"Loaded layout: {module_path}")
    return layouts


# 外部系统和Reflex系统通信媒介
QUEUE: asyncio.Queue = asyncio.Queue()  # type: ignore[valid-type]

# 注册可用的LAYOUT（从 layouts.toml 动态加载）
LAYOUTS = load_layouts()

# 动态生成LAYOUT运行时
LAYOUT_COMPONENTS = [(l.name(), l.create()) for l in LAYOUTS]
LAYOUT_CHANNEL_STATES = []

for i, _l in enumerate(LAYOUTS):
    _n, _lc = LAYOUT_COMPONENTS[i]
    LAYOUT_CHANNEL_STATES.append(build(_n, _lc.State, _l, QUEUE))


# =========================== Layout Mirror ===========================

@dataclass
class _LayoutMirror:
    """Layout 状态镜像 — Reflex State 的模块级只读缓存。

    self.layout 是权威数据源（持久化，hot-reload 存活），_LAYOUT 是模块级镜像。
    moss_listener 同步写，context_messages() 只读。
    """
    name: str = "simple"

    def get_component(self):
        for n, c in LAYOUT_COMPONENTS:
            if n == self.name:
                return c
        return LAYOUT_COMPONENTS[0][1]


_LAYOUT = _LayoutMirror()

# Layout 状态快照 — moss_listener 循环末尾一把读，context_messages() 消费
_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
_SNAPSHOTS: dict[str, LayoutSnapshot] = {
    name: LayoutSnapshot(name, component.State, _SNAPSHOT_DIR)
    for name, component in LAYOUT_COMPONENTS
}


# =========================== 课程 & 讲课状态 ===========================

_COURSE_MGR = CourseManager()
_LECTURE_BRAIN = LectureBrain()


# =========================== SSE 队列 ===========================

# 聊天 SSE
_SSE_EVENT: asyncio.Event = asyncio.Event()
_SSE_QUEUE: list[dict] = []
_SSE_LOCK: asyncio.Lock = asyncio.Lock()

# 字幕 SSE
_SUBTITLE_EVENT: asyncio.Event = asyncio.Event()
_SUBTITLE_QUEUE: list[dict] = []
_SUBTITLE_LOCK: asyncio.Lock = asyncio.Lock()

# 弹幕 SSE
_DANMAKU_EVENT: asyncio.Event = asyncio.Event()
_DANMAKU_QUEUE: list[dict] = []
_DANMAKU_LOCK: asyncio.Lock = asyncio.Lock()
