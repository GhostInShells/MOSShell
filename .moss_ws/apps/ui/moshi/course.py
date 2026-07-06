"""课程数据结构与加载逻辑。

数据模型（纯 dataclass，无外部依赖）：
- CourseMeta: 启动扫描时的轻量元信息
- Chapter: 单个章节的完整数据
- Course: 完整课程，含所有章节

加载方式：
- scan_courses(): 从文件系统快速扫描课程列表（不加载章节内容）
- load_course(): 从 ResourceStorage 加载完整课程数据（标准路径）
"""

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chapter:
    """单个章节数据。"""
    id: str
    order: int
    title: str
    theme: str
    suggested_layout: str
    duration: str
    content: str  # chapter md body（已剥离 YAML frontmatter）


@dataclass
class CourseMeta:
    """启动时扫描的轻量元信息，不加载章节内容。"""
    title: str
    chapter_count: int
    duration: str


@dataclass
class Course:
    """完整课程数据。"""
    title: str
    performance: str       # 格式化后的表演纪律
    knowledge: str         # .meta.md body（AIOS 知识背景）
    chapters: dict[str, Chapter]
    ordered_ids: list[str]


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """分离 YAML frontmatter 和 markdown body。"""
    text = text.strip()
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    return yaml.safe_load(parts[1]) or {}, parts[2].strip()


# ── 文件系统扫描（轻量，仅返回 CourseMeta，不加载章节内容）──────────────────

async def scan_courses(assets_dir: Path) -> dict[str, CourseMeta]:
    """扫描可用课程目录，返回 {name: CourseMeta}。不加载章节内容。

    章节计数从目录下 0*.md 文件数量获取（每个文件头部有 YAML frontmatter）。
    """
    result: dict[str, CourseMeta] = {}
    if not assets_dir.exists():
        return result

    for d in sorted(assets_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / ".meta.md"
        if not meta_file.exists():
            continue

        fm, _ = _split_frontmatter(meta_file.read_text())
        # 扫描章节文件（0*.md），从各文件 frontmatter 验证
        chapter_files = sorted(d.glob("0*.md"))
        chapters_found = 0
        for ch_file in chapter_files:
            ch_fm, _ = _split_frontmatter(ch_file.read_text())
            if ch_fm.get("id"):
                chapters_found += 1

        result[d.name] = CourseMeta(
            title=fm.get("description", d.name),
            chapter_count=chapters_found,
            duration=fm.get("total_duration", "?"),
        )

    return result


# ── 标准资源加载（从 ResourceStorage，完整 Course）──────────────────────────

async def load_course(storage, name: str) -> Course:
    """从资源存储加载完整课程数据。

    storage 需提供 list_infos() 和 get(path) 方法（即 ResourceStorage 接口）。
    标准路径：通过 CourseResourceStorage 获取，而非直接读文件。

    返回包含所有章节内容、表演纪律、知识背景的 Course 对象。
    """
    # 1. 获取所有资源元信息
    all_infos = await storage.list_infos()

    # 2. 筛选该课程的所有 infos（课程级 + 章节级）
    course_infos = [i for i in all_infos if i.course_name == name]
    if not course_infos:
        raise ValueError(f"课程 '{name}' 不存在")

    # 3. 分离课程级 meta（chapter_id 为空）和章节级 infos
    course_meta = None
    chapter_infos = []
    for info in course_infos:
        if info.chapter_id:
            chapter_infos.append(info)
        else:
            course_meta = info

    if course_meta is None:
        raise ValueError(f"课程 '{name}' 缺少 _meta 信息")

    # 4. 按 order 排序章节
    chapter_infos.sort(key=lambda i: i.order)

    # 5. 构建 Chapter 对象（从 storage 获取完整内容）
    chapters: dict[str, Chapter] = {}
    for info in chapter_infos:
        # 通过 storage 标准接口获取章节文件内容
        item = await storage.get(info.path)
        raw_content = await item.get() if item else ""

        # 剥离 YAML frontmatter，保留 markdown body
        _, body = _split_frontmatter(raw_content)

        chapters[info.chapter_id] = Chapter(
            id=info.chapter_id,
            order=info.order,
            title=info.title,
            theme=info.theme,
            suggested_layout=info.suggested_layout,
            duration=info.duration,
            content=body.strip(),
        )

    # 6. 构建表演纪律字符串
    performance = (
        f"节奏：{course_meta.performance_rhythm}；"
        f"连续性：{course_meta.performance_continuity}；"
        f"容错：{course_meta.performance_fallback}"
    )

    # 7. 组装 Course
    ordered = sorted(chapters.values(), key=lambda c: c.order)

    return Course(
        title=course_meta.title,
        performance=performance,
        knowledge=course_meta.knowledge,
        chapters=chapters,
        ordered_ids=[c.id for c in ordered],
    )
