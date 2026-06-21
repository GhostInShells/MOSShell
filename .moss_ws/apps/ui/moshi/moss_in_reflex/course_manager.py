"""CourseManager — 课程状态机与持久化。

不依赖 Reflex，可独立单元测试。
"""

from framework.helpers.layout_snapshot import SNAPSHOT_VERSION
from moss_in_reflex.course_storage import CourseInfo, CourseItem, CourseStorage


class CourseManager:
    """课程四阶段状态机 + 章节持久化。

    四阶段：idle → discussing → preparing → teaching
    """

    def __init__(self) -> None:
        self._mode: str = "idle"               # idle | discussing | preparing | teaching
        self._name: str = ""                    # 课程标识（目录名），如 "水调歌头"
        self._title: str = ""                   # 课程显示标题，如 "水调歌头·明月几时有"
        self._outline: list[str] = []           # 章节名有序列表
        self._chapters: dict[str, dict] = {}    # 内存章节数据 {章节名: data_dict}
        self._chapter_index: int = 0            # 当前章节序号

    # ── 只读属性 ──────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def name(self) -> str:
        return self._name

    @property
    def title(self) -> str:
        return self._title

    @property
    def outline(self) -> tuple[str, ...]:
        return tuple(self._outline)

    @property
    def saved_chapters(self) -> tuple[str, ...]:
        return tuple(self._chapters.keys())

    @property
    def chapter_index(self) -> int:
        return self._chapter_index

    @property
    def is_loaded(self) -> bool:
        return bool(self._chapters)

    # ── 纯状态转换（同步，无 IO）──────────────────────────────

    def start_prepare(self, course: str = "") -> str:
        """有参→设课名+切讨论大纲；无参→重置回空闲。"""
        if not course:
            self._name = ""
            self._title = ""
            self._outline = []
            self._chapters = {}
            self._mode = "idle"
            self._chapter_index = 0
            return "reset to idle"
        else:
            self._name = course
            self._title = ""
            self._outline = []
            self._chapters = {}
            self._mode = "discussing"
            self._chapter_index = 0
            return f"preparing {course}"

    def start_teaching(self, start_chapter: int = 0) -> str:
        """切到讲课模式。调用方负责 load + _switch_to_chapter。"""
        if not self._outline:
            raise ValueError("no outline set — call set_outline first")
        if start_chapter < 0 or start_chapter >= len(self._outline):
            raise ValueError(
                f"start_chapter {start_chapter} out of range (0-{len(self._outline) - 1})"
            )
        self._mode = "teaching"
        return f"teaching from chapter {start_chapter}: {self._outline[start_chapter]}"

    # ── 持久化操作（异步，需要 CourseStorage）─────────────────

    async def set_outline(
        self, course: str, title: str, chapters: str, storage: CourseStorage,
    ) -> str:
        """锁定大纲，写 meta.json，切备课模式。"""
        chapter_list = [c.strip() for c in chapters.split(",") if c.strip()]
        if not chapter_list:
            raise ValueError("chapters 不能为空")

        self._name = course
        self._title = title
        self._outline = chapter_list
        self._chapters = {}
        self._mode = "preparing"
        self._chapter_index = 0

        meta_data = {
            "version": SNAPSHOT_VERSION,
            "title": title,
            "chapters": chapter_list,
            "chapters_count": len(chapter_list),
        }
        info = CourseInfo(
            host="workspace-assets",
            path=f"{course}/meta",
            title=title,
            chapters_count=len(chapter_list),
        )
        item = CourseItem(info, meta_data)
        await storage.put(item)

        return f"meta://{course}"

    async def save_chapter(
        self,
        chapter: str,
        locator_list: list[str],
        snapshot: dict,
        storage: CourseStorage,
    ) -> str:
        """存档当前章节到 JSON 文件，更新内存进度。

        Args:
            chapter: 章节标识，必须在大纲中
            locator_list: Ghost 传入的图片 locator 列表
            snapshot: LayoutSnapshot.get_full() 返回的页面状态全量数据
            storage: 课程持久化存储
        """
        if not self._name:
            raise ValueError("no course set — call set_outline first")
        if chapter not in self._outline:
            raise ValueError(
                f"章节 '{chapter}' 不在大纲中。大纲：{', '.join(self._outline)}"
            )

        chapter_data = {
            "version": SNAPSHOT_VERSION,
            "sub_title": snapshot.get("sub_title", ""),
            "main_text": snapshot.get("main_text", ""),
            "annotations": snapshot.get("annotations", []),
            "appreciation": snapshot.get("appreciation", ""),
            "images": locator_list,
        }

        info = CourseInfo(host="workspace-assets", path=f"{self._name}/{chapter}")
        item = CourseItem(info, chapter_data)
        locator = await storage.put(item)

        self._chapters[chapter] = chapter_data

        return locator

    async def load_course(self, course: str, storage: CourseStorage) -> str:
        """从文件系统读 meta + 全部章节到内存，切备课模式。"""
        meta_item = await storage.get(f"{course}/meta")
        if meta_item is None:
            raise ValueError(f"课程 {course} 不存在")
        meta = await meta_item.get()
        chapter_list = meta.get("chapters", [])

        chapters: dict[str, dict] = {}
        for ch_name in chapter_list:
            ch_item = await storage.get(f"{course}/{ch_name}")
            if ch_item is not None:
                chapters[ch_name] = await ch_item.get()

        self._name = course
        self._title = meta.get("title", course)
        self._outline = chapter_list
        self._chapters = chapters
        self._mode = "preparing"
        self._chapter_index = len(chapters)

        loaded = len(chapters)
        pending = len(chapter_list) - loaded
        return f"loaded {loaded} chapters, {pending} pending"

    # ── 章节数据（供 _switch_to_chapter 构建 LoadChapterEvent）──

    def get_chapter_data(self, n: int) -> dict:
        """返回指定章节的数据字典。

        images 字段是 locator 字符串列表，由调用方解析为 PIL 对象。
        """
        ch = self._chapters[self._outline[n]]
        return {
            "title": ch.get("sub_title", self._title or self._name),
            "sub_title": ch.get("sub_title", ""),
            "main_text": ch.get("main_text", ""),
            "annotations": ch.get("annotations", []),
            "appreciation": ch.get("appreciation", ""),
            "images": list(ch.get("images", [])),
        }

    def set_chapter_index(self, n: int) -> None:
        """翻页后更新当前章节序号。"""
        self._chapter_index = n

    # ── 工具 ──────────────────────────────────────────────────

    @staticmethod
    def parse_locators(locators: str) -> list[str]:
        """逗号分隔的 locator 字符串 → 列表。"""
        if not locators.strip():
            return []
        return [l.strip() for l in locators.split(",") if l.strip()]
