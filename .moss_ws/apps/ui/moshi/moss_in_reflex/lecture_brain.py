"""LectureBrain — 讲课状态机。

管理讲课流程的实时状态：段落推进、暂停/恢复、问题收集。
不依赖 Reflex，不持有 CourseManager 引用——翻页决策由命令侧根据返回值执行。
"""

import time
from dataclasses import dataclass, field


# ── 状态常量 ──

IDLE = "idle"
LOADING = "loading"
LECTURING = "lecturing"
PAUSED = "paused"
ENDED = "ended"

# ── advance_point 返回值 ──

POINT_ADVANCED = "point_advanced"      # 段内推进，继续讲
CHAPTER_ADVANCED = "chapter_advanced"  # 本章讲完，需翻页（命令侧处理）
LECTURE_ENDED = "lecture_ended"        # 全部讲完


@dataclass
class LectureBrain:
    """讲课状态机。

    idle → loading → lecturing ⇄ paused → ended

    核心职责：
    - 管理 talking_points 生命周期（pending → active → done）
    - 处理暂停/恢复（语音打断 + Ghost 回答后继续）
    - advance_point 自动判断翻页
    - 收集讲课过程中的问题
    """

    status: str = IDLE
    current_course: str = ""
    current_chapter: int = 0
    total_chapters: int = 0
    points: list[dict] = field(default_factory=list)       # talking_points（可变）
    transitions: list[str] = field(default_factory=list)
    key_data: list[str] = field(default_factory=list)
    estimated_duration: int = 0
    questions: list[dict] = field(default_factory=list)     # [{source, text, sender}]
    point_started_at: float = 0.0

    # ── 状态查询 ──

    @property
    def is_active(self) -> bool:
        return self.status in (LOADING, LECTURING, PAUSED)

    @property
    def active_point(self) -> dict | None:
        """当前正在讲的要点。"""
        for p in self.points:
            if p.get("status") == "active":
                return p
        return None

    @property
    def progress_summary(self) -> str:
        """段落进度摘要，供 context 展示。"""
        done = sum(1 for p in self.points if p.get("status") == "done")
        total = len(self.points)
        return f"{done}/{total}"

    # ── 状态转换 ──

    def start_loading(
        self,
        course: str,
        chapter: int,
        total_chapters: int,
        talking_points: list[dict],
        transitions: list[str] | None = None,
        key_data: list[str] | None = None,
        estimated_duration: int = 0,
    ) -> None:
        """进入 loading 状态。由 start_teaching 命令调用。

        talking_points 应为 [{id, text, status:"pending"}] 格式。
        首个要点自动标为 active。
        """
        self.status = LOADING
        self.current_course = course
        self.current_chapter = chapter
        self.total_chapters = total_chapters

        # 深拷贝 talking_points，统一 status
        self.points = [
            {"id": p.get("id", str(i)), "text": p.get("text", ""), "status": "pending"}
            for i, p in enumerate(talking_points)
        ]
        self.transitions = list(transitions or [])
        self.key_data = list(key_data or [])
        self.estimated_duration = estimated_duration
        self.questions = []

        # 首个要点激活
        if self.points:
            self.points[0]["status"] = "active"
            self.point_started_at = time.monotonic()

    def start_lecturing(self) -> None:
        """loading → lecturing。页面渲染完成后调用。"""
        if self.status != LOADING:
            raise RuntimeError(f"只能在 loading 状态调 start_lecturing，当前 {self.status}")
        self.status = LECTURING

    def pause(self, source: str = "voice", text: str = "", sender: str = "") -> None:
        """lecturing → paused。语音打断或人工控场。

        Args:
            source: "voice" | "manual"
            text: 问题文本
            sender: 提问者标识
        """
        if self.status not in (LECTURING, LOADING):
            return  # 不报错，幂等

        self.status = PAUSED
        if text.strip():
            self.questions.append({
                "source": source,
                "text": text.strip(),
                "sender": sender,
            })
        # 当前 active 段落保持 active，point_started_at 不动

    def resume(self) -> None:
        """paused → lecturing。Ghost 回答完问题后恢复讲课。"""
        if self.status != PAUSED:
            raise RuntimeError(f"只能在 paused 状态调 resume，当前 {self.status}")
        self.status = LECTURING
        # point_started_at 重置，给恢复后的段落全新计时
        self.point_started_at = time.monotonic()

    def end(self) -> str:
        """强制结束讲课 → ended。"""
        self.status = ENDED
        return LECTURE_ENDED

    # ── 段落推进 ──

    def advance_point(self) -> str:
        """标记当前 active → done，激活下一个 pending。

        Returns:
            "point_advanced"  — 段内推进，继续讲当前章
            "chapter_advanced" — 本章全部 done，需要翻页（命令侧负责）
            "lecture_ended"    — 全部章节讲完
        """
        if self.status not in (LECTURING, PAUSED):
            raise RuntimeError(f"只能在 lecturing/paused 状态调 advance_point，当前 {self.status}")

        # 标记当前 active → done
        for p in self.points:
            if p.get("status") == "active":
                p["status"] = "done"
                break

        # 找下一个 pending
        next_point = None
        for p in self.points:
            if p.get("status") == "pending":
                next_point = p
                break

        if next_point is not None:
            next_point["status"] = "active"
            self.point_started_at = time.monotonic()
            return POINT_ADVANCED

        # 本章全部 done
        return CHAPTER_ADVANCED

    # ── 翻页（命令侧调用） ──

    def advance_chapter(self, chapter: int, talking_points: list[dict],
                        transitions: list[str] | None = None,
                        key_data: list[str] | None = None,
                        estimated_duration: int = 0) -> str:
        """加载新章节的 talking_points。命令侧判断还有章节时调用。

        Returns:
            "chapter_advanced" — 新章节已加载
            "lecture_ended"    — 这是最后一章，讲课结束
        """
        self.current_chapter = chapter

        self.points = [
            {"id": p.get("id", str(i)), "text": p.get("text", ""), "status": "pending"}
            for i, p in enumerate(talking_points)
        ]
        self.transitions = list(transitions or [])
        self.key_data = list(key_data or [])
        self.estimated_duration = estimated_duration

        if self.points:
            self.points[0]["status"] = "active"
            self.point_started_at = time.monotonic()

        # 保持 lecturing 状态（pause 后翻页的场景：仍在 pause 中）
        if self.status == PAUSED:
            # 翻页后仍暂停，等 Ghost 调 resume
            pass
        else:
            self.status = LECTURING

        return CHAPTER_ADVANCED

    # ── 超时检测 ──

    def check_timeout(self) -> bool:
        """检查当前 active 段落是否超时。

        Returns True 如果超时（超过 estimated_duration * 1.5 且至少 30 秒）。
        命令侧根据返回值决定是否自动推进。
        """
        if self.status != LECTURING:
            return False
        if self.estimated_duration <= 0:
            return False
        if self.point_started_at <= 0:
            return False

        elapsed = time.monotonic() - self.point_started_at
        timeout = max(self.estimated_duration * 1.5, 30)
        return elapsed > timeout

    # ── 问题收集 ──

    def add_question(self, source: str, text: str, sender: str = "") -> None:
        """添加问题记录（用于结束后总结）。"""
        self.questions.append({
            "source": source,
            "text": text,
            "sender": sender,
        })

    def summary(self) -> str:
        """生成讲课总结文本。"""
        if not self.questions:
            return "本次讲课无提问记录。"

        voice_qs = [q for q in self.questions if q.get("source") == "voice"]
        feishu_qs = [q for q in self.questions if q.get("source") == "feishu"]

        lines = [f"## 讲课总结\n"]
        lines.append(f"课程：{self.current_course}")
        lines.append(f"共收到 {len(self.questions)} 个问题")
        lines.append(f"  - 语音打断提问：{len(voice_qs)} 个")
        lines.append(f"  - 飞书群提问：{len(feishu_qs)} 个")
        lines.append("")

        if voice_qs:
            lines.append("### 语音提问")
            for q in voice_qs:
                lines.append(f"  - {q.get('sender', '匿名')}: {q.get('text', '')}")
            lines.append("")

        if feishu_qs:
            lines.append("### 飞书群提问")
            for q in feishu_qs:
                lines.append(f"  - {q.get('sender', '匿名')}: {q.get('text', '')}")
            lines.append("")

        return "\n".join(lines)
