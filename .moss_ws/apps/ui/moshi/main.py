"""Moshi — show_moshi 导演 App。

启动时扫描 assets/moshi_courses/ 下的可用课程，通过 context_messages
三层叠加自动推送给 Ghost：课程列表层始终可见 + 课程概况层（加载后）+
章节层（进入后）。Ghost 渐进式进入：先读课程列表 → 选课程 → 逐章推进。

课程数据通过标准 MOSS 资源体系（moshi-course scheme）管理：
优先从 ResourceRegistry/IoC 获取已注册的 CourseResourceStorage，
回退到直接构建（仅开发/测试环境）。

启动时附带原生桌面壳窗口（PySide6 + QWebEngineView），内嵌 reflex 前端，
替代浏览器。Qt 和 MOSS Matrix 通过 qasync 共享主线程的单一 asyncio 事件循环。

底部字幕条流式消费 matrix.session.get_logos()，在窗口中实时渲染 Ghost 输出。
"""

import asyncio
import sys

import qasync
from PySide6.QtWidgets import QApplication
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.message import Message

from course import Course, CourseMeta, load_course as _load_course
from src.course_storage import CourseResourceStorage
from src.window import MoshiWindow


async def _get_course_storage(matrix: Matrix) -> CourseResourceStorage:
    """获取 CourseResourceStorage 实例。

    标准路径：从 IoC 容器获取已注册实例（由 mode 的 resources.py 注册）。
    回退路径：直接构建（开发/测试环境，mode manifests 未加载时）。
    """
    # 标准路径：从 IoC 容器获取已注册的 CourseResourceStorage
    try:
        return matrix.container.force_fetch(CourseResourceStorage)
    except (KeyError, AttributeError):
        pass

    # 回退：直接构建（开发/测试环境，mode manifests 未加载时）
    assets_dir = matrix.workspace.assets().abspath() / "moshi_courses"
    storage = CourseResourceStorage(assets_dir)
    storage.scan()
    return storage


async def _stream_logos(matrix: Matrix, window: MoshiWindow) -> None:
    """后台消费 Ghost logos 流，喂入字幕条。"""
    import logging
    _log = logging.getLogger("moshi.logos")
    try:
        session = matrix.session
        sid = session.session_id
        key = session.stream_key_expr(f"logos/{sid}")
        window.subtitle.set_status(f"session: {sid[:12]}... | key: ...{key[-40:]}")
        async for delta in session.get_logos():
            window.subtitle.append_text(delta)
            window.subtitle.set_status("")  # 收到数据后清掉状态
    except asyncio.CancelledError:
        pass
    except Exception:
        _log.exception("logos stream error")
        window.subtitle.set_status("logos stream error")


async def _scan_from_storage(storage: CourseResourceStorage) -> dict[str, CourseMeta]:
    """从 CourseResourceStorage 扫描课程列表，返回 {name: CourseMeta}。"""
    infos = await storage.list_infos()
    result: dict[str, CourseMeta] = {}
    for info in infos:
        if info.chapter_id:
            continue  # 跳过章节级 meta，只看课程级
        result[info.course_name] = CourseMeta(
            title=info.title,
            chapter_count=info.chapter_count,
            duration=info.total_duration,
        )
    return result


async def main(matrix: Matrix):
    storage = await _get_course_storage(matrix)
    available = await _scan_from_storage(storage)

    # 运行时状态
    course: Course | None = None
    current_id: str = ""  # "" = 停在 _meta 层，未进入具体章节

    channel = new_channel(
        name="moshi",
        description=(
            "show_moshi 导演。自动列出可用课程，加载后管理章节推进。"
        ),
    )

    # ── context_messages ──────────────────────────────────────────────────
    #
    # 三层叠加，每层独立判断可见性：
    #
    #   Layer 1（课程列表）  — 全程可见。所有可用课程一览，进入具体课程后
    #                          标注当前选中的是哪个。
    #   Layer 2（课程概况）  — 加载课程后可见。含 meta 描述、表演纪律、章节
    #                          索引。进入章节后保留，不消失。
    #   Layer 3（当前章节）  — 进入章节后可见。含章节剧本全文 + 推进指令。
    #
    @channel.build.context_messages
    async def context_messages() -> list[Message]:
        messages: list[Message] = []

        # ═══ Layer 1: 课程列表 — 始终可见 ═══
        if available:
            lines = []
            for name, meta in available.items():
                marker = " ◀ 当前" if (course and name == _current_course_name(course, available)) else ""
                lines.append(
                    f"  {name}: {meta.title}（{meta.chapter_count}章, {meta.duration}）{marker}"
                )
            courses_str = "\n".join(lines)
        else:
            courses_str = "（无可用课程）"
        messages.append(
            Message.new("moshi_courses").with_content(
                f"【可用课程】\n{courses_str}"
            )
        )

        # ═══ Layer 2: 课程概况 — 加载课程后可见，进入章节后保留 ═══
        if course:
            chapters_summary = "\n".join(
                f"  {course.chapters[i].order}. {course.chapters[i].title}"
                f"（{i}, 布局 {course.chapters[i].suggested_layout}）"
                for i in course.ordered_ids
            )
            messages.append(
                Message.new("moshi_overview").with_content(
                    f"【当前课程】{course.title}\n"
                    f"【表演纪律】{course.performance}\n"
                    f"【章节索引】\n{chapters_summary}\n"
                    f"\n{course.knowledge}"
                )
            )

            # Layer 2b: _meta 层推进指令（有课程但未进入章节）
            if not current_id:
                messages.append(
                    Message.new("moshi_directive").with_content(
                        "【指令】你现在处于课程概述层。立即调用 "
                        "<apps.ui_moshi:next_chapter /> 进入第一章，开始表演。"
                        "不要停留、不要解释概述内容——直接推进。"
                    )
                )

        # ═══ Layer 3: 当前章节 — 进入章节后可见 ═══
        if course and current_id:
            chap = course.chapters[current_id]
            # 章节推进指令
            idx = course.ordered_ids.index(current_id)
            if idx + 1 < len(course.ordered_ids):
                next_id = course.ordered_ids[idx + 1]
                next_chap = course.chapters[next_id]
                directive = (
                    f"【指令】当前在第{chap.order}章。完成表演后，"
                    f"调用 <apps.ui_moshi:next_chapter /> 进入"
                    f"第{next_chap.order}章「{next_chap.title}」。"
                )
            else:
                directive = "【指令】当前在最后一章。完成表演后收束谢幕。"

            messages.append(
                Message.new("moshi_chapter").with_content(
                    f"第{chap.order}章「{chap.title}」（{chap.id}）\n"
                    f"主题：{chap.theme}\n"
                    f"建议布局：{chap.suggested_layout}\n"
                    f"时长：{chap.duration}\n"
                    f"\n{chap.content}\n"
                    f"\n{directive}"
                )
            )

        return messages

    # ── 命令 ──────────────────────────────────────────────────────────────

    @channel.build.command()
    async def load_course(name: str) -> Observe:
        """加载指定课程。name 为可用课程列表中的课程名。"""
        nonlocal course, current_id
        if name not in available:
            opts = ", ".join(available.keys())
            return Observe.new(f"未知课程 '{name}'。可用：{opts}")

        # 标准路径：通过 ResourceStorage 加载（非直接文件读取）
        course = await _load_course(storage, name)
        current_id = ""
        chaps = " → ".join(course.ordered_ids)
        return Observe.new(
            f"已加载「{course.title}」，共{len(course.ordered_ids)}章。\n"
            f"章节路径：{chaps}\n\n"
            f"现在立即调用 <apps.ui_moshi:next_chapter /> 进入第一章。"
            f"不要停留——直接推进。"
        )

    @channel.build.command()
    async def next_chapter() -> Observe:
        """推进到下一章。首次调用进入第一章。"""
        nonlocal current_id
        if not course:
            return Observe.new("尚未加载课程。请先 load_course。")
        if not current_id:
            current_id = course.ordered_ids[0]
        else:
            idx = course.ordered_ids.index(current_id)
            if idx + 1 >= len(course.ordered_ids):
                return Observe.new("已是最后一章。收束表演，准备谢幕。")
            current_id = course.ordered_ids[idx + 1]
        chap = course.chapters[current_id]
        return Observe.new(
            f"进入第{chap.order}章「{chap.title}」\n"
            f"布局：{chap.suggested_layout} | 时长：{chap.duration}\n"
            f"立即按剧本开始表演。"
        )

    @channel.build.command()
    async def jump_chapter(id: str) -> Observe:
        """跳转到指定章节。id 为章节标识符。"""
        nonlocal current_id
        if not course:
            return Observe.new("尚未加载课程。请先 load_course。")
        if id not in course.chapters:
            opts = ", ".join(course.ordered_ids)
            return Observe.new(f"未知章节 '{id}'。可用：{opts}")
        current_id = id
        chap = course.chapters[id]
        return Observe.new(
            f"跳转到第{chap.order}章「{chap.title}」\n"
            f"布局：{chap.suggested_layout} | 时长：{chap.duration}\n"
            f"立即按剧本开始表演。"
        )

    await matrix.provide_channel(channel)

    # 保持 channel 存活，直到 Matrix 关闭
    await matrix.wait_closed()


def _current_course_name(course: Course, available: dict[str, CourseMeta]) -> str:
    """反向查找当前加载的课程名（从 available 中匹配 title）。"""
    for name, meta in available.items():
        if meta.title == course.title:
            return name
    return ""


async def _run():
    app = QApplication.instance()
    window = MoshiWindow()
    window.show()
    window.subtitle.hide()

    matrix = Matrix.discover()
    app.aboutToQuit.connect(matrix.close)

    async def _combined(m: Matrix) -> None:
        await asyncio.gather(
            main(m),
            _stream_logos(m, window),
        )

    await matrix.arun(_combined)


if __name__ == "__main__":
    _ = QApplication(sys.argv)  # qasync 通过 QApplication.instance() 复用
    qasync.run(_run())
