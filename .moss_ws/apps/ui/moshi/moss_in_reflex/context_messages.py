"""各阶段 context_messages 文案模板。

四个纯函数，不依赖任何模块状态变量——入参即所需信息。
moss_in_reflex.py 的 context_messages() 负责判断阶段、传入参数、拼 Message 对象。
"""


def idle() -> str:
    return (
        "## 当前状态：空闲\n\n"
        "等待人类指令。课程命令（start_prepare、set_outline、save_chapter、"
        "load_course、start_teaching）仅在人类明确要求时调用。"
    )


def discussing(course_name: str) -> str:
    return (
        f"## 当前状态：讨论大纲\n\n"
        f"课程：{course_name}\n"
        f"请与人类讨论章节结构和各章内容侧重。"
        f"必须在人类确认大纲后，调用 set_outline 锁定，这也是此阶段你唯一可用的指令。\n\n"
        f"约束：\n"
        f"  ❌ 不要操作 UI（不要 stream_title 等）\n"
        f"  ❌ 不要调用 save_chapter、switch_chapter\n"
        f"  ❌ 不要直接填内容——先定结构"
    )


def preparing(outline: list[str], saved_chapters: list[str]) -> str:
    saved = set(saved_chapters)
    total = len(outline)
    current_idx = len(saved_chapters)

    lines = [
        f"## 当前状态：{'备课完成' if len(saved) >= total else '备课'}",
        f"",
        f"课程：{total} 章",
    ]

    for i, ch in enumerate(outline):
        if ch in saved:
            lines.append(f"  ✓ {ch}")
        elif i == current_idx:
            lines.append(f"  → {ch}")
        else:
            lines.append(f"     {ch}")

    lines.append("")

    if len(saved) >= total:
        lines.append("全部完成 ✓  可以开始讲课。")
    else:
        lines.append(
            f"已存档 {len(saved)} 章，当前填充第 {current_idx + 1} 章。"
        )

    lines.append("")
    lines.append("约束：")
    lines.append("  ❌ 不要调用 switch_chapter（那是讲课命令）")
    lines.append("  ❌ 不要自己决定大纲——需要调整时先和人类讨论")
    lines.append("  ✓ 每章填完后等人类说\"好了，存档\"再调用 save_chapter")

    return "\n".join(lines)


def teaching(course_name: str, outline: list[str], chapter_index: int) -> str:
    total = len(outline)
    cur = outline[chapter_index] if 0 <= chapter_index < total else "（未知）"
    prev_ch = outline[chapter_index - 1] if chapter_index > 0 else "（无）"
    next_ch = outline[chapter_index + 1] if chapter_index + 1 < total else "（无）"

    return (
        f"## 当前状态：讲课\n\n"
        f"课程：{course_name}\n"
        f"进度：第 {chapter_index + 1}/{total} 章 — {cur}\n"
        f"上一章：{prev_ch}\n"
        f"下一章：{next_ch}\n\n"
        f"当页面上出现新章节内容时，主动开讲。讲解可以不局限于页面文字——"
        f"讲完本章后等待人类指令才能翻页。\n\n"
        f"【强约束】：\n"
        f"  ❌ 不要修改/重新渲染页面内容——只在人类明确指出问题时才改\n"
        f"  ❌ 不要跳到备课行为（除非人类说\"改一下\"）\n"
        f"  ✓ 等待人类说\"下一章\"\"回到上一章\"再翻页\n"
    )
