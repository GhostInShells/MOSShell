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
    lines.append("")
    lines.append("### speaker_notes（演讲者笔记）")
    lines.append("存档每章时，除页面内容外，还需生成演讲者笔记。")
    lines.append("save_chapter 的 speaker_notes 参数为 JSON，结构：")
    lines.append('  {"talking_points": [{"id":"0","text":"开场","status":"pending"},...],')
    lines.append('   "transitions": ["接下来...","现在看..."],')
    lines.append('   "key_data": ["关键数据1"],')
    lines.append('   "estimated_duration": 120}')
    lines.append("talking_points 按叙述顺序排列，status 统一为 pending。")

    return "\n".join(lines)


def teaching(course_name: str, outline: list[str], chapter_index: int,
             speaker_notes: dict | None = None) -> str:
    total = len(outline)
    cur = outline[chapter_index] if 0 <= chapter_index < total else "（未知）"
    prev_ch = outline[chapter_index - 1] if chapter_index > 0 else "（无）"
    next_ch = outline[chapter_index + 1] if chapter_index + 1 < total else "（无）"

    lines = [
        f"## 当前状态：讲课",
        f"",
        f"课程：{course_name}",
        f"进度：第 {chapter_index + 1}/{total} 章 — {cur}",
        f"上一章：{prev_ch}",
        f"下一章：{next_ch}",
        f"",
    ]

    # ── 演讲者手卡（静态参考，实时进度见 lecture-state）──
    sn = speaker_notes or {}
    talking_points = sn.get("talking_points", [])
    if talking_points:
        # 过渡词
        transitions = sn.get("transitions", [])
        if transitions:
            lines.append("### 过渡词")
            for t in transitions:
                lines.append(f"  🔄 {t}")
            lines.append("")

        # 关键数据
        key_data = sn.get("key_data", [])
        if key_data:
            lines.append("### 关键数据")
            for kd in key_data:
                lines.append(f"  📊 {kd}")
            lines.append("")

        # 预计时长
        est = sn.get("estimated_duration", 0)
        if est:
            lines.append(f"预计时长：约 {est} 秒")
            lines.append("")

    lines.append(
        f"请按演讲要点逐段讲解。每讲完一个要点后调 <apps.ui_moshi:advance_point /> "
        f"标记完成，调完后**停止输出**，等待系统 Signal 唤醒后再继续下一段。"
        f"段落间隙调 <apps.ui_moshi:check_messages /> 检查飞书消息。"
    )
    lines.append("")
    lines.append("【CTML 规范】：")
    lines.append("  ❌ 禁止输出 <_> 或任何自创 CTML 标签——会破坏命令解析器")
    lines.append("  ❌ 禁止调用 chat_reply——讲课模式下没有聊天功能")
    lines.append("  ❌ 禁止自言自语或模拟对话——你是主讲人，不是聊天对象")
    lines.append("  ✓ 用纯文本叙述段落，只在推进时输出 <apps.ui_moshi:advance_point />")
    lines.append("  ✓ advance_point 必须是本轮最后一条命令——调完即停，等 Signal 再开新一轮")
    lines.append("  ✓ 当 advance_point 返回 lecture_ended 时，讲课已结束。停止输出，不要再调任何命令。")
    lines.append("")
    lines.append("【强约束】：")
    lines.append("  ❌ 不要修改/重新渲染页面内容——只在人类明确指出问题时才改")
    lines.append("  ❌ 不要跳到备课行为（除非人类说\"改一下\"）")
    lines.append("  ❌ 不要调 switch_state——布局已自动设为 lesson")
    lines.append("  ❌ 不要调 load_course——课程已加载完毕")
    lines.append("  ✓ 全部要点讲完后调 advance_point → 自动翻页或返回 lecture_ended，结束后不要再调命令")

    return "\n".join(lines)
