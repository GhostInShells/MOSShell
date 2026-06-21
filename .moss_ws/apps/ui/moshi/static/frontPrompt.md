# AI 原型图生成提示词

> 每条提示词均为**独立完整版本**，可直接复制到 Midjourney / DALL-E / ComfyUI 运行。
> 全局风格：深色主题（#1a1a2e）、中文无衬线字体、16:9 横屏、极简沉浸、产品截图风格。
> 每条提供 English 和中文两个版本，任选其一使用。

---

## B1. 讲课 L0 / L1 — 全屏舞台 + 字幕 + 弹幕

> L1 悬浮控制点为隐形热区，静态截图不可见，与 L0 共用同一提示词。

**English：**

```
A clean, immersive fullscreen lecture UI for a live teaching app, product screenshot style. The entire screen is a whiteboard canvas filling 100% of the 16:9 viewport — absolutely no toolbar, no sidebar, no chat panel, no menus. At the bottom of the screen, a semi-transparent dark gradient overlay bar displays live speech-to-text subtitles in large white sans-serif Chinese characters (e.g. "今天我们讨论三个核心问题"). In the middle area, multiple lines of colorful danmaku bullet comments drift from right to left at varying speeds, showing real audience messages like "这个讲得好！" in green, "能解释下这个概念吗？" in yellow, "666" in white. In the bottom-right corner, a single small circular "✕" exit button — 48px diameter, dark semi-transparent background, the icon subtly visible, designed to fade to 30% opacity after inactivity. The overall feel: a stage, not a software tool — the presenter stands beside the screen, the content fills every pixel, the interface disappears. Dark theme background (#1a1a2e), clean Chinese typography (PingFang SC style). Minimal, immersive, professional. --ar 16:9 --v 6
```

**中文版：**

```
一个简洁、沉浸的全屏讲课模式 UI，产品截图风格。整个 16:9 屏幕被一个白板画布 100% 填满——完全没有工具栏、没有侧边栏、没有聊天面板、没有菜单。屏幕底部有一条半透明深色渐变遮罩条，上面显示大号白色无衬线中文字体的实时语音转文字字幕（如"今天我们讨论三个核心问题"）。屏幕中间区域有多行彩色弹幕以不同速度从右向左飘过，内容为真实观众消息：绿色"这个讲得好！"、黄色"能解释下这个概念吗？"、白色"666"。右下角有一个极小的圆形"✕"退出按钮——直径 48px，深色半透明背景，图标若隐若现，无操作 3 秒后淡化为 30% 不透明度。整体感觉：不是软件工具，是舞台——讲师站在屏幕旁，内容填满每一个像素，界面消失了。深色主题背景（#1a1a2e），干净的中文排版（PingFang SC 风格）。极简、沉浸、专业。--ar 16:9 --v 6
```

---

## B2. 讲课 L2 — 全屏舞台 + 字幕 + 弹幕 + 讲师后台面板

**English：**

```
A clean, immersive fullscreen lecture UI for a live teaching app, product screenshot style. The entire screen is a whiteboard canvas filling 100% of the 16:9 viewport — no toolbar, no sidebar, no chat panel. At the bottom, a semi-transparent dark gradient overlay bar displays live speech-to-text subtitles in large white sans-serif Chinese characters. In the middle area, multiple lines of colorful danmaku bullet comments drift from right to left with audience messages like "这个讲得好！" and "能解释下这个概念吗？". In the bottom-right corner, a small circular "✕" exit button (48px, dark semi-transparent). NEW ELEMENT — in the bottom-left corner, a small compact dark semi-transparent card panel (about 280x200px) serving as a presenter backend monitor. The card has a title row "🔥 热点问题" at top with a small fire emoji. Below it, a list of 3-5 clustered audience questions, each showing the question text in white (e.g. "AI 能替代讲师吗？") and a heat number badge in orange on the right (e.g. "23", "18", "12", "9", "5"). At the bottom of the card, a green status indicator dot with text "🤖 自动回复已开启". The overall feel: same immersive stage as the base lecture mode, but with a subtle ops panel for the presenter to monitor audience sentiment. Dark theme (#1a1a2e), clean Chinese typography, 16:9. --ar 16:9 --v 6
```

**中文版：**

```
一个简洁、沉浸的全屏讲课模式 UI，产品截图风格。整个 16:9 屏幕被白板画布 100% 填满——没有工具栏、没有侧边栏、没有聊天面板。屏幕底部有一条半透明深色渐变遮罩条，显示大号白色无衬线中文字体的实时语音转文字字幕。中间区域有多行彩色弹幕以不同速度从右向左飘过，内容为观众消息。右下角有一个极小的圆形"✕"退出按钮（48px，深色半透明）。新增元素——屏幕左下角有一个紧凑的深色半透明卡片面板（约 280x200px），作为讲师后台监控面板。卡片顶部标题行"🔥 热点问题"带有小火苗图标。下方是 3-5 条聚类后的观众提问列表，每条左侧显示白色提问文字（如"AI 能替代讲师吗？"），右侧显示橙色热度数字徽章（如"23"、"18"、"12"、"9"、"5"）。卡片底部有一个绿色状态指示灯和文字"🤖 自动回复已开启"。整体感觉：与基础讲课模式相同的沉浸舞台感，但增加了一个低调的后台面板供讲师监控观众情绪。深色主题（#1a1a2e），干净的中文排版，16:9 横屏。--ar 16:9 --v 6
```

---

## B3. 练习 L0 — 主屏 + 演讲者手卡 + 计时器

**English：**

```
A clean practice mode UI for a presentation rehearsal app, product screenshot style. The full 16:9 screen shows a whiteboard/slide canvas filling the entire viewport — this is what the audience would see. There is absolutely no toolbar, no chat panel, no sidebar, no menus. In the bottom-right corner, a small floating dark semi-transparent card (about 300x160px) labeled "📝 演讲者手卡" (Speaker Cue Card) — this is visible only to the presenter, not the audience. The card shows three lines of text in white Chinese characters: Line 1 "📌 要点: 回顾核心论点——杠杆率决定天花板", Line 2 "🔄 过渡: 接下来我们看第三个案例...", Line 3 "📊 关键数据: 同比增长 37%". Prominently displayed in the center-bottom area is a large digital timer "⏱ 03:24" in white monospace font on a dark semi-transparent pill-shaped background. No webcam, no camera feed, no picture-in-picture. The feel: a private rehearsal booth — just the content, the cue cards to prevent forgetting lines, and the timer. Dark theme (#1a1a2e), clean Chinese typography, 16:9. --ar 16:9 --v 6
```

**中文版：**

```
一个简洁的练习模式 UI，用于演讲排练应用，产品截图风格。整个 16:9 屏幕展示白板/幻灯片画布填满视口——这是听众看到的画面。完全没有工具栏、没有聊天面板、没有侧边栏、没有菜单。右下角有一个浮动的小型深色半透明卡片（约 300x160px），标注"📝 演讲者手卡"——仅讲师可见，听众看不到。卡片显示三行白色中文文字：第一行"📌 要点: 回顾核心论点——杠杆率决定天花板"，第二行"🔄 过渡: 接下来我们看第三个案例..."，第三行"📊 关键数据: 同比增长 37%"。屏幕中央偏下位置醒目地显示一个大号数字计时器"⏱ 03:24"，白色等宽字体，深色半透明胶囊形背景。没有摄像头画面、没有 PIP 画中画。整体感觉：一间私人排练室——只有内容、防忘词的手卡、以及计时器。深色主题（#1a1a2e），干净的中文排版，16:9 横屏。--ar 16:9 --v 6
```

---

## B4. 练习 L1 — 主屏 + 手卡 + 计时器 + AI 弹幕 + 压力滑块

**English：**

```
A clean practice mode UI for a presentation rehearsal app, product screenshot style. The full 16:9 screen shows a whiteboard/slide canvas filling the viewport (audience view). In the bottom-right corner, a small floating dark semi-transparent speaker cue card labeled "📝 演讲者手卡" showing three lines: "📌 要点: 回顾核心论点", "🔄 过渡: 接下来我们看第三个案例...", "📊 关键数据: 同比增长 37%". In the center-bottom area, a large digital timer "⏱ 03:24" in white monospace font on a dark pill-shaped background. NEW ELEMENTS — ① A danmaku layer in the middle area: AI-generated simulated audience comments drift right-to-left at varying speeds. Examples: a red skeptical comment "这个逻辑有问题吧？", a green encouraging comment "讲得不错！", a yellow question "能举个实际例子吗？", a white neutral "666". ② At the very bottom of the screen, a pressure level slider control bar: a horizontal bar spanning about 60% of the screen width, left label "低压" in green, right label "高压" in red, the track showing a green-to-red gradient, with a circular thumb indicator positioned at the current level. Above the slider, a small label "压力等级 Lv.2". The speaker cue card and timer remain visible. No webcam, no PIP. Dark theme (#1a1a2e), clean Chinese typography, 16:9. --ar 16:9 --v 6
```

**中文版：**

```
一个简洁的练习模式 UI，用于演讲排练应用，产品截图风格。整个 16:9 屏幕展示白板/幻灯片画布填满视口（听众视角）。右下角有一个浮动的小型深色半透明演讲者手卡，标注"📝 演讲者手卡"，三行内容："📌 要点: 回顾核心论点"、"🔄 过渡: 接下来我们看第三个案例..."、"📊 关键数据: 同比增长 37%"。屏幕中央偏下位置有大号数字计时器"⏱ 03:24"，白色等宽字体，深色胶囊形背景。新增元素——① 屏幕中间弹幕层：AI 生成的模拟观众弹幕以不同速度从右向左飘过。示例：红色质疑"这个逻辑有问题吧？"，绿色鼓励"讲得不错！"，黄色提问"能举个实际例子吗？"，白色中性"666"。② 屏幕最底部增加压力等级滑块控制条：一条横跨屏幕约 60% 宽度的水平滑轨，左侧绿色标签"低压"，右侧红色标签"高压"，轨道呈绿到红渐变，圆形滑块指示器停在当前档位。滑块上方有小字标注"压力等级 Lv.2"。手卡和计时器保持可见。没有摄像头、没有 PIP。深色主题（#1a1a2e），干净的中文排版，16:9 横屏。--ar 16:9 --v 6
```

---

## B5. 练习 L2 — 复盘评分卡（练习结束后的界面）

**English：**

```
A post-practice review scorecard UI for a presentation rehearsal app, product screenshot style. This is a fullscreen overlay that appears after the practice session ends. The background is a dark semi-transparent overlay (#1a1a2e at 95% opacity) covering the entire 16:9 viewport. At the top center, a title "🎯 练习复盘" in large white Chinese characters. Below it, centered prominently, a four-dimension radar/spider chart with labeled axes: 清晰度 (Clarity), 节奏 (Pacing), 互动 (Engagement), 内容 (Content). Each axis has a score number (e.g. 85, 72, 90, 78) and the filled area forms an irregular polygon in a semi-transparent blue. Below the radar chart, a horizontal timeline bar spanning about 80% of screen width, labeled "⏱ 时间轴". The timeline has colored markers: green dots at smooth sections, red dots at danmaku spike moments (labeled "弹幕密集"), amber warning dots at weak segments (labeled "建议改进"). Each marker has a small timestamp below it (e.g. 01:23, 05:47). At the bottom of the screen, two action buttons side by side: a primary blue button "✏️ 去改课" (Go Edit Course) on the left with a bold style, and a secondary gray button "🔄 再练一次" (Practice Again) on the right with an outlined style. The overall feel: a coach's data-driven feedback report — friendly but professional. Dark theme, Chinese typography, 16:9. --ar 16:9 --v 6
```

**中文版：**

```
一个练习结束后的复盘评分卡 UI，用于演讲排练应用，产品截图风格。这是一个全屏覆盖层，在练习 session 结束后弹出。背景为深色半透明遮罩（#1a1a2e，95% 不透明度）覆盖整个 16:9 视口。顶部居中显示大号白色中文标题"🎯 练习复盘"。其下方居中位置是一个四维雷达图/蛛网图，坐标轴标注：清晰度、节奏、互动、内容，每个轴上有分数数字（如 85、72、90、78），填充区域形成不规则多边形，使用半透明蓝色。雷达图下方是一条横跨屏幕约 80% 宽度的水平时间轴，标注"⏱ 时间轴"。时间轴上有彩色标记点：绿色圆点标注流畅段落，红色圆点标注弹幕密集时刻（旁注"弹幕密集"），琥珀色警告点标注建议改进段落（旁注"建议改进"）。每个标记点下方有小字时间戳（如 01:23、05:47）。屏幕底部有两个并排按钮：左侧主按钮"✏️ 去改课"，蓝色填充，粗体样式；右侧次按钮"🔄 再练一次"，灰色描边样式。整体感觉：一份教练视角的数据驱动反馈报告——友好但专业。深色主题，中文排版，16:9 横屏。--ar 16:9 --v 6
```

---

## B6. 备课 L0 — 主屏 + AI 对话框 + 保存/退出

**English：**

```
A clean content editing mode UI for a course authoring app, product screenshot style. The 16:9 layout has three zones. Zone 1 — a very slim top bar (about 40px height): on the left, three mode tab labels in white Chinese text "备课" (highlighted/active, with a subtle blue underline), "练习", "讲课"; on the right, two buttons — "💾 保存" (Save, subtle blue background) and "✕ 退出" (Exit, dark gray). Zone 2 — the central editing area fills most of the remaining space, showing a whiteboard canvas with sample slide content: a title in large bold Chinese characters "产品增长策略", a subtitle, a bullet list of 3 items, and a placeholder image area on the right side. Zone 3 — a collapsible AI chat panel at the bottom (about 180px height): the panel has a dark background slightly lighter than the main theme, showing a conversation bubble from AI "需要我帮你调整第三章的结构吗？" in white text on a blue-tinted bubble, and below it a text input field with placeholder text "输入你的问题..." and a "发送" (Send) button on the right. No left sidebar, no right panel, no discussion area, no toolbar menus. The feel: a minimal writing workshop — just the content, an AI assistant in the bottom panel, and save/exit controls at top. Dark theme (#1a1a2e), Chinese typography, 16:9. --ar 16:9 --v 6
```

**中文版：**

```
一个简洁的内容编辑模式 UI，用于课程创作应用，产品截图风格。16:9 布局分为三区。区域一——顶部极窄栏（约 40px 高）：左侧三个白色中文模式标签"备课"（高亮/激活态，带淡蓝色下划线）、"练习"、"讲课"；右侧两个按钮——"💾 保存"（淡蓝色背景）和"✕ 退出"（深灰色）。区域二——中央编辑区填满大部分剩余空间，展示白板画布上的示例幻灯片内容：大号粗体中文标题"产品增长策略"、一行副标题、三条要点列表、右侧一个图片占位区域。区域三——底部可折叠 AI 聊天面板（约 180px 高）：面板背景为比主背景稍亮的深色，显示一条 AI 对话气泡"需要我帮你调整第三章的结构吗？"，白色文字，蓝色调气泡背景；下方是文本输入框，占位文字"输入你的问题..."，右侧"发送"按钮。没有左侧边栏、没有右侧面板、没有讨论区、没有工具栏菜单。整体感觉：一个极简的写作工坊——只有内容、底部的 AI 助手、以及顶部的保存和退出。深色主题（#1a1a2e），中文排版，16:9 横屏。--ar 16:9 --v 6
```

---

## B7. 备课 L1 — 主屏 + 左侧大纲 + AI 对话框 + 保存

**English：**

```
A clean content editing mode UI for a course authoring app, product screenshot style. The 16:9 layout has four zones. Zone 1 — slim top bar (40px): left side mode tabs "备课" (active, blue underline), "练习", "讲课"; right side "💾 保存" and "✕ 退出" buttons. Zone 2 — NEW left sidebar: a dark panel (about 220px width) showing a course outline. The sidebar header says "📑 课程大纲" in white. Below it, a vertical list of chapters: "第一章 引言" with a filled green progress dot (completed), "第二章 市场分析" with a filled green dot, "第三章 增长策略" highlighted with a blue background and a half-filled dot (in progress), "第四章 案例研究" with an empty gray dot (pending), "第五章 总结" with an empty gray dot. Each chapter row has a small drag handle icon on the right. The right edge of the sidebar has a subtle vertical divider line with a small collapse arrow "◀" button. Zone 3 — central editing area (narrowed to accommodate sidebar), showing a whiteboard canvas with slide content. Zone 4 — bottom AI chat panel (same as L0). Dark theme (#1a1a2e), Chinese typography, 16:9. --ar 16:9 --v 6
```

**中文版：**

```
一个简洁的内容编辑模式 UI，用于课程创作应用，产品截图风格。16:9 布局分为四区。区域一——顶部极窄栏（40px）：左侧模式标签"备课"（激活态，蓝色下划线）、"练习"、"讲课"；右侧"💾 保存"和"✕ 退出"按钮。区域二——新增左侧大纲侧边栏：深色面板（约 220px 宽），顶部白色标题"📑 课程大纲"。下方是纵向章节列表："第一章 引言"带绿色实心进度圆点（已完成），"第二章 市场分析"带绿色实心点，"第三章 增长策略"以蓝色背景高亮，带半填充圆点（进行中），"第四章 案例研究"带灰色空心圆点（待完成），"第五章 总结"带灰色空心点。每行右侧有一个小型拖拽手柄图标。侧边栏右边缘有一条细微的纵向分隔线，带有一个小折叠箭头"◀"按钮。区域三——中央编辑区（因侧边栏而收窄），展示白板画布上的幻灯片内容。区域四——底部 AI 聊天面板（与 L0 相同）。深色主题（#1a1a2e），中文排版，16:9 横屏。--ar 16:9 --v 6
```

---

## B8. 备课 L2 — 主屏 + 大纲 + 素材导入 + AI 蒸馏 + 保存

**English：**

```
A clean content editing mode UI for a course authoring app, product screenshot style. The 16:9 layout has all L1 elements plus two new additions. Zone 1 — slim top bar with mode tabs and save/exit buttons. Zone 2 — left outline sidebar with chapter list and progress dots (same as L1). Zone 3 — NEW element: a file drop zone at the top of the editing area. A dashed-border rectangular area spanning the width of the editing panel, with a central cloud-upload icon, primary text "📁 拖拽文件到此处导入" in white, secondary text "支持 PDF、图片、网页链接" in smaller gray text below, and a "浏览文件" (Browse) button on the right side. Below the drop zone, the whiteboard canvas shows slide content. Zone 4 — ENHANCED bottom AI chat panel (taller than L1, about 260px). Inside the panel: a progress indicator showing "🔍 正在蒸馏素材..." with a thin blue progress bar at 60%, below it a row of three story template cards — "🏆 英雄之旅" (selected, blue border), "💡 颠覆认知" (default), "📖 案例驱动" (default) — each as a small rounded card with title and brief description. Below the templates, a horizontal difficulty slider labeled "知识点深浅: 浅 ← → 深" with a circular thumb at the middle position. Below that, the standard text input field with "发送" button. Dark theme (#1a1a2e), Chinese typography, 16:9. --ar 16:9 --v 6
```

**中文版：**

```
一个简洁的内容编辑模式 UI，用于课程创作应用，产品截图风格。16:9 布局包含所有 L1 元素外加两个新增。区域一——顶部极窄栏，模式标签和保存/退出按钮。区域二——左侧大纲侧边栏，章节列表和进度圆点（与 L1 相同）。区域三——新增元素：编辑区顶部出现一个文件拖拽区。一个虚线边框矩形区域横跨编辑面板宽度，中央有一个云上传图标，主要文字"📁 拖拽文件到此处导入"为白色，下方较小的灰色辅助文字"支持 PDF、图片、网页链接"，右侧有一个"浏览文件"按钮。拖拽区下方是白板画布展示幻灯片内容。区域四——增强版底部 AI 聊天面板（比 L1 更高，约 260px）。面板内从上到下依次为：进度指示器显示"🔍 正在蒸馏素材..."，带一条细蓝色进度条（60%）；下方一排三个故事线模板卡片——"🏆 英雄之旅"（已选中，蓝色边框）、"💡 颠覆认知"（默认态）、"📖 案例驱动"（默认态）——每张为小型圆角卡片，包含标题和简短描述。模板下方是一条水平深浅滑块，标注"知识点深浅: 浅 ← → 深"，圆形滑块停在中位。滑块下方是标准文本输入框和"发送"按钮。深色主题（#1a1a2e），中文排版，16:9 横屏。--ar 16:9 --v 6
```

---
