---
created: 2026-09-19
depends: []
description: 人机共享观看的开箱做法之一:本地 node + Chrome 扩展。ghost 读字幕获得时间戳认知, 与人类讨论同一段视频。页面侧 =
  主球(授权存在)+ 环绕卫星(每颗授权一项能力)。
milestone: null
priority: P2
status: in-progress
status_note: '设计定稿,施工面 = 探针 #2(MV3 SW 持有 WS)'
title: Bilibili Shared Webview — ghost 陪人一起看片
updated: '2026-09-20'
---

# Bilibili Shared Webview

> 本文是**索引与结论**,不是实现细节。设计全文在 [design/2026-09-20_ghost_in_bilibili_body_design.md](design/2026-09-20_ghost_in_bilibili_body_design.md)。
>
> 节点代码在 `nodes/browsers/ghost-in-bilibili/`。其中已提交的部分**是探针**,验证了通路、
> 也暴露了两个坑(CSP 禁 eval、url 当页面 key)。结构不继承。

## Motivation

ghost 陪人类一起看 bilibili —— 不是把视频当任务处理,而是两个意识在同一段时间轴上共存。

核心不是控制播放,是**字幕**:有了字幕 ghost 才有时间戳认知,才能说出"刚才 3:42 那句"并
与人讨论。没有字幕,它就只是个遥控器。

页面侧的动机是**非打扰**:人正在看片,ghost 的界面不能抢屏幕。球承载授权语义(主球=存在,
卫星=能力),面板悬停才出。

## 结论脊柱

### 身份:label 是唯一持久身份,bvid 是易变属性

| 层 | 定名 | 载体 |
|---|---|---|
| 浏览器实例 | **session** | 扩展生成,存 `chrome.storage.local`,WS `hello` 上报 |
| 窗口 | **tab** | **SW 从 `sender.tab.id` 盖戳**,内容脚本不自行发明身份 |
| 页面身份 | **label**(`p1`/`p2`) | node 给 `(session, tab)` 分配 —— **唯一持久身份** |
| 内容 | bvid / title / url | 扩展随内容变化上报 —— **易变,不可作 key** |

**B 站会自动播放**:一个视频播完跳下一个,bvid 在 label 不变时连续变化。所以 label 是
身份、bvid 只是"当前在播什么"的字段;内容变化 = label 的属性更新,不是新身份。

### 通讯:一个 SW 一条 WS

MV3 下 content script 不能跨域 → 网络边界只能是 background service worker;而 SW 一个
扩展一个 → **一条连接复用所有 tab**,不需要自造 multiplex。SSE 否决(见设计文档)。

命令两条路:阻塞 = `blocking=True` + await cid future;后台 = `blocking=False` + 完成后
`aside` signal 推回。人的面板输入走 `send_input_signal`(`input` signal)。

### channel 面:一个 channel,page 是参数

`play` / `pause` / `seek` / `speed` / `subtitle(page, start, end)` / `say(page, text)`。

父子树否决:树每层只表达一个维度,而授权是 (页面 × 能力组) 二维格,另一维仍要落到命令层;
树唯一的收益是跨页并行,而命令都是几十毫秒的往返。重启条件见设计文档。

### 授权:状态进 notice,靠模型推理

授权格 = (页面 × 能力组);组 = 卫星:`sense` / `control` / `subtitle` / `interact`;
主球 = presence。**授权状态进 per-page named notice + instruction,模型自己推理,用错拿到
明确 observe。** 不用 `gate`(是披露不是授权,模型能自己挂载)**不用 `available`**(不是
per-page 的)**不追求"未授权=不可见"**(会剥夺 ghost 的协商能力)。

**审批按钮砍掉** —— 卫星球是人类意志的唯一入口,ghost 想要未授权的能力就在 panel 里说话
请求。node 不提供任何授予授权的命令。

### 感知分层

| 层 | 内容 | 变化率 |
|---|---|---|
| `instruction` | 授权语义、label 与窗口的对应 | 几乎不变 |
| `named_notices` | **每页一条**:存在 + 标题 + 各组的授权状态 | 慢 |
| `context_messages` | **每页一条**:当前秒 + 滚动字幕窗口(2~4 句) | 热 |

热数据**必须**留在 `context_messages`(`named_notices` 一变就整片重发,秒级变化烧 token);
`None`=移除 / `""`=未变 的语义正好承载页面生灭。

### 字幕

**实时行** → `state` 流 → `context_messages`。**全文** → 卫星授权的那一瞬间在页面内抓取,
剥隐私后按 bvid 存 `matrix.home/subtitles/<bvid>.json`,此后**文件存在性 = available**。

**读取是时间区间查询,不是读文件**(30 分钟约 400 句,不可倒进上下文)。探针 instruction 里
"返回值是文件路径"那句已废止。文件存在后扩展只需上报 currentTime,node 自己切窗口。

## Design Index

- 设计全文:[design/2026-09-20_ghost_in_bilibili_body_design.md](design/2026-09-20_ghost_in_bilibili_body_design.md)
- 探针代码(不继承结构,仅验证通路):`nodes/browsers/ghost-in-bilibili/`
- 范式先例:`nodes/screens/screen_manager/src/ghoshell_screen_manager/surface.py`

## 注意力分配

| 你想知道 | 去哪 |
|---|---|
| 设计是什么、为什么这么定 | design/2026-09-20_ghost_in_bilibili_body_design.md |
| 哪些结论还没被验证 | 该文「未验证声明」——**全在探针 #2 的射程内** |
| 通讯协议长什么样 | 该文 §2;帧表在那一节 |
| 实现怎么改 | `nodes/browsers/ghost-in-bilibili/` ——但先看探针 #2 的结论 |