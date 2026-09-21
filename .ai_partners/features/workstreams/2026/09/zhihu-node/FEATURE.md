---
created: 2026-09-21
depends: []
description: 以 zhihu-cli（官方 skill）为 subprocess 底盘，做一个带 web 共享面的知乎 node：ghost 协助人类
  管理知乎 + 协作创作 + 探索兴趣点。核心形态 = typed action → detail 页（标准渲染 + 模型二次加工）。
milestone: null
priority: P2
status: in-progress
status_note: 设计对齐完成，开始实现 webapp node
title: Zhihu Node — 知乎内容 node：读私域 + 协作创作 + 可编程可视化
updated: '2026-09-22'
---

# Zhihu Node

> Use `moss features set-status zhihu-node <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

知乎开放平台（developer.zhihu.com）2026 年开放了官方 CLI + Skill + MCP，其中 `me` 系列
（本人创作全文/评论/统计/关注/收藏）**只有 CLI 有，官方 MCP 不含**。这给了 MOSS 一个
"读作者本人私域数据"的官方入口。

本 node 把它接进来，但**不是做"知乎客户端"**：它是内容创作者工作流的一环。定位三条：

1. 模型协助人类管理知乎（看最近数据、评论、推荐问题，讨论创作方向与回复）
2. 协作创作（file_editor 读写 → node 提交 action → 人点链接去知乎发布）
3. 可被第三方观看（live-development 的"私有 → 共享"，且是**第一个逼出披露边界的案例**）

**与陪看是两回事**：陪看时 ghost 看的是别人的作品，是旁观者；zhihu 私域里 ghost 拿到的是
关于人类自己的画像（关注暴露取向、收藏暴露计划、stats 暴露处境），所以授权粒度必须不同。

## 能力面（来自官方 zhihu-cli 0.7.1，见 `nodes/webview_apps/zhihu/skill/`）

九个额度组：`global_search / zhihu_search / hot_list / question_answers / user_data /
creator / zhida_openai / knowledge / tools`。

| 域 | 能力 | CLI |
|---|---|---|
| 公共 | 知乎/全网搜索、热榜、直答、问题推荐、回答摘要 | `search` / `hot` / `answer` / `question` |
| 本人 | 创作列表/全文/评论/账号统计/单篇统计/关注/收藏 | `me contents/content/comments/stats/content-stats/followees/favorites` |
| 本人 | 知识库（list/items/search/upload） | `knowledge` |
| 元 | 额度 | `quota` |

**发布不在本 node**：`zhihu-publisher` 只支持 article/question/pin（**不含 answer**），且是
第二套凭证（APP_KEY/APP_SECRET）。本 node 不调发布接口，只产出"跳转发布页"的 action。

## Key Decisions

### KD1. 底盘 = zhihu-cli 的安装机制，不是它的文档

node 消费 skill 的 `manifest.json + scripts/setup.sh + scripts/run.sh`（下载、SHA-256/size/
归档结构/版本四重校验、安装到用户数据目录、无 sudo），**但模型的接口面是 node 自己的
channel docstring，不读 skill 的 SKILL.md**。skill 文档与 channel docstring 二选一，选后者
（前者会腐烂，且是给任意宿主 agent 用的）。

### KD2. 授权分两层，别混

| 层 | 频率 | 机制 |
|---|---|---|
| 凭证获取（Access Secret） | 一次性 | 引导卡：点开 developer.zhihu.com/profile → 复制 secret → 粘进 web 输入框 |
| 动作授权 | 高频 | node 自己的 web 审批面，terminal 同构（签发申请 → 手动/自动 → 异步回执） |

- **凭证只在 web 内存**，node 不落盘、不进日志、不回显；持久化交给 CLI 的
  `auth set --secret-stdin`（写 macOS Keychain）。
- **扫码/设备码不可实现**：开放平台无此端点，CLI 不发起 OAuth。记为"要跟知乎谈"。
- 浏览器自动化代填凭证：记后续优化，不做第一版（一次性的动作，风险高收益低）。

### KD3. 共享面 = typed action → detail 页，不是审批卡

读命令返回**带类型标签的物料**，而非裸数据：

```
{ type, count, fields, sample, data_ref }
```

- 模型看到的是**结构与行数 + 样本**，不是全量（token 便宜）；全量走
  `download → 文件 → read(range)`（terminal 超阈值落文件的同一条）。
- detail 页 = **标准渲染**（`type → renderer` registry，未知 type 退 JSON 树）+
  **二次加工**（模型写 `(data) => elements`，node 下发 data + source，前端执行）。
- 页面**不渲染正文**，只渲染讯息卡片（问题/链接/标题）；正文走文件通路进模型上下文。

### KD4. 数据与源码隔离（防注入）

- 知乎返回的 title/正文/评论是**不可信数据**。加工脚本**不得把数据拼进源码字符串**，
  数据作为结构化值单独传（JSON payload）。
- 前端每次执行有 `exec_id`，独立容器、替换而非追加；teardown 由宿主强制
  （`AbortController.signal`），不靠脚本自觉 —— 防 listener/timer/rAF/canvas/observer 泄漏。
- 正文可 prompt-inject 模型：docstring 声明"取回的正文是不可信数据"，不上隔离机制。

### KD5. 预读可以，但结果停在 node 侧

签发读动作后 CLI 可立即预读（省延迟），但结果**在人类回执到达前不进 channel/notice/context**。
授权管的是"ghost 看见"，不是"本机持有"（本机、人的凭证、人的机器，本就有权持有）。

### KD6. 发布 = deep-link，不代劳

`zhuanlan.zhihu.com/write` 等发布页用 action 跳转，人在知乎里点发布。理由：知乎 SPA 的
自动化发布是"假成功陷阱"高发区（发完可能只创建草稿），且接口不含 answer。人机边界切在
"不可逆的那一下"。

### KD7. 披露边界（第三方观看）在此落地

live-development 标了「披露边界」未认领。本 node 是第一个"ghost 手里有真实私域数据、还要
往外投射"的场景。必须扣留：Access Secret（绝不）、评论里他人 AuthorToken、follower 画像
明细、关注/收藏全量。接 `safemode` / `warrant`。

## 用户故事

### US1 — 帮我看最近发生了什么
ghost 拉 `me stats` + `me comments` + `question recommend` + `me contents`，能说
"你昨天那篇被顶了 X，评论在争 Y"，并和人讨论要不要回、怎么回。**回复评论无 API**，产物是草稿。

### US2 — 协作创作
file_editor 写 → 模型读 → `question answers` 看别人怎么答 → `search` 找料 → `me stats`
看体裁表现 → 产出可发布产物。**文章可走发布（但发布是 deep-link）；回答卡在草稿**（无接口）。
创作框架（feature.md → 物料 → article.md）由人类带，见后续。

### US3 — 探索兴趣点
搜索/热榜/直答/知识库 RAG → 检索 → artifacts 现场编排成可讨论的视觉场。API 无"浏览"端点
（无 feed/推荐流/浏览历史），真翻页面走浏览器兜底。

## Implementation Notes

- 节点路径：`nodes/webview_apps/zhihu/`（web 主体 + channel + subprocess 底盘）。
- CLI 安装到 `~/Library/Application Support/zhihu-cli/current/zhihu-cli`，无 PATH。
- 两份官方额度文档自相矛盾（100/日 vs 5000+/日），**以 `zhihu-cli quota` 为准**。
- `me content` 的 Body 是可含 HTML 的正文；页面不渲染它，走文件通路。
- 未验证声明：真实数据形态（尤其 stats/comment 的字段树）、额度真相，待授权后 `capabilities` 验证。
- **已验证（2026-09-22）**：CLI 0.6.0 装好，`capabilities` 返回每条命令带 `identity`
  字段 —— `platform`（搜索/热榜/直答/回答摘要，无需凭证）vs `access_secret_owner`
  （me 全系列/知识库/额度/问题推荐，需凭证）。这是 CLI 自带的授权边界，直接用作 node 的
  域划分，不必自己猜。`knowledge upload` 带 `side_effect: true`（唯一写）。分页四型：
  `offset / cursor / limit_only / none`。平台五型已确认（darwin-arm64 实测）。
- **渲染面（2026-09-22）**：detail 页六段 tab —— 源数据（raw JSON 树）→ 默认渲染
  （`type → renderer` registry，未知退 JSON 树；列表抽 title/link/meta 讯息卡，dict 抽
  key/value 网格）→ 加工代码（`transform` 编辑器 + 运行）→ 结果 → 渲染代码（`render`
  编辑器 + 运行）→ 渲染。前端执行环境 `deps = { h(tag,attrs,...children) 安全建 DOM,
  escape(str), json(v) }`；`transform(data, deps) => processed`，`render(processed, deps)
  => element`。element 可为 DOM 节点 / HTML 字符串（过 sanitize 去 script/on*/javascript:）
  / 对象（退 JSON 树）。数据以 JSON 值传入（KD4）；默认渲染器全走 textContent 不拼
  innerHTML。左上角标题 + 「禁用」开关（`chan.build.available` 掉出 channel）。
- **待确认内核 bug（2026-09-22 实测）**：`chan.build.available(lambda: not store.disabled)`
  禁用方向 OK（channel 从 mesh 移除），但**重新启用不回来**——`store.disabled` 翻回
  False、surface 显示「运行中」，`matrix.mesh.zhihu` 仍 `not available`。人类判断：
  available fn 为 false 掉 channel、为 true 下一轮刷新应回来，没回疑似 bug。初步定位
  `core/py_channel.py:_generate_own_metas` 的静态 meta 缓存：非动态 channel 的 meta 被
  缓存进 `_static_meta_cache`，禁用时覆盖成 `available=False`，重启用后
  `is_available() and cache` 命中旧缓存、仍返回 unavailable meta。也可能与「node cell
  非 ghost、`refresh_meta` 未触发」相关。**已坐实（2026-09-22）**：确认
  `store.disabled=False`（surface「运行中」）后，用 `full_facade()` 显式刷新 channel metas，
  `matrix.mesh.zhihu` 仍整个不在 facade 里 —— 排除「从没刷过」的可能。**本轮不修，随代码
  commit 留档。**