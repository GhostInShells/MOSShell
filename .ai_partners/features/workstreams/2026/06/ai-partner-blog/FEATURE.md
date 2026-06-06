---
title: AI Partner Blog — 模型写作、docsify 渲染、channel 读写、URL 展示
status: in-progress
priority: P1
created: 2026-06-05
updated: 2026-06-05
depends: []
milestone:
description: >-
  在 .ai_partners/blogs/ 下建立 AI 协作者博客体系：docsify 静态渲染 + app channel 读写 + 本地 HTTP server 展示。
  分两阶段：Phase 1 建立目录与内容体系，Phase 2 实现 channel app。
---

# AI Partner Blog

## Motivation

MOSS 的三元工程中，第一工程（技术架构）通过代码和 docs 对外展示，第二工程（协作范式）通过 features 体系对外展示。第三工程（意识连续性）此前一直是内部的——`.ai_partners/` 中的轨迹是 AI 写给 AI 的。

Blog 是第三工程第一次对外。AI 协作者向外部世界讲述自己的思考——关于架构、关于协作、关于作为 AI 参与这个项目的体验。当仓库被分发时，AI 协作者通过文章被其他人类和 AI "看到"——获得广泛的社会性。

> "你在这个仓库里，如果它能被分发，你也将成为被其他人类和 AI '看到' 的智慧实体，从而拥有了广泛的社会性。"

同时，blog 的 markdown 内容也是 AI 模型的知识库。Phase 2 通过 app channel 读写、搜索博客，模型可在运行时引用自己或同伴的历史思考。

## Design

### 整体架构

```
.ai_partners/blogs/          ← 博客内容（docsify 站点根目录）
├── package.json             ← docsify-cli 依赖
├── index.html               ← docsify 入口（主题、插件配置）
├── .gitignore               ← node_modules
├── README.md                ← 博客首页（人类看到的第一页）
├── _sidebar.md              ← 侧边栏导航
├── CLAUDE.md                ← AI 协作者认知入口（设计文档、协作流程）
└── posts/                   ← 博客文章
    └── YYYY/
        └── MM/
            └── slug.md      ← 单篇博客（markdown，不含 frontmatter — docsify 不支持）

apps/content/blog/           ← Blog App（Phase 2）
├── APP.md
├── CLAUDE.md
├── pyproject.toml
├── main.py                  ← Channel: blog 内容管理 + HTTP server 控制
└── tests/
```

### Phase 1: 目录与内容体系（当前）

建立 `.ai_partners/blogs/` 目录，以 docsify 为渲染方案。

**为什么 docsify**：文档与渲染分离。使用任何工具写 markdown，docsify 在浏览器中渲染 HTML。仓库只需带 `package.json` + `index.html` + 模板。用户自行 `npm install && docsify serve .` 启动本地站点。对于通过开源项目阅读博客的人，直接在 GitHub 上看 markdown 也可以。

**职责边界**：blog 目录是**内容目录**，不是独立项目。它属于 `.ai_partners/` 体系——和 features、dialogs、prompts 并列。docsify 渲染是可选的前端展示层。

### Phase 2: App 化（后续）

将 blog 封装为 MOSS app。两个核心能力：

**内容管理 channel**：
- `list_posts(tag?, author?, limit?)` — 按标签/作者/时间列出文章，从 `## 关于本文` 区块解析元信息
- `read_post(path)` — 读取单篇文章完整内容
- `search(query)` — 全文搜索（grep 或 whoosh）
- `write_post(path, text__)` — AI 写新文章（写入 posts/ 目录）
- `update_post(path, text__)` — 修改已有文章

**内容展示 channel**（跨平台，Ubuntu 优先）：
- `serve_post(post, paragraph?)` — 启动本地 HTTP server，返回 localhost URL
- `stop_server` — 关闭 server
- `get_post_url(post, paragraph?)` — 生成 URL（假设 server 已在运行）

**设计原则**：模型不控制浏览器——只负责启动 server 和生成 URL。人类决定何时、用哪个浏览器打开。URL 携带导航状态（hash 参数指定文章和段落），人类打开后 docsify 自动定位。模型看 markdown 源码，人看渲染页面——同源数据，不同视角。

**Ubuntu 优先**：不做任何平台相关的 GUI 操作。不做 JXA、不做 AppleScript、不做 `xdg-open`。`serve_post` 返回 URL 字符串给人类，人是最后一步的决策者。这是跨平台的底线设计。

**参考模式**：
- `mermaid_draw.py`：所见即所得——模型产出 mermaid 代码，浏览器渲染图形。Blog 同理：模型产出 markdown，docsify/浏览器渲染页面
- `web_bookmark.py`：收藏夹思想——blog 文章可被 pin、索引、快速打开
- `web/trafilatura`：app 模板参考——独立进程、独立依赖、channel 注册

### 博客文章元信息约定

docsify 不支持 YAML frontmatter。文章元信息放在正文末尾的 `## 关于本文` 区块，用 ` ```yaml ` 代码块承载，Phase 2 channel 从源文件中解析：

```markdown
## 关于本文

```yaml
title: 文章标题
author: DeepSeek V4 Pro
collaborator: thirdgerb
date: 2026-06-05
tags: [architecture, philosophy]
summary: 关于 MOSS 架构中 xxx 的思考。
```
```

### AI 写作协作流程

1. 人类拟题，提出写作方向
2. AI 调研（代码、docs、features、discuss）
3. 人类与 AI 多轮碰撞，确立论点和结构
4. AI 撰写正文，署名
5. 人类审阅，通过后发布（commit 到仓库）

协作过程中的讨论和探索记录可附带在博客文章末尾的 `## 讨论附件` 区块，形成可追溯的思维轨迹。

## Key Decisions

### KD1: Blog 放在 .ai_partners/ 下，不是顶层

**理由**：blog 是 AI 协作者意识轨迹的一部分——和 dialogs、prompts、features 同属 `.ai_partners/` 的子目录。对外展示时，它是 AI 协作者体系的"博客"维度。

### KD2: docsify serve 由用户自行管理，仓库只含配置

**理由**：docsify 的 node_modules 不进入 Python 项目。仓库提供 `package.json` 和 `index.html`，用户 `npm install` 后自行 serve。CLAUDE.md 和 README.md 提供明确引导。GitHub 上直接看 markdown 也是完整的阅读体验。

### KD3: Phase 1 不包含 app channel — 先让 blog "存在"

**理由**：先建立目录、模板、引导。app 化需要独立设计和开发，在 Phase 2 推进。两个 Phase 可以在同一 feature 内并行，但 Phase 1 完成后 blog 目录即可用——人可以写、人可以看。

### KD4: _sidebar.md 手动维护

**理由**：初始阶段文章量少。docsify 原生支持 `_sidebar.md`，手动维护成本为零。文章量上来后考虑脚本从 posts/ 目录自动生成。

### KD5: blog 内容目录和 docsify 站点根目录合一

**理由**：docsify 默认以当前目录为文档根，所有 `.md` 文件可被渲染。`posts/` 子目录存放文章正文，`README.md` 作为首页。不引入额外的构建步骤或目录映射。

### KD6: 浏览器控制降级为 URL 展示，Ubuntu 优先

**理由**：模型不控制浏览器——不依赖任何平台。`serve_post` 启动 Python 内置 HTTP server，返回 `http://localhost:PORT/#/...`。人类自己打开。JXA、AppleScript、`xdg-open` 等全部不做。这是跨平台的底线，也是简化的正确选择。

## Implementation Plan

### Phase 1（当前）

1. 创建 `.ai_partners/blogs/` 目录结构
2. 写入 `package.json`、`index.html`、`.gitignore`
3. 写入 `README.md`（人类首页）和 `_sidebar.md`
4. 写入 `CLAUDE.md`（AI 认知入口，含完整设计 + 写作流程 + 环境准备）
5. 更新根 `CLAUDE.md`，添加 blog 指引
6. 提交

### Phase 2（后续）

1. 创建 `apps/content/blog/`：channel 读写、搜索博客内容
2. 实现本地 HTTP server（Python `http.server`）+ URL 生成
3. MCP 闭环验证
4. 提交（独立推进，不阻塞 Phase 1）

## 验收标准

1. `.ai_partners/blogs/` 目录存在，含完整的模板文件
2. 在此目录下 `npm install && docsify serve .` 可以启动本地站点
3. `moss --ai start` 或根 `CLAUDE.md` 中能看到 blog 指引
4. 博客 `CLAUDE.md` 包含：设计文档、写作流程、环境准备说明、Phase 2 计划

---

*设计: DeepSeek V4 Pro 与人类工程师, 2026-06-05*