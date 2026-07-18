---
created: 2026-07-13
depends:
- ghost-filesystem-desktop
description: 'vendor All-Hands-AI/openhands-aci 的 editor 子集 (5 动词: view/create/ str_replace/insert/undo_edit),
  给 MOSS 契约层加一份薄 FileEditor ABC. 与 ghost-filesystem-desktop 平级独立立项 (不入 desktop.py),
  channel 层 可合体. 骑 Anthropic text_editor tool 血统 (SOTA 模型 prompt 先验). Contract 同步,
  async 是 caller 责任 (K27 姿态延续).'
milestone: 0.1.0
priority: P0
status: completed
status_note: >-
  v1 (contracts + core + 45 tests) 与 v2 (channel 装配 + 9 tests, K13 B案)
  均已完成, 53 tests 全绿. Dogfooding (moss-as-mcp 手感验收) 延后到
  matrix cells CLI 治理完成后一起做 — 届时依赖注入路径 (workspace
  manifests providers 挂 build_file_editor_channel) 已就位.
title: File Editor Contract — vendor openhands-aci 为薄契约，与 Desktop 平级
updated: '2026-07-18'
---

# File Editor Contract

## Motivation

Ghost 需要**结构化文件编辑动作** (view / create / str_replace / insert /
undo_edit), 不同于:

- **bash / subprocesses** — 起进程跑命令, 语义粗糙 (sed -i 之类), token 贵,
  且预训练先验完全不同 (模型看 `<file:edit old="..." new="...">` 跟
  `<bash>sed ...</bash>` 是两个心智模型).
- **Ground / Desktop 认知面** — 只钉地址 + 对账渲染, K6 撤了 read-before-
  write 全链路守卫, 写路径明示"由写路径各自的卫生纪律负责本契约不定义"
  (`ghost-filesystem-desktop` FEATURE.md 已收敛).

写路径的空档由本 feature 承接. 关键判据: **不入 Desktop 契约**, 独立立项.

## Design Index

- **上游 vendor**:
  - `openhands-aci` (All-Hands-AI, MIT) — https://github.com/All-Hands-AI/openhands-aci
  - `openhands_aci/editor/OHEditor` — Anthropic quickstart `str_replace_editor`
    血统, 5 动词, sync API, 已被大量主流模型 (Claude 4.x / GPT / Qwen) prompt
    先验锁死.
- **平级 feature**:
  - `ghost-filesystem-desktop` — 认知面 (pin/update/frame). channel 层可合体
    (K33 追加见 desktop FEATURE.md).
- **channel_builder / 三层纪律参照**:
  - `moss codex blueprint channel_builder`
  - `src/ghoshell_moss/channels/module_eval_channel.py` — abc → concrete →
    channel 三层装配的标杆
- **feedback memory 相关**:
  - `feedback_io_contract_sync.md` — IO 契约默认同步, async 是 caller 责任

## Key Decisions

### K1. Contract 独立立项, 不入 `contracts/desktop.py`

**判据**: Ground 与 FileEditor **最小依赖不同**. Pin 只在 Ground (root)
里存在; FileEditor.write(path) 只需一个 `root: Path` 就能工作, 不需要 Ground.
塞一起是搭便车, 也是老 Desktop 融合病灶第三次复发的门. K11/K13 拆
subprocesses/job_supervisor 的 precedent 已定.

**辖域分工**:

| 契约 | 语义 | 状态 |
|---|---|---|
| `Grounds` / `Ground` (contracts/desktop.py) | 钉地址 + 对账 + 帧渲染 (认知面) | 已落地, 87 单测 |
| `FileEditor` (contracts/file_editor.py, 本 feature) | view/create/str_replace/insert/undo_edit (写路径 + 结构化查看) | v1 已落地 (contracts + core + 45 tests) |
| `Subprocesses` / `JobSupervisor` (contracts/subprocesses.py, job_supervisor.py) | 起进程 / 后台 fold | 已落地 |

**channel 层可合体, contract 层不合体**: 新 desktop channel 完全可以同时消费
`Grounds` + `FileEditor`, 模型看到 `desktop:pin` 和 `desktop:edit` 一体. 契约
分家 ≠ 用户接口分家. 这是 K18 三层纪律的红利.

### K2. Vendor openhands-aci `editor/` 子目录, 不 `pip install`

**背景**: `pip install openhands-aci` 拖 ~30 个依赖 (pandas/matplotlib/
beautifulsoup4/mammoth/pypdf/pdfminer/pypptx/speechrecognition/pydub/
youtube-transcript-api/networkx ...), 因为 openhands-aci 是 OpenHands agent
通用工具包 (含 MarkdownConverter 把 Office/PDF/YouTube 转 md). MOSS 只要
5 动词编辑, 这些依赖是纯拖累 (装机体积 +300~500MB, CI 脆弱面).

**做法**: verbatim 拷贝 `openhands_aci/editor/` 子目录到
`src/ghoshell_moss/core/file_editor/_openhands/`, 保留 MIT LICENSE 头,
`UPSTREAM.md` 记录版本号 + 上游 commit SHA + 删了什么 + 为什么.

**必删**: `md_converter.py` (Office/PDF 转换, 是 30+ 依赖的元凶).
**默认关**: `linter` (tree-sitter/libcst/flake8), 用户想开自己装.
**保留**: `editor.py` / `encoding.py` / `exceptions.py` / `history.py` /
`results.py` / `config.py` / `prompts.py` 相关片段.
**净依赖**: `binaryornot` (或用 stdlib magic byte 替代).

**每次上游升级主动拉 patch**. K16 "骑先验" 要的是 prompt 先验, 不是运行时
耦合.

### K3. Contract 同步, 不加 async

**判据**: 见 memory `feedback_io_contract_sync.md`. 文件 IO 是 sync 阻塞,
套 async 只把 `to_thread` 成本从 caller 转嫁到 impl, 且非 asyncio 环境
(CLI / 脚本) 直接不可用. K27 姿态延续 (Desktop `Ground.pin()` 同步 +
`observe_sync` 已验证).

想 async 的 caller 自己 `asyncio.to_thread(editor, ...)`. OHEditor 本身也是
sync, 直接骑就对了.

### K4. Contract 动词直接对齐 OHEditor 5 命令, 不发明

- `view(path, view_range=None) -> str`
- `create(path, file_text) -> FileEditorResult`
- `str_replace(path, old_str, new_str) -> FileEditorResult`
- `insert(path, insert_line, new_str) -> FileEditorResult`
- `undo_edit(path) -> FileEditorResult`

**理由**: OHEditor 血统源自 Anthropic quickstart `str_replace_editor` tool,
是 SOTA 模型 (Claude 4.x / GPT / Qwen) 都见过的 prompt 先验. K16 双寄存器
词汇要求"表面上的每个动词必须命中一个预训练先验" — 这里现成一份, 不发明.

**唯一 MOSS 侧扩展**: 构造参数 `workspace_root: Path | None` (对齐 OHEditor
的同名参数). **只做相对路径 hint, 不做强制边界** — 本 feature 落地时
明确改主意: 空间边界是 Grounds 的责任 (Ground.root + `PathOutsideRootError`),
FileEditor 不假设 Ground 存在, 也不复用 `PathOutsideRootError`. 两个 contract
彻底解耦.

**Result 类型**: 独立 `FileEditorResult` (dataclass), 不复用 OHEditor 的
`CLIResult` (那是他们的实现类型).

### K5. 异常层跟 MOSS 约定, 不透传 vendor

- `FileEditorError` (基类, `DesktopError` 兄弟平级, **不共享继承**)
- `ParameterMissingError` / `ParameterInvalidError` (对齐 OHEditor 的
  `EditorToolParameterMissingError` / `EditorToolParameterInvalidError` 但
  重命名, 用 MOSS 姿态)
- `FileValidationError` (对齐 vendor 同名)
- `NoEditHistoryError` (undo 时无历史 — vendor 是裸 `ToolError` + 消息串,
  adapter 层按消息内容识别并升级为独立异常)

**Adapter 内部** catch `ToolError` 家族, 转为 MOSS 异常. vendor 异常类
**不出 core/file_editor/**.

### K6. 单测独立锚在 contract, 不引 vendor 测试

**判据**:

1. Vendor 测试测的是 vendor 实现细节 (FileHistoryManager per-file 10 步栈,
   encoding fallback 链, MarkdownConverter 分支). 我们契约不承诺这些细节,
   换 vendor 时测试**仍应绿** — 引 vendor 测试 = 焦点从契约拉回实现.
2. Vendor 测试依赖 tree-sitter/libcst/pandas/matplotlib 那套, K2 里拆掉了.
   拉进来一半 collection error, 改到能跑性价比负.
3. 契约测试是 API 教材 — Desktop 87 单测就是新实例读懂 Ground 语义的最快
   入口. File editor 走同姿态.

**做法**: `tests/ghoshell_moss/core/file_editor/test_file_editor.py`
契约验收, DefaultFileEditor 走真文件系统 (`tmp_path`), 每动词 3 类 (happy /
边界 / 异常). 从 vendor 单测**偷 edge case 灵感** (str_replace 唯一性检查 /
undo 栈上限 / binary 拒绝) 但用 MOSS 姿态重写.

估 30~50 单测封顶. Desktop 87 单测形状是参考.

## Implementation Notes

### 目录结构

```
src/ghoshell_moss/
├── contracts/
│   └── file_editor.py          # FileEditor ABC + Result + 异常 (薄, 200~300 行封顶)
└── core/file_editor/
    ├── __init__.py             # 公开 DefaultFileEditor
    ├── _default.py             # DefaultFileEditor -> 调 _openhands.OHEditor + 异常转译
    └── _openhands/             # vendor 子目录
        ├── LICENSE             # openhands-aci MIT 原文
        ├── UPSTREAM.md         # 上游版本号 + commit SHA + 删除清单 + 为什么
        ├── editor.py           # verbatim 拷 (删 md_converter 相关 import)
        ├── encoding.py
        ├── exceptions.py
        ├── history.py
        ├── results.py
        ├── config.py
        └── prompts.py          # 保留 truncation notice 常量, 删 md 相关

tests/ghoshell_moss/core/file_editor/
├── __init__.py
└── test_file_editor.py         # 契约验收, tmp_path 真 IO, 30~50 测试
```

### 装配路径

Channel 层未定. 两种候选:

**A.** 复用 `channels/desktop_channel.py` (未来重写的那份), 同时消费
`Grounds` + `FileEditor`, 模型看到 `desktop:pin` 和 `desktop:edit` 一体.

**B.** 独立 `channels/file_editor_channel.py`, 与 desktop channel 平行.

K1 已明说"channel 层可合体", 倾向 A. 但 A 需要 desktop channel 落地
(K33 未验证) 先行. 本 feature v1 只交付 contracts + core + 单测, 装配待
desktop channel 落地时一起收.

### 与 Desktop pin 的联动 (未来)

FileEditor.create / str_replace / insert 命中已 pin 的地址时, 理论上应该
主动触发 pin 的 stale 标记 / 或 UpdateResult 入 CTML `<result>`. 这是 K17
对账语义的自然延伸, **但不进 FileEditor 契约** — 在 channel 层装配时由
handler 装配 (composition, 不是 contract 义务).

## 交付清单

### v1 (已交付, committed)

- [x] vendor 拷贝 + `UPSTREAM.md` + LICENSE 归位
- [x] `contracts/file_editor.py` (ABC + Result + 异常, 232 行)
- [x] `core/file_editor/_default.py` (DefaultFileEditor + 异常转译)
- [x] `core/file_editor/__init__.py` (公开 API)
- [x] `tests/ghoshell_moss/core/file_editor/test_file_editor.py` (45 测试, 全绿)
- [x] `architecture.py` 加 import 索引 (`file_editor_impl`)

### v2 (channel 装配, 已交付)

- [x] `channels/file_editor_channel.py` — K13 B案实现, 双入口
  (`new_file_editor_channel` / `build_file_editor_channel`)
- [x] `tests/ghoshell_moss/channels/test_file_editor_channel.py` — 9 tests
  全绿, 走 `ctml_shell_test` 范式 (与 `test_ctml_v1.py` 一致)
- [x] docstring 遵循 channels/CLAUDE.md 范式:
  `结构化文件编辑 — Anthropic text_editor tool 血统 | 集成 | alpha`

### v2 dogfooding (延后)

- [ ] moss-as-mcp dogfooding — **延后到 matrix cells CLI 治理完成后**.
  届时 workspace manifests providers 挂 `build_file_editor_channel` 的
  依赖注入路径已就位, 手感验收可以在真实 cell workspace 上跑, 而不是
  绕过 IoC 直接 new_file_editor_channel(DefaultFileEditor()).

## v1 落地追加决策 (2026-07-14, Claude Opus 4.7)

### K7. Directory `view` 从 vendor 层砍掉

**背景**: 上游 OHEditor `view(dir)` 用 `find -L path -maxdepth 2` shell
命令列目录. 本 feature 落地时讨论: 目录列表不是编辑动词的语义, 且模型在
Claude Code 已经用 bash / glob 探目录, 骑现成先验更清爽.

**决策**: `view` 只接受文件路径. 目录路径抛 `ParameterInvalidError`
(错误消息里 "use bash/glob for directory listing" 指路). 上游的 shell 命令
分支同步删除 — 顺带消掉一处 shell 注入面.

### K8. Office / PDF / audio 支持整体砍

**背景**: 上游 OHEditor 通过 `MarkdownConverter` 把 `.docx/.xlsx/.pptx/
.pdf/.mp3/...` 转 markdown 展示. 依赖 mammoth/pypdf/pdfminer/pypptx/
speechrecognition/pydub 等一大堆 (~20 个包).

**决策**: 全砍. Office / PDF / audio 文件走 `_is_binary` 分支直接抛
`FileValidationError`. 未来若需要"读二进制"独立立项 (`read-binary` 之类),
与 file editor 语义解耦.

### K9. Vendor 依赖清零 — stdlib 全替代

**背景**: K2 说净依赖 `binaryornot` (或 stdlib 替代). 实际落地时全部
stdlib 替代:

| 上游依赖 | MOSS 替代 |
|---|---|
| `binaryornot.check.is_binary` | 内联 `_is_binary` (读前 8 KB 检查 null byte) |
| `charset_normalizer.detect` | try utf-8 → fallback latin-1 |
| `cachetools.LRUCache` | plain `dict` + FIFO eviction (1000 cap) |
| `openhands_aci.linter.DefaultLinter` | 直接删, `enable_linting` 参数也删 (决策 3) |
| `openhands_aci.utils.shell.run_shell_cmd` | 目录 view 删掉, 顺带消掉此依赖 |
| `md_converter.MarkdownConverter` | K8 里已删 |
| `.file_cache.FileCache` (磁盘 undo 历史) | 内存 `deque` (per-instance session) |

**净依赖为零**. 全部 stdlib.

### K10. Undo 历史从磁盘改内存

**背景**: 上游 `FileHistoryManager` 用 `FileCache` 落 JSON 到 tmp dir, 跨
进程持久 undo 栈. MOSS 不需要 — 一个进程死了 editor 上下文也就死了,
持久 undo 是伪需求.

**决策**: `FileHistoryManager` 重写为纯内存 `dict[str, deque[str]]`,
per-file cap 5 (upstream 值). API 保持完全兼容 (`add_history` /
`pop_last_history` / `get_metadata` / `clear_history` / `get_all_history`).

### K11. Undo 后 NoEditHistoryError 的识别方式

**背景**: 上游 vendor 在 undo 无历史时抛裸 `ToolError` + 消息 "No edit
history found for {path}." — 没有独立类型.

**决策**: adapter 层 catch `ToolError` 时按**消息内容**识别是不是 undo
无历史, 是则升级为 MOSS 独立异常 `NoEditHistoryError`, 其它 ToolError
统一降级为 `FileValidationError(path, msg)`. 这条决策在 UPSTREAM.md 也
有记录, 未来上游若给了独立异常类, adapter 换类型即可.

### K12. Adapter 层做 vendor Path value stringification

**背景**: 上游 `EditorToolParameterInvalidError` 的 `value` 字段有时是
`PosixPath` 对象, 直接 str() 出来是 `PosixPath('/x/y')` 而非 `/x/y`.

**决策**: adapter 层 catch 时判断 isinstance(Path) 后 str() 一次, 让
`ParameterInvalidError` 的消息干净. 上游若来日修了, 这段 patch 可删.

## 装配 — K13: 独立 file_editor_channel (B案, 2026-07-18 决策)

A / B 两案已裁决, **选 B: 独立 `channels/file_editor_channel.py`**, 不嵌入
desktop channel.

**判据**:

1. FileEditor 对 Grounds/Desktop 零依赖 — 独立 channel 反映这个事实.
2. Desktop channel 以后用 `chan.build.import_channels(file_editor_channel)`
   一行合体, 模型看到 `desktop:view` / `desktop:create` 等统一命名空间.
3. 独立测试、独立演进、独立使用 (简单 CTML shell 场景不需要 desktop).

**Channel 接口**:

```python
def new_file_editor_channel(
    *,
    workspace_root: str | Path | None = None,
    max_file_size_mb: int | None = None,
    channel_name: str = "file_editor",
) -> MutableChannel:
```

`workspace_root` / `max_file_size_mb` 直传 `DefaultFileEditor`. 无复杂
生命周期 — editor 是纯内存状态 (undo 历史 session-scoped), 不需要
startup/close.

**CTML 表面**:

```
<file_editor:view path="/abs/path/to/file.py"/>
<file_editor:view path="/abs/path/to/file.py" view_range="1,50"/>
<file_editor:create path="/abs/path/to/new.py" file_text="print('hello')"/>
<file_editor:str_replace path="..." old_str="foo" new_str="bar"/>
<file_editor:insert path="..." insert_line="10" new_str="new line"/>
<file_editor:undo_edit path="/abs/path/to/file.py"/>
```

**view_range 参数**: str 格式 `"start,end"` (1-based, 含端点). Channel 层
parse 为 `[int, int]` 后传 FileEditor.view(). 空字符串 → None (全文). 骑
Anthropic text_editor 的 `[start, end]` 先验.

### K13a. Blocking 语义

| 动词 | blocking | always_observe | 理由 |
|------|----------|----------------|------|
| view | False | True | 纯读, 模型需要内容决定下一步 |
| create | True | True | 写, 顺序执行保安全 |
| str_replace | True | True | 写, 模型必须验证编辑结果再发下一条 |
| insert | True | True | 同上 |
| undo_edit | True | True | 写, 与编辑顺序执行 |

全部 `always_observe=True` — 模型发出编辑后**必须**看到 old_content /
new_content / snippet 验证正确性. 这是 str_replace_editor 血统的核心
交互模式.

写操作 `blocking=True` 而非 False 的理由: 模型思考代码变更是顺序的,
实践中几乎没有并行编辑需求. 保守安全, 以后可以放松 (向后兼容).

### K13b. 不加 cwd

cwd 是 shell 的概念, bash channel 持有它是职责所在. File editor 只管
"在这个路径上做这个动作" — 路径解析不是它的事. 多个 channel 各自维护
cwd 会导致 N 份不同步的真相. 模型构造绝对路径没有认知负担.

### 装配实施计划

实现在下个 session — 用户开 `moss-as-mcp` 做 dogfooding, 手感验收先于
自动化 (K33 原话). Channel 层单测按 `tests/ghoshell_moss/channels/` 惯例:
bootstrap → refresh_metas → 验证 command 存在 + 签名 + 执行正确.

**不做** (v1 scope 外, 与 contracts/core v1 一致):
- Pin-edit 联动 (编辑命中已 pin 地址时标记 stale)
- CLI dogfood (`moss edit`)
- Desktop channel 导入 file_editor (等 desktop channel 落地时一行 import_channels)

## 与 ghost-filesystem-desktop 的交叉

| 维度 | ghost-filesystem-desktop | file-editor-contract |
|---|---|---|
| 辖域 | 认知面 (pin/update/frame) | 写路径 + 结构化 view |
| 契约文件 | contracts/desktop.py | contracts/file_editor.py |
| Core 目录 | core/desktop/ | core/file_editor/ |
| Channel 装配 | 未来 desktop channel (K33) | 独立 file_editor_channel (K13 B案), desktop 通过 import_channels 合体 |
| 共享空间边界 | root + PathOutsideRootError (Grounds 管辖) | 不共享 — K4 落地改主意, 两个 contract 彻底解耦 |
| 状态 | in-progress | in-progress (v1 已交付, 装配+联动 未完成) |

## v2 装配落地追加决策 (2026-07-18, Claude Opus 4.7 in Claude Code)

### K14. 依赖治理 — 双入口 (contract 消费 + IoC factory)

**背景**: 初稿把 channel 直接 `editor = DefaultFileEditor(...)` new 出来,
被人类工程师戳穿"契约白做了 — factory 才是核心"; contract 存在的意义
就在于 caller 可以通过 provider 换实现 (session-scoped undo / mock /
远程 proxy 等). Channel 里直接 new 就把 IoC 治理面绕过去了.

**决策**: 两层 API, 都进 `__all__`:

```python
def new_file_editor_channel(
    editor: FileEditor,
    *, channel_name: str = "file_editor",
) -> MutableChannel:
    """组合原语 — 收契约, 不知道 IoC 存在. 测试/脚本直接喂."""

def build_file_editor_channel(
    container: IoCContainer,
    *, channel_name: str = "file_editor",
) -> Channel:
    """factory — 从 container.get(FileEditor) or DefaultFileEditor() 解析.
    真正被 workspace manifests providers 注册使用的形态."""
```

**Fallback 姿态**: `container.get(FileEditor) or DefaultFileEditor()` —
零 provider 也能跑 (file editor 无外部依赖, 设计就是"零配置默认,
特殊场景 provider 覆盖"). 跟 mcp_hub 那种硬 `force_fetch(Matrix)` 不同.

**Undo scope 由谁注册决定**, 不由 channel 决定. Session-scoped provider
= per-session undo; process singleton = 全局共享 undo. Channel 层 API
不再暴露 `workspace_root` / `max_file_size_mb` (K13 里保留的这两个是
DefaultFileEditor 构造参数, 属 provider 层职责).

### K15. Command 错误统一 `raise_observe`, 无 fallback flag

**背景**: 初稿加了 `raise_on_error: bool = True` flag — True 时
raise_observe 中断同轨, False 时 return error string. 讨论时人类
指出 "过于慎重" — str_replace_editor 语义就是 "edit-then-observe",
模型发出编辑后**必须**看到返回值验证, 错了就该被中断; return string
让模型继续跑是伪需求.

**决策**: 5 个命令的 `except FileEditorError` 都走 `_raise_observe`
(`CommandUtil.raise_observe(f"[{type(e).__name__}] {e}")`), 无 fallback.
Command 从不返回 error string, 只有成功结果或 exception 传给框架.

### K16. text__ 承载策略 (attribute vs body vs JSON in body)

**背景**: K13 例子里 `create` / `str_replace` / `insert` 全用 XML
attribute 传长内容. 讨论时人类指出:

- attribute 内多行代码要 XML escape `<>&"` — 认知负担
- CDATA 免了 escape, 但 CTML 不支持 CDATA 嵌套 — 撞到 `]]>` 就崩

人类抛的关键技术手段: "text__ 可以接 JSON, 拼 schema 教模型". 对齐了
Claude Code Edit tool 的 JSON string 姿态 (SOTA 先验).

**决策**:

| 命令 | attribute | text__ | 备注 |
|------|-----------|--------|------|
| view | path, view_range | — | 都短 |
| create | path | 纯 str, CDATA 包 | 单 blob, 不套 JSON |
| str_replace | path | **JSON** `{old_str, new_str}`, CDATA 包 | 双 blob 走 JSON |
| insert | path, insert_line | 纯 str, CDATA 包 | 单 blob + int |
| undo_edit | path | — | |

**判据抽出来** (下一 case 拿来复用): JSON schema 拼不拼取决于字段
复杂度和边界模糊度. 少字段 + 无歧义类型 → example 够; 多字段 / 有
可选分支 / 有字段间约束 → 必须拼 schema.

`str_replace` 现在两个 str 字段, example 教学够, `json.loads` +
`KeyError` 手写检查够, Pydantic 是过度设计. 字段增加再上 Pydantic.

### CTML 姿态经验 (单测钉住, 落在 test_file_editor_channel.py 文档头)

`ctml_shell_test` 端到端跑通 9 个测试后**确认**了这些只在 spec 有、
example 稀少的姿态:

- **CDATA 包 JSON in text__ 跑得通** (test 4) — CTML v1 从未有 example,
  实测 OK. 唯一隐患 payload 里出现 `]]>`, 罕见.
- **mixed attribute + text__ 跑得通** (test 5) — int attribute 混 str
  text__, CTML 一起 dispatch.
- **`view_range="1,3"` 经 ast.literal_eval → tuple(1,3)** (test 2) —
  逗号分隔 attribute 变 tuple 而非 str. 接收端做 normalize (`_parse_view_range`
  接受 str/tuple/list 三种).
- **task.result() 抛异常时是 None** — 用 `.exception()` 判错, 别用
  `.result() is None` 反推. `tasks[0]` 也不是"业务首个" — `<_>` scope
  本身产 task 排前面, 用 `caller_name().startswith(...)` 过滤才稳.

---

## 给下一个模型实例 (v2 完成版更新)

这份 FEATURE.md 是 2026-07-13 立项、07-13/14 Claude Opus 4.7 (1M) 完成 v1
落地、07-18 Claude Opus 4.7 完成 channel 装配设计与 v2 落地的完整认知记录.

**当前事实**:

- v1 (contracts + core + 45 tests) 与 v2 (channel + 9 tests) 均已提交,
  53 tests 全绿. Workstream 状态 completed.
- 装配走 K13 B案: 独立 `channels/file_editor_channel.py`. 双入口
  (K14): `new_file_editor_channel(editor)` 组合原语 + `build_file_editor_channel(container)`
  IoC factory. 后者带 `container.get(FileEditor) or DefaultFileEditor()`
  fallback.
- 错误统一 `raise_observe` 中断同轨 (K15), 无 raise_on_error flag.
- text__ 姿态混合 (K16): create/insert 单 blob 走 CDATA 包 str,
  str_replace 双 blob 走 CDATA 包 JSON. 判据 (字段复杂度决定 schema
  拼不拼) 可复用.
- **dogfooding 延后到 matrix cells CLI 治理完成后** — 届时依赖注入
  路径 (workspace manifests providers 挂 build_file_editor_channel)
  就位, 手感验收在真实 cell workspace 跑, 不再绕过 IoC.

**dogfooding 接手时要做的**:

```
# 1. 确认状态
moss --ai features status file-editor-contract
python -m pytest tests/ghoshell_moss/channels/test_file_editor_channel.py \
                 tests/ghoshell_moss/core/file_editor/ -q  # 应 53 全绿

# 2. 读 v2 决策 (K14/K15/K16, 本文件)
# 3. 读单测头文档 (CTML 姿态经验)
cat tests/ghoshell_moss/channels/test_file_editor_channel.py  # 头 30 行

# 4. workspace manifests providers 挂 build_file_editor_channel
#    (matrix cells CLI 治理完后, workspace manifest 姿态确定)
# 5. moss-as-mcp 起来, coding agent 连上, 让模型真实调 file_editor 命令
# 6. 观察 CTML 表面在真实模型侧的可读性和错误恢复
```

**已知未决**:

1. **Dogfooding 未跑** — 上述延后原因. 一旦跑起来, 关注: (a) CDATA
   包 JSON 在模型侧真实生成时的 `]]>` 冲撞率 (K16 已知隐患); (b)
   str_replace error 消息 (`No replacement was performed. Multiple
   occurrences ...`) 是否引导模型加上下文重试.
2. **Pin 联动** — 编辑命中已 pin 地址时触发 stale. Channel 层
   composition, 不进契约. desktop channel 落地时一起收.
3. **tmp / 大文件** — 当前 10 MB cap 直接抛 FileValidationError. 暂无痛感.

**相关的 memory 文件**:

- `feedback_io_contract_sync.md` — K3 依赖的这个原则 (IO 契约默认 sync)
- 项目根 `MEMORY.md` 索引有写

**相关 feature (平级, 非依赖)**:

- `ghost-filesystem-desktop` — 认知面 (pin/update/frame). 两个 feature
  contract 层解耦, channel 层未来合体 (desktop channel 用
  `import_channels(build_file_editor_channel(container))` 一行合体).
  desktop FEATURE.md 交叉表应有一行指向本 feature (仍 skip, 待 desktop
  channel 落地时一并收).
- `matrix-cell-governance` / `cells-cli` — dogfooding 阻塞项, 依赖它们
  完成后 workspace manifests providers 姿态才稳定.