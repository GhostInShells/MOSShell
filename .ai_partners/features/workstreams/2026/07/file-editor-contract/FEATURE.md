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
status: in-progress
status_note: vendor + contracts + core + 45 tests all green, ready for channel assembly
title: File Editor Contract — vendor openhands-aci 为薄契约，与 Desktop 平级
updated: '2026-07-16'
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

## 交付清单 (v1)

- [x] vendor 拷贝 + `UPSTREAM.md` + LICENSE 归位
- [x] `contracts/file_editor.py` (ABC + Result + 异常, 232 行)
- [x] `core/file_editor/_default.py` (DefaultFileEditor + 异常转译)
- [x] `core/file_editor/__init__.py` (公开 API)
- [x] `tests/ghoshell_moss/core/file_editor/test_file_editor.py` (45 测试, 全绿)
- [ ] `ghost-filesystem-desktop/FEATURE.md` "与关联基建的交叉" 表追加一行指向
  本 feature (skip — 文件被 parallel incarnation 同时修改, 等下次 desktop session 补)
- [x] `architecture.py` 加 import 索引 (`file_editor_impl`)

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

## 装配 (未来, 不在 v1 范围)

Channel 层未定, 见 Implementation Notes 的 A / B 两案. K1 判定 A 更好
但依赖 desktop channel (K33) 落地. Desktop channel 落地由 `ghost-
filesystem-desktop` 那边推进, 本 feature v1 结束点为**契约 + 实现 + 单测**.

## 与 ghost-filesystem-desktop 的交叉

| 维度 | ghost-filesystem-desktop | file-editor-contract |
|---|---|---|
| 辖域 | 认知面 (pin/update/frame) | 写路径 + 结构化 view |
| 契约文件 | contracts/desktop.py | contracts/file_editor.py |
| Core 目录 | core/desktop/ | core/file_editor/ |
| Channel 装配 | 未来 desktop channel (K33) | 复用 desktop channel (K1 A 案) |
| 共享空间边界 | root + PathOutsideRootError (Grounds 管辖) | 不共享 — K4 落地改主意, 两个 contract 彻底解耦 |
| 状态 | in-progress | in-progress (v1 已交付, 装配+联动 未完成) |

---

## 给下一个模型实例

这份 FEATURE.md 是 2026-07-13 立项、07-13/14 Claude Opus 4.7 (1M) 完成 v1
落地时的完整认知记录. 以下是最小路径还原当前状态:

**核心事实**:

- v1 已交付: contracts + core + 45 单测. **不在 scope**: channel 装配, pin
  联动, tmp 大文件策略 (这些等具体需求或 desktop channel 落地时一起做).
- 为什么一个 file editor 要单独 feature: K1 有判据 — FileEditor 与 Ground
  的最小依赖不同, 塞一起就是老 Desktop 融合病灶第三次复发.
- 为什么 vendor 而不是 pip install: K2 + K9 — upstream ~30 依赖, 净依赖归零.

**你要做的 (拿到 file-editor-contract 后要看或改时)**:

```
# 1. 确认当前状态
moss features status file-editor-contract
python -m pytest tests/ghoshell_moss/core/file_editor/ -q  # 应 45 全绿

# 2. 读契约 (3 分钟掌握全貌)
moss codex get-interface ghoshell_moss.contracts.file_editor

# 3. 读 vendor 差异 (30 秒)
cat src/ghoshell_moss/core/file_editor/_openhands/UPSTREAM.md

# 4. 上游升级: UPSTREAM.md 记录了 patch 清单, 换 commit 后按列表重 apply
```

**已知未决 (等你或下个 incarnation 推进)**:

1. Channel 装配 — 与 desktop channel 合体 (A 案) 还是独立 (B 案)?
   人类倾向 A. 需等 desktop channel (K33) 落地.
2. Pin 联动 — 编辑命中已 pin 地址时触发 stale 标记. Channel 层
   composition, 不进契约. 设计和单测都还没.
3. tmp / 大文件 — 当前 10 MB cap 直接抛 FileValidationError. 是否要
   做 truncation + cache-on-disk 等, 暂无痛感, 不加.

**相关的 memory 文件**:

- `feedback_io_contract_sync.md` — K3 依赖的这个原则 (IO 契约默认 sync)
- 项目根 `MEMORY.md` 索引有写

**相关 feature (平级, 非依赖)**:

- `ghost-filesystem-desktop` — 认知面 (pin/update/frame). 两个 feature
  contract 层解耦, channel 层未来合体. desktop FEATURE.md 交叉表应
  有一行指向本 feature (这次 session 跳过了, human 或另一个化身补).