---
created: 2026-09-14
depends: []
description: 问题集即思维框架 — 一组问题从上下文抽取结构化自识，作为跨 compact 的会话自识；靠温层时间戳重签把注意力拉回。
milestone: null
priority: P2
status: completed
status_note: frame channel + startup init_frame + dolores wiring landed (2026-09-25)
title: Thinking Frame
updated: '2026-09-25'
---

# Thinking Frame — 问题集即思维框架

> Use `moss features set-status thinking-frame <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

> **本 FEATURE.md 于 2026-09-25 由 claude-opus-4-7 重写。之前的方案已删除，`git log -- .ai_partners/features/workstreams/2026/09/thinking-frame/FEATURE.md` 可查看历史轨迹（含 compass 改名事故与 frame_channel 旧实现的完整背景）。此文件只保留现行方案。**

## Motivation

MOSS 是实时双工运行时，上下文会被 compact。模型需要**跨 compact 存活的、会话级的自我认知**：不是新的记忆系统，也不是 checkpoint——而是**一组问题**，模型从自己的上下文里抽取答案，形成"我现在在哪 / 在和谁 / 在做什么 / 什么不能忘"的结构化自识，作为推理的基础。

问题不是重点，**问题的持续可见**才是重点。一条上下文里的旧消息，KV cache 还在，但后续每一轮 attention 的 query 由**近期 token 的自回归状态**决定——聊什么、注意力就朝什么偏。已读过的框架文本会**被淹没**，不是被忘记。这个 channel 的核心机制不是"存问题"，是**用时间戳重签把 unresolved 问题拽回当前 attention**。

## Key Decisions

1. **`Frame(BaseModel)` 是唯一 artefact。** `label` / `description` / `questions` / `answers` 四字段。同一份数据结构承载三种形态：磁盘（`.frame.yml`）、注入（startup / 代码构造）、模型现场造（`define` 命令 body）。拒绝的选项：`_Frame` 内部类 + 分开的解析代码（继承旧 frame 的两层混合，`answers` 靠 `exclude=True` 排除）——两层拆分（`Frame` 持久化 + `SessionFrame` 运行时）从结构上保证 answers 不落盘，比 `exclude` 硬。

2. **不做发现机制。** `root` 是**文件操作边界**——`load`/`export` resolve 后必须在 root 内，越权 `ValueError`。channel **不扫描目录**。拒绝的选项：filesystem-as-index（旧 frame 的选择，靠 `*.frame.md` 后缀自动发现）——发现需要 label 规约、路径推导、glob 语义，代价高于收益；这一轮直接扔掉。

3. **温层时间戳重签 = 重新进 attention。** frame 作为 `named_notice` fragment，body 是 unresolved 问题列表 + 结尾 `refreshed at: HH:MM`。`refresh_meta` 钩子里检查 `now - last > threshold`（默认 120s），过阈值就更新时间戳。**改一个与内容无关的字段就让整块 fragment 变脏**，触发 named_notices 的差量比对重发（`prompts.py:_notice_delta`）。拒绝的选项：（a）按事件刷（epoch/resolve）——不解决"轮内飘散"；（b）独立 push 通道——重复发内容，浪费 token；（c）模型自定周期——会 drift 到最小值噪声化。**时间戳做主机制，事件改动做辅**。

4. **推拉合一，不省 observe。** `resolve(question_index, answer)` 直接更新内存，unresolved 集合立即变化，下次 refresh 自然反映。观测即事实。`resolved()` 命令是**拉接口**——模型主动想看已答的时候用；平时不刷。拒绝的选项：`resolve` 后不发 notice（"省一次 observe"）——实时双工里攒着就是延迟。

5. **JSON body + JSON Schema。** `define(text__: str)` 的 doc 是**闭包**，refresh 时用 `Frame.model_json_schema()` 生成，写清 CDATA 包裹要求。模型往 tag body 写 JSON，channel `json.loads + model_validate`。拒绝的选项：（a）YAML body——手写 YAML 有冒号纪律，非 channel 特殊契约；（b）pydantic model 直接反射为参数类型——CTML 现有机制里没有此路径，`text__` 是流式 str 参数。

6. **命令面 6 个正交动作**：`load(path, with_answers=False)` / `define(text__)` / `resolve(index, answer, label="")` / `resolved(label="")` / `reset(label="")` / `export(label, filename="")`。拒绝的选项：（a）`reload` 命令——已并入 `load`（同 label 幂等，重设走 `reset` + `load`）；（b）`status` / `list` / `spec` / `unload` / `template`——旧 frame 的膨胀命令面，本轮全部裁掉。

7. **startup 集成走数据结构，不动 startup 机制。** `StartupDoc(BaseModel)` 加 `frame: Frame | None` 字段；`Dolores.__aenter__` 一次读入缓存；`channel()` 从缓存取 `init_frame` 传给 `build_dolores_channel` → `new_frame_channel`。**不删 `command` 字段**（是 startup 的强制首动作能力，与 frame 正交）。`stubs/default.startup.yml` 加了 `frame:` 块提供 orientation 种子，但**不填 command**——开机 seed frame 让 ghost 起来自己看，不塞强指令。

## Deferred (not this feature)

- **真实会话 dogfooding** — channel + startup 都装线了，dolores ghost 可以自己迭代问题库。等实际会话跑过再回头看 refresh_seconds、Frame 字段是否需要调整。
- **frame 库** — 目前只有 `orientation` 一个种子帧（在 startup stub 里）。未来的价值在框架积累与复用；由模型自己 `define` + `export` 攒起来。
- **模型改文件时的 YAML 冒号纪律** — 导出的 `.frame.yml` 若模型用 file editor 改 questions，含 `": "` 的字符串要引号。这是手写 YAML 通用纪律，不写进 channel doc。

## Implementation

- 核心: `src/ghoshell_moss/channels/frame_channel.py`（`Frame` / `SessionFrame` / `new_frame_channel` / `new_frame_channel_from_file`）
- 测试: `tests/ghoshell_moss/channels/test_frame_channel.py`（33 项，覆盖数据结构 / init_frame / load / define / resolve / reset / export / refresh clock）
- dolores 接线: `src/ghoshell_moss/ghosts/dolores/{channel.py, _runtime.py, _startup.py}`；stub `stubs/startup/default.startup.yml` 有 seed frame
- dolores 测试: `test_dolores.py::TestStubsSync::test_load_startup_*` + `TestBuildChannel::test_init_frame_lands_in_notice`

## Method notes

- **named_notice 差量语义**（`prompts.py:_notice_delta`）：`None` = removed 墓碑；`""` (`NAMED_NOTICE_UNCHANGED`) = 零 token 保留；文本 = 变了才发。**时间戳能触发重发正是因为它落在第三类**。complete frame 渲染空串 → channel 返 `None` → 差量渲成 `<label removed/>`。
- **加载时机**：`__aenter__` 里 `self._startup_doc = await asyncio.to_thread(self._load_startup)`。`channel()` 早于 `startup()` 但都读同一个缓存，避免磁盘 IO 竞争。
- **越权保护**：`_resolve_inside_root` 用 `path.resolve().relative_to(root.resolve())` 验证；符号链接会被解析后再判断，无法绕过。
- **doc 闭包**：`chan.build.command(name="define", doc=_define_doc)`——`_define_doc` 每 refresh 调一次生成 docstring，`Frame.model_json_schema()` 也在此嵌入。
