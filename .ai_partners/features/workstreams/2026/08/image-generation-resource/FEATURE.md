---
title: Image Generation Resource
status: draft
priority: P2
created: 2026-08-05
updated: 2026-08-05
depends: [matrix-resources]
milestone:
description: >-
  将火山引擎 doubao-seedream 生图能力封装为可查询的 Matrix resource — 文本可查的
  生成记录存储 (generation log), 图是挂在记录上的可选 payload, 经引用解析.
---

# Image Generation Resource

> Use `moss features set-status image-generation-resource <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

火山引擎 doubao-seedream 图片生成 (一次 curl 即可调用) 是极薄的外部能力, 适合做 Matrix
resources 的第一个"生成类"资源投影: 把生图结果变成可查询、可跨上下文引用的资源.

**定位**: P2, 不是 8 月重点。但它是产品形态里可验证的对象——协议面、通道面、资源面三层的
组装是"一次简单的封装"背后真正的机制戏, 适合作为最小完整样例在产品中验证与演示。

讨论收敛出的形状 (2026-08-05): 这不是"生图 resource", 而是一个 **generation log 资源**
——文本可查的生成记录存储, 图作为挂在记录上的可选 payload。四个机制问题驱动了设计:

1. 它究竟是不是一个 local_image 存储对象? —— 不是 (KD1)。
2. 调用逻辑是同步 command 还是异步 command? —— 后台 command (KD2)。
3. image 走文件协议 vs matrix-resources 协议? —— 后者可跨 OS (KD3)。
4. 生成结果不一定要看到图 —— 读图是另一个能力 (KD4)。

## Design Index

- 承接: `matrix-resources` FEATURE.md (统一寻址 + KD4 引用逃生门 + KD9 协议极薄)
- 范本: `core/resources/local_image.py` (ResourceInfo/Item/Storage 三元组 + JSONL + 文件系统)

## Key Decisions

### 1. 不是 local_image 存储对象 — generation log + 引用

**选择**: doubao-image 的身份是**生成记录 + 引用**，不是字节。local_image 是 data-plane
存储 (`get()` 返回 `Image.Image`)，doubao-image 的认知面是文本 (prompt、params、locator、
引用)。storage backend 与 resource shape 正交——问题不是"继承哪个类"，而是"图这个 payload
归谁、引用怎么解析"。

**v1 取舍**: 自包含 (JSONL + 文件，plumbing 照抄 local_image)，但 meta 里留一个
**image_ref 槽位** (URI/URL 表达，不硬绑定本地 `file://` 字符串)，让"索引 + 引用"升级
不动 scheme 契约。

### 2. 调用 = 后台 command (create_signal_task), 非同步

**选择**: 生图 10-60s，同步 command 会卡死 channel 内顺序执行。用
`CommandUtil.create_signal_task` (channel_builder.py:247) 做后台 command：command 内注册
async closure，不阻塞返回，closure 完成后结果 Signal 推给 mindflow。

**open item — pending→ready 生命周期**:
- (a) command 返回占位 ack，closure 完成后 Signal 带回最终 locator → storage 无半成品，最简
- (b) command 先铸 locator (storage 预留 `{id: pending}`)，URI 从生成那一刻稳定 →
  "compact 不遗忘"更早成立，代价是 meta 多 status 字段 + put 支持填充预留槽

v1 倾向 (a)。但 (b) 值得记账：ghost 若想从生成那一刻就保存 locator 跨上下文引用，
(a) 的临时句柄会失效。

### 3. 引用按消费方解析 — 本机路径 / 跨 OS HTTP URL

**选择**: 沿用 matrix-resources KD4 的逃生门——大 payload 不走网络膜，data face 放**引用**：
本机文件路径，跨机 HTTP URL。协议层不特判，storage 决定。v1 走文件路径完全合法，usage()
声明"本机仅文件引用"；跨 OS 是同一字段的 v2 升级，不是协议变更。

**推论**: image_ref 用 URI/URL 表达，本机 registry 能解析、跨机也能解析。硬绑定本地路径
字符串就把跨 OS 的路堵死。

### 4. 生成结果不一定要看到图 — 读图是另一个能力

**选择**: 默认渲染 = 文本 (meta JSON：prompt、params、locator、引用)，不是图。messages
face 给 ghost 的就是文本。"看图"是另一个能力：多模态消费方 (vision ghost / 图片分析
channel) 主动走 data face 拉字节。RESOURCE_TYPE 可以是自己的 generation record，
as_messages 默认文本——类型面留给代码，认知面给 ghost (matrix-resources KD1 两面论)。

**推论**: 对纯文本 ghost，引用可能永远不被 deref，它只是查日志。贴合 KD9"资源不是上下文
全集"——上下文里只进 locator 和 prompt 摘要，不进图。

## Implementation Notes

- **仓库现状**: `src/` 下无任何 doubao/volces/ark 使用 (grep 确认)。wrapper 用裸 HTTP
  (httpx POST 到 ark endpoint) 即可，不需要火山 SDK。
- **待实测**: seedream 单次调用响应是 base64 还是 URL——决定 wrapper 里是"解码存盘"还是
  "下载存盘"，二选一。
- **query**: keyword 匹配 prompt 即可 (v1)，不需要 embedding。embedding 只买语义召回
  ("kitten" 搜 "猫")，成本是每次生成多一次 embedding 调用 + 向量索引。接口留语义搜索插槽
  (contracts/resource.py 已声明"查询语义由后端定义")。真需要时澄清是 prompt 语义还是
  image 语义 (图-图相似) —— 后者是不同量级的 feature。
- **依赖**: 本 feature 建在 matrix-resources 的协议面上；若 matrix-resources 未落地
  (as_messages / 引用逃生门)，可先只做进程内 storage + channel command。
