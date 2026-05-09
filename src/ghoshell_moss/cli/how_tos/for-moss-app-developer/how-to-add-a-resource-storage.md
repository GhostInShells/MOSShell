---
title: How to add a resource storage
description: 理解 Resource 体系并注册新的资源存储：Meta、Storage、Factory、声明、验证
---

# How to Add a Resource Storage

MOSS 用 `scheme://host/path` 作为全局资源句柄。任何数据——图片、文档、记忆——都通过这个字符串引用和传递。

## 核心概念

```bash
moss codex get-interface ghoshell_moss.contracts.resource
```

四个抽象：

- **ResourceMeta** (Pydantic) — AI 可读的元信息。`host` + `path` + `description`，`locator` 是计算属性 `scheme://host/path`
- **ResourceItem[Meta, Data]** — meta 立即可用，`get()` 懒加载实际数据
- **ResourceStorage** — 单 (scheme, host) 的 CRUD 后端，带 `usage()`/`help()` 自解释
- **ResourceRegistry** — 跨 scheme+host 的路由层

## locator 为什么重要

`scheme://host/path` 是一个**可在任何地方传递的句柄**：

- 两个 agent 之间分享资源，传一个字符串
- 存储在对话历史里，下次 session 还能找回
- 跨 app、跨进程，只要连同一个 Registry，就能通过 locator 拿到同一个资源

它不是一个文件名，不是一个 URL，它是一个**类型化的全局指针**。

## 做三件事

### 1. 定义 Meta + Item + Storage

参考开箱实现：

```bash
moss codex get-source ghoshell_moss.core.resources.local_image:LocalImageStorage
```

- Meta 继承 `ResourceMeta`，固定 `scheme()`，加业务字段
- Item 继承 `ResourceItem`，`get()` 做实际 I/O
- Storage 继承 `ResourceStorage`，实现 CRUD + `usage()` + `help()`

### 2. 实现 ResourceStorageFactory

```bash
moss codex get-interface ghoshell_moss.contracts.resource:ResourceStorageFactory
```

`ResourceStorageFactory` 是无副作用的描述符：

- `scheme()` / `host` / `description()` — 自解释，告诉 AI 这是什么
- `factory(container)` — 真正实例化 `ResourceStorage`

参考实现：

```bash
moss codex get-source ghoshell_moss.core.resources.local_image:LocalImageResourceProvider
```

### 3. 在 manifests 里声明

在你的 workspace 的 `src/MOSS/manifests/resources.py`：

```python
from your_package import YourStorageFactory
your_storage = YourStorageFactory()
```

不需要改启动代码、不需要理解 Matrix。声明即生效。

## 验证

```bash
# 列出已声明的资源
moss manifests resources

# 在 REPL 里交互
moss repl
> matrix.resources().schemes()            # 列出 scheme
> matrix.resources().usage("pil-image")   # 查看用法
> item = await matrix.resources().get("pil-image://workspace-assets/beach_photo")
> item.meta                                # 元信息，立即可用
> data = await item.get()                  # 实际数据
```
