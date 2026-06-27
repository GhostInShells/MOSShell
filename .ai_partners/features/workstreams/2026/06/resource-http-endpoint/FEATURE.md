---
title: Resource HTTP Endpoint — Matrix 资源的 HTTP 访问层
status: completed
priority: P1
created: 2026-06-24
updated: 2026-06-24
depends: []
milestone:
description: >-
  为 Matrix ResourceRegistry 中的资源提供 HTTP endpoint，使视频等大文件资源可通过浏览器直接访问。
  实现为 MOSS App，内嵌 FastAPI + uvicorn server，Range 请求支持视频 seek。
---

# Resource HTTP Endpoint

## Motivation

Matrix 的资源系统（`ResourceRegistry`）已经提供了完整的资源寻址和获取能力：
`scheme://host/path` → `ResourceStorage.get(path)` → `ResourceItem.get()` → 实际数据。

但当前没有外部访问层。一个 Ghost 通过 Channel 拿到了视频资源的 locator
（如 `local-video://media/sample.mp4`），想把它展示给人类用户——没法给浏览器一个 URL。

本 feature 创建 `web/resource_server` app，在 Matrix 上挂一个 HTTP server，
把 locator 映射为 HTTP URL。人类浏览器可以直接播放视频、查看图片、下载文件。

**核心场景**：视频播放。浏览器需要 Range 请求支持 seek、正确的 Content-Type、
以及流式传输不爆内存。图片和文本是附带的 bonus。

## Key Decisions

### 1. 实现为 MOSS App，不是 Channel 也不是 Host 内建

**选择**：独立 App `web/resource_server`，进程隔离，通过 `main(matrix)` 入口获取 Matrix。

**拒绝的替代方案**：
- Host 内建 HTTP server：耦合 Host 生命周期，所有 mode 都带着，不灵活。
- Channel 内嵌：Channel 的语义是"模型通过 CTML 调用的能力"，HTTP endpoint 的消费者是浏览器不是模型，语义不匹配。

**影响**：App 独立管理依赖（`fastapi[standard]` + `uvicorn`），独立启停。需要它的 mode 在 `bringup_apps` 中声明。

### 2. URL 映射：`GET /resources/{scheme}/{host}/{path}` — locator 到 HTTP 的 1:1 映射

**路由设计**：

| 方法 | 路径 | 行为 |
|------|------|------|
| `GET` | `/resources/{scheme}/{host}/{path:path}` | 获取资源，返回原始数据流 |
| `GET` | `/resources/{scheme}/{host}` | 浏览该 host 下的资源列表（JSON） |
| `GET` | `/resources/{scheme}` | 浏览该 scheme 下的所有 host（JSON） |
| `GET` | `/resources` | 列出所有已注册的 scheme 及其 host |
| `GET` | `/health` | 健康检查 |

locator `scheme://host/path` 直接映射为 URL path `/resources/scheme/host/path`。

### 3. HTTP 库选择：FastAPI + uvicorn

**选择**：`FastAPI` + `uvicorn` — Starlette 的 `FileResponse` 已内置 Range/ETag/If-Modified-Since 处理，零额外工作量。人类开发者更偏好 FastAPI 生态。

**讨论**：初始设计选了 `aiohttp`（轻量），在实现阶段切换到 FastAPI（更熟悉、周边生态更好）。两者对核心需求（Range 请求、FileResponse、streaming）的支持等价。

### 4. 视频 Range 请求：依赖 storage 层提供文件路径或流式接口

**核心约束**：`ResourceItem.get()` 返回的是 Python 对象（如 `Image.Image`、`str` 等）。
对于视频，返回什么？如果返回文件路径，HTTP server 可以直接用 `aiohttp.FileResponse`
（零拷贝、内核级 sendfile、自动 Range）。如果返回 bytes/流，server 需要手动实现 Range。

**选择**：分两步走。

第一步（本 feature）：假设存在一个 `VideoResourceStorage`，其 `ResourceItem.get()` 返回
本地文件路径（`str`）。如果没有，本 App 带一个最简实现——一个基于本地目录的文件存储，
scheme 叫 `local-file`，`get()` 返回文件路径。这样立刻就能用。

第二步（后续 feature）：设计通用的 streaming resource 接口，让 `ResourceItem` 支持
`stream()` 方法返回 `AsyncIterator[bytes]` + content-type + size，不依赖本地文件。

**影响**：
- 当前只支持本地文件资源（`FileResponse` 需要 `pathlib.Path`）。
- 视频 seek 由 Starlette 的 `FileResponse` 自动处理（解析 `Range` header，返回 `206 Partial Content`）。

### 5. Content-Type 推断

**选择**：双重 fallback 链：
1. 如果 `ResourceInfo` 有 `content_type` 或 `mime_type` 字段 → 直接用
2. 如果 `ResourceItem.get()` 返回文件路径 → 用 `mimetypes.guess_type(path)`
3. fallback → `application/octet-stream`

`mimetypes` 是标准库，零依赖。

### 6. 安全：只暴露读，不做写

**选择**：只有 GET 和 HEAD，没有 PUT/POST/DELETE。

**拒绝的替代方案**：支持 PUT 上传。拒绝原因：上传语义复杂（multipart、权限、配额），
且 Matrix resource 的写入应该通过 Channel/CTML 走，HTTP 层面只做只读展示。
真有上传需求时再加，不影响当前设计。

### 7. 配置：通过 ConfigStore 体系管理

**选择**：定义 `ResourceServerConfig(ConfigType)`，通过 `get_conf()` / `get_or_create_conf()` 读写配置。持久化为 `configs/resource_server.yml`。

```python
class ResourceServerConfig(ConfigType):
    host: str = "127.0.0.1"
    port: int = 20880

    @classmethod
    def conf_name(cls) -> str:
        return "resource_server"
```

**默认值**：`host=127.0.0.1`（只监听本地），`port=20880`。

不改默认监听 `0.0.0.0`——安全第一，需要外部访问时用户显式改配置。

### 8. App 的行为：不注册 Channel

这个 App 不调用 `matrix.provide_channel()`。它不是给模型用的能力，
是给人类浏览器的。App 的唯一职责是起 HTTP server 并阻塞到 Matrix 关闭。

## Implementation Notes

### 入口骨架

```python
# .moss_ws/apps/web/resource_server/main.py
from fastapi import FastAPI
import uvicorn
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.contracts.configs import ConfigType, get_or_create_conf, get_conf


class ResourceServerConfig(ConfigType):
    host: str = "127.0.0.1"
    port: int = 20880

    @classmethod
    def conf_name(cls) -> str:
        return "resource_server"


async def main(matrix: Matrix):
    conf = get_or_create_conf(matrix.container,
        ResourceServerConfig(host="127.0.0.1", port=20880))
    conf = get_conf(matrix.container, ResourceServerConfig)

    app = create_app(matrix)

    config = uvicorn.Config(app, host=conf.host, port=conf.port)
    server = uvicorn.Server(config)
    await server.serve()
```

### resource_handler 核心逻辑

```python
@app.get("/resources/{scheme}/{host}/{path:path}")
async def get_resource(scheme: str, host: str, path: str):
    locator = f"{scheme}://{host}/{path}"
    item = await registry.get(locator)
    if item is None:
        return JSONResponse({"error": f"Resource not found: {locator}"}, status_code=404)

    data = await item.get()

    if isinstance(data, (str, Path)):
        return FileResponse(data)

    if isinstance(data, bytes):
        content_type = getattr(item.info, "content_type", None) or "application/octet-stream"
        return Response(content=data, media_type=content_type)

    return {"locator": locator, "type": type(data).__name__}
```

### Range 请求的关键认知

浏览器 `<video>` 标签加载视频时，先发一个 `Range: bytes=0-` 的 GET 试探服务器是否支持部分内容。
如果服务器返回 `200` + 全部数据，浏览器可能拒绝播放（或者只能等全部下载完才能 seek）。

Starlette `FileResponse` (FastAPI 继承自 Starlette) 自动处理：
- 解析 `Range` header
- 设置 `Content-Range`、`Accept-Ranges: bytes`
- 返回 `206 Partial Content`
- 支持多段 Range（`bytes=0-1024,2048-4096`）
- 自动 `ETag` 和 `If-Modified-Since`

所以只要 `ResourceItem.get()` 能返回文件路径，视频播放就零额外工作量。

### 浏览器 URL 可见性

App 启动后，在 context messages 或日志中输出 HTTP 地址：
```
Resource HTTP server started at http://127.0.0.1:20880
Resources: http://127.0.0.1:20880/resources
```

Ghost 可以通过读取 App 日志或 Matrix 环境信息获取这个地址，拼出完整的资源 URL
返回给人类用户。例如 locator `local-video://media/demo.mp4`
→ URL `http://127.0.0.1:20880/resources/local-video/media/demo.mp4`。

## Implementation Tasks

### T1: 创建 App 骨架
- [x] `moss apps create web/resource_server`
- [x] 在 `pyproject.toml` 中添加 `fastapi[standard]` + `uvicorn` 依赖
- [x] 编写 APP.md 元数据（description: "HTTP endpoint for Matrix resources"）

### T2: 实现 HTTP server 核心
- [x] `main.py`: `main(matrix)` 入口，创建 FastAPI Application
- [x] `health_handler`: 返回 `{"status": "ok"}`
- [x] `list_schemes_handler`: 从 registry 列出所有 scheme
- [x] `list_hosts_handler`: 从 registry 列出某 scheme 下所有 host
- [x] `list_resources_handler`: 调用 `registry.list_infos(scheme, host)` 返回 JSON
- [x] `resource_handler`: 核心——locator → item.get() → response

### T3: Content-Type 推断
- [x] 对 `FileResponse` 路径，Starlette 自动用 `mimetypes.guess_type()` 推断 MIME
- [x] 对 bytes 返回值，从 `ResourceInfo` 读 `content_type` 或 fallback `application/octet-stream`

### T4: 带一个最简本地文件 storage（bootstrapping）
- [x] 实现 `LocalFileStorage(ResourceStorage)`：scheme=`local-file`，扫描本地目录
- [x] `ResourceItem.get()` 返回文件路径（`Path`）
- [x] `list_infos()` 列出目录下所有文件，支持文件名搜索
- [x] 在 `main()` 中注册到 Matrix ResourceRegistry
- [x] **目的**：立刻能放一个 mp4 到目录里，浏览器访问验证全链路

### T5: 配置外置
- [x] `runtime/configs/resource_server.yaml`：host、port
- [x] `main()` 中通过 `_load_config()` 读取 YAML 配置

### T6: 验证
- [x] `moss apps test web/resource_server` 可启动
- [x] `curl http://127.0.0.1:20880/health` → 200
- [x] `curl http://127.0.0.1:20880/resources` → 返回已注册的 scheme/host 列表
- [x] Range 请求验证：`Range: bytes=0-4` → 206 Partial Content with Content-Range
- [x] 404 处理验证
- [x] Content-Type 推断验证：text/plain, ETag, Last-Modified 自动生成

### T7: 归档
- [x] FEATURE.md status → completed
- [x] Commit FEATURE.md 与代码一同提交

## Related Code

- `src/ghoshell_moss/contracts/resource.py` — ResourceInfo, ResourceItem, ResourceStorage, ResourceRegistry
- `src/ghoshell_moss/core/resources/memory_registry.py` — InMemoryResourcesRegistry（registry 的默认实现）
- `src/ghoshell_moss/core/resources/local_image.py` — LocalImageStorage（参考：scheme 定义、list_infos、get 模式）
- `src/ghoshell_moss/core/blueprint/matrix.py` — `Matrix.resources()` 返回 ResourceRegistry
- `.moss_ws/apps/genkits/video/main.py` — 最简 App 入口参考
- `.moss_ws/apps/ui/reflex/` — 有 HTTP 服务的 App（aiohttp 之外的另一种路线，FastAPI + uvicorn）
