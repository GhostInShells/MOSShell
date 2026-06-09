# Moss in Reflex

基于 [Reflex](https://reflex.dev/) 构建的 MOSS 前端界面，通过 ZMQ 通道与 MOSS AI Agent 进行通信，实现动态布局渲染和流式内容更新。

## 简介

本项目是 ghoshell-moss 框架的 Reflex Web 前端适配层。它将 MOSS Agent 的消息事件（流式文本、列表操作、结构化数据、布局切换等）实时映射到 Reflex 组件状态上，从而在浏览器中动态展示 AI Agent 的输出。

核心思路：MOSS Agent 通过 ZMQ 发送事件 -> Reflex 后端接收并解析事件 -> 自动驱动前端组件状态更新 -> 浏览器实时渲染。

## 架构

```
MOSS Agent (ZMQ)  --->  Moss in Reflex (Reflex App)  --->  Browser
                            |
                        asyncio.Queue
                            |
                    事件分发 (Event Router)
                            |
                  +---------+---------+
                  |                   |
              Layout 切换        字段操作
           (SimpleLayout,     (stream / set /
            可扩展...)        append / pop / clear / pydantic)
```

### 关键模块

| 模块 | 说明 |
|------|------|
| `moss_in_reflex/moss_in_reflex.py` | 应用入口，包含 Reflex App、State、MOSS 通信和生命周期管理 |
| `framework/events.py` | 事件模型定义（`StreamEvent`、`SetEvent`、`AppendEvent`、`PopEvent`、`ClearEvent`、`LayoutEvent`） |
| `framework/layouts/simple.py` | 示例布局组件，展示标题、副标题、标签、段落和 Markdown 内容 |
| `framework/layouts/helpers/mixin.py` | 布局组件的 `NameMixin` 抽象基类 |
| `framework/runtime/event_generator.py` | 根据 State 类的类型注解动态生成事件处理方法 |
| `examples/run_with_simple_agent.py` | 示例：启动一个 MOSS SimpleAgent 并通过 ZMQ 连接到本前端 |

## 事件类型

| 事件 | 说明 |
|------|------|
| `LayoutEvent` | 切换当前布局 |
| `StreamEvent` | 流式追加文本到 `str` 类型字段 |
| `SetEvent` | 设置任意字段值（支持 JSON 设置 BaseModel 字段） |
| `AppendEvent` | 向 `list` 类型字段追加元素（支持 JSON 追加 dict/BaseModel 元素） |
| `PopEvent` | 弹出 `list` 类型字段的最后一个元素 |
| `ClearEvent` | 清空指定字段 |

## 支持的字段类型

| 字段类型 | 自动生成的方法 | 说明 |
|---------|--------------|------|
| `str` | `stream_<field>`, `clear_<field>` | 流式文本追加、清空 |
| `list[str]` | `push_<field>`, `pop_<field>`, `clear_<field>`, `stream_append_<field>` | 列表操作 + 流式追加到最后一个元素 |
| `list[T]`（T 为 dict 或 BaseModel） | `push_<field>`, `pop_<field>`, `clear_<field>` | 通过 JSON 操作列表 |
| `BaseModel` 子类 | `set_<field>`, `clear_<field>` | 通过 JSON 设置/重置结构化数据 |

## 媒体与图表组件

布局中可使用以下 Reflex 内置组件渲染富媒体内容：

- **图片**：`rx.image(src=url)` — 展示 URL 或 assets 路径的图片
- **视频**：`rx.video(url=url)` — 嵌入视频播放器
- **图表**：`rx.recharts.*` — 折线图、柱状图、饼图等，数据字段通常为 `list[dict]`

## 快速开始

### 环境要求

- Python >= 3.12
- uv 或 pdm 包管理器

### 安装

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 运行前端

```bash
reflex run
```

启动后访问 `http://localhost:3000` 查看界面。

### 运行示例 Agent

在另一个终端中运行：

```bash
python examples/run_with_simple_agent.py
```

该示例会启动一个 MOSS SimpleAgent，通过 ZMQ（`tcp://127.0.0.1:9528`）与 Reflex 前端通信。

## 扩展布局

1. 在 `framework/layouts/` 下创建新的布局类，继承 `rx.ComponentState` 和 `NameMixin`：

```python
import reflex as rx
from pydantic import BaseModel, Field
from framework.layouts.helpers.mixin import NameMixin

# 可选：定义 Pydantic 模型用于结构化数据
class CardData(BaseModel):
    title: str = Field(default="")
    image_url: str = Field(default="")
    description: str = Field(default="")

class MyLayout(rx.ComponentState, NameMixin):
    """My custom layout description."""
    content: str = ""
    images: list[str] = []
    card: CardData = CardData()
    chart_data: list[dict] = []

    @classmethod
    def name(cls) -> str:
        return "my_layout"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.vstack(
            rx.text(cls.content),
            rx.foreach(cls.images, lambda url: rx.image(src=url, width="200px")),
            rx.heading(cls.card.title),
            rx.text(cls.card.description),
            rx.recharts.line_chart(
                rx.recharts.line(data_key="value"),
                rx.recharts.x_axis(data_key="name"),
                data=cls.chart_data,
                width="100%", height=300,
            ),
            **props,
        )
```

2. 在 `moss_in_reflex/layouts.toml` 的 `layouts` 数组中注册：

```toml
layouts = [
    "framework.layouts.simple.SimpleLayout",
    "framework.layouts.my_layout.MyLayout",
]
```

框架会自动为新布局的 State 字段生成对应的事件处理方法。

## 技术栈

- **Reflex** - Python 全栈 Web 框架
- **ghoshell-moss** - MOSS AI Agent 框架
- **ZMQ (ZeroMQ)** - 进程间通信
- **Pydantic** - 数据模型验证
- **Tailwind CSS v4** - 样式（通过 Reflex 插件集成）

## License

Apache License 2.0
