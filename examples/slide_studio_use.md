# Slide Studio 使用指南

Slide Studio 是 MOSShell 中的一个强大功能模块，允许 AI Agent 通过图形界面展示和讲解幻灯片素材。本指南将详细介绍如何使用 Slide Studio。

## 环境准备

### 1. 安装依赖

在项目根目录下执行以下命令：

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate

# 安装所有依赖（包含 Slide Studio 所需依赖）
uv sync --all-extras
```

### 2. 配置环境变量

请参考 [README.md](../README.md) 中的"创建环境和配置环境变量"章节，配置好 `.env` 文件。

## 启动和使用 Slide Studio

### 1. 启动 MOSShell Agent

```bash
.venv/bin/python examples/moss_agent.py
```

### 2. 与 AI Agent 交互

启动后，与 AI Agent 对话：

- **打开 Slide Studio**：告诉 AI "打开slide" 或 "启动幻灯片"
- **AI 会打开一个 UI 窗口**，显示 Slide Studio 界面
- **选择素材**：告诉 AI "打开[素材名]" 或 "讲解[素材名]"

### 3. 识别pptx自动转换为slide assets

```bash
.venv/bin/python examples/scripts/slide_studio_converter.py
```

## 素材文件结构

Slide Studio 的素材存储在 `examples/.workspace/assets/slide_studio/` 目录下。

### 素材文件夹结构

```
examples/.workspace/assets/slide_studio/
├── [素材名]/
│   ├── .meta.yaml          # 素材元数据文件（可选）
│   ├── 1_image.png         # 第一张图片
│   ├── 1_image.png.md      # 第一张图片的说明文档
│   ├── 2_image.png         # 第二张图片
│   ├── 2_image.png.md      # 第二张图片的说明文档
│   ├── 3_image.png         # 第三张图片
│   ├── 3_image.png.md      # 第三张图片的说明文档
│   └── ...                 # 更多图片和说明文档
```

### 关键文件说明

#### .meta.yaml 文件（可选）

第一次打开素材时如果没有此文件会自动创建。格式参考示例：

```yaml
# from class: ghoshell_moss_contrib.channels.slide_studio:SlideConfig
created_at: 1770738941.8579
description: ''              # 素材描述
name: ''                     # 素材名称
origin_filetype: ''          # 原始文件类型
updated_at: 1770738941.8579  # 更新时间
```

#### 图片说明文档 (.md 文件)

每个图片都需要一个对应的 `.md` 文件，文件名格式为：`[图片文件名].md`

**文件格式示例：**

```markdown
---
title: 人机协作：从工具到伙伴的温暖未来
outline: "画面解读：这张图传递的人机信任与协作信号
核心观点：AI 的本质是放大人类能力，而非取代
价值启示：人机协同如何让工作与生活更美好"
---

# 演讲词
该页旨在阐述人机协作的核心理念，即人工智能并非旨在取代人类，而是作为增强人类能力的协作伙伴。其核心观点是，AI 擅长处理数据、执行重复性任务，而人类则在决策、创造力和情感交流方面具有优势，二者结合能够实现高效的协同工作模式，共同创造更大的价值。

# FAQ
Q：人工智能的体现方式目前有几种？
A：Chatbox、工具、具身智能等。
```

**关键字段说明：**

- **头部元数据**（YAML格式）：
  - `title`：幻灯片标题
  - `outline`：内容大纲，AI 会基于此进行讲解
- **正文内容**：可以包含任意 Markdown 格式的内容
  - 建议使用标题（如 `# 演讲词`、`# FAQ`）来组织内容结构
  - AI 会根据这些标题来组织讲解内容

## 重要注意事项

### 图片播放顺序

**关键特性**：Slide Studio 的图片播放顺序是按照**系统默认的文件排序**来确定的。

这意味着：

1. 图片会按照文件名在文件系统中的自然排序进行播放
1. 建议使用数字前缀（如 `1_`、`2_`、`3_`）来确保正确的播放顺序
1. 避免使用可能引起排序混乱的文件命名方式

### 最佳实践

1. **文件命名规范**：

   - 图片文件：`1_image.png`、`2_image.png`、`3_image.png`...
   - 说明文件：`1_image.png.md`、`2_image.png.md`、`3_image.png.md`...

1. **内容组织**：

   - 每个 `.md` 文件头部必须包含 `title` 和 `outline` 字段
   - 正文内容可以使用任意 Markdown 格式
   - 建议使用清晰的标题结构帮助 AI 组织讲解内容

1. **素材管理**：

   - 第一次使用新素材时，Slide Studio 会自动创建 `.meta.yaml` 文件
   - 可以后续编辑 `.meta.yaml` 文件来完善素材信息

## 故障排除

如果遇到问题：

1. 确保所有依赖已正确安装：`uv sync --all-extras`
1. 检查 `.env` 文件配置是否正确
1. 确认素材文件夹结构符合要求
1. 检查图片和对应的 `.md` 文件是否存在且格式正确

现在您可以开始使用 Slide Studio 来创建和展示精彩的幻灯片内容了！
