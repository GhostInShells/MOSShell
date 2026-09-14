---
name: 'avatar'
description: 'Live2D 虚拟形象躯体 — 一套形象 = 一个自包含套件, 驱动眼动/唇动/表情/肢体与动作'
category: live2d
singleton: true
exec:
  command: .venv/bin/python
  args: main.py
---

你挂载在一个 Live2D 虚拟形象上。人类在浏览器页面里看见它 —— 页面、模型资产、
WebSocket 由本 node 同源提供，无处跨域。

## 机制

一套形象 = `avatars/<name>/` 一个自包含目录：

- `channel.py` —— 这个形象的命令面与 instruction，用 channel builder 写。**模型是这套
  形象唯一的作者**：想让它给你什么能力，就改这个文件。
- `model/` —— 本地模型资产。Live2D 条款禁止分发，不入库，每台机器按 `INSTALL.md` 自建。

没有 `channel.py` 的套件，由驱动从模型自带的 `cdi3.json` / `model3.json` 自动映射出
一份示范命令面。自动集成只是示范路径，不是主路径。

启动时用 argument 选定套件。**换形象 = 重启 node**，没有运行期切换。

## 查找形象

本地有哪些套件看 `avatars/` 目录。本 node 的 skills 在 `skills/` 下，
覆盖"如何自建环境"与"如何新增一套形象"。

## 使用

选定形象后，它的 channel 就是你的膜 —— 命令签名即接口，直接反射给你。驱动遵守
**command 即真相**：你下过什么命令，就是躯体的当前状态，驱动不向页面回读。

命令语义、协作约定、状态边界在 channel 的 instruction 里，不在此重述。
