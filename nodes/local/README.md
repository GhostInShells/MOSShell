# nodes/local

**本地试验物料区。** 这里的东西不进版本库，只在本地持续存在。

## 是什么

放本地研究 / 试验的物料：

- 逆向工程的成果（反汇编器、协议笔记、APK）
- 抓包记录、一次性脚本
- 任何「有用但不宜公开」的东西

## 不是什么

**不是 MOSS 的开箱 node。** 这里的目录：

- 没有 `NODE.md`，不会被 `matrix.nodes:list` 发现
- 不会被 `moss nodes run` 起来
- 不参与开箱能力面

要做一个真正的 node，请放到 `nodes/<category>/<name>/`（带 `NODE.md`）。

## 机制

`nodes/local/.gitignore` 忽略本目录下的一切（除它自己和本 README）。
新增物料直接丢进来，不需要改根 `.gitignore`。

## 目录

| 路径 | 是什么 |
|---|---|
| `ky_ufo/` | FLOW-UFO / KY-UFO 系玩具无人机的控制逆向（Body 类能力的方言研究） |
