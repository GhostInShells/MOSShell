---
name: live2d-avatar-setup
description: 自建 Live2D avatar node 环境 — vendor 三个 JS + 模型资产 + venv。之后 moss nodes run 就能起来。
---

# 自建 avatar 环境

目标：让 `moss nodes run nodes/live2d/avatar` 能起来并渲染一个形象。
浏览器侧 JS 与模型资产都不可分发，所以要本地自建（详细见 `INSTALL.md`）。

## 1. Python venv

```bash
cd nodes/live2d/avatar && uv sync
```

## 2. vendor 三个 JS（页面按此顺序加载）

```bash
cd vendor
curl -sL -o pixi.min.js      "https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js"
curl -sL -o cubism4.min.js   "https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js"
# Core 专有，手动下载：live2d.com 取 Cubism SDK for Web → Core/live2dcubismcore.min.js
# Pixi 必须 6.x：pixi-live2d-display 0.4.0 的 peer 依赖是 @pixi/*@^6, 7.x 会黑剪影重影
```

## 3. 模型资产

```bash
git clone --depth 1 https://github.com/Live2D/CubismWebSamples /tmp/cws
mkdir -p avatars/hiyori/model
cp -R /tmp/cws/Samples/Resources/Hiyori/. avatars/hiyori/model/
```

## 4. 验证

```bash
moss nodes install nodes/live2d/avatar
moss nodes run nodes/live2d/avatar -- --avatar hiyori
# 打开日志里的 http://127.0.0.1:8770/ 看形象
```

验证层次（低 → 高）：

1. **起得来** — 日志有 `page at http://...`，无 traceback。
2. **页面加载模型** — 浏览器开页面，无控制台报错，形象出现。
3. **命令驱动参数** — 通过 moss-shell / CTML 发一条参数命令，页面上形象有可见变化。
