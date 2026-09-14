# avatar — 安装

独立 venv node。Python 侧只有底座 + 一个 HTTP/WS server；渲染在浏览器侧，所以没有
Live2D SDK 一类的 Python 依赖。**浏览器侧三个 JS 与模型资产都不可分发**（见 vendor/README.md
与 Live2D 条款），需要每台机器本地自建。

## 1. Python venv

```bash
cd nodes/live2d/avatar
uv sync
```

## 2. 浏览器侧 JS（vendor/）

`pixi` 与 `pixi-live2d-display` 是 MIT，可直接拉：

```bash
cd vendor
curl -sL -o pixi.min.js    "https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js"
curl -sL -o cubism4.min.js "https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js"
```

> Pixi 必须用 6.x —— `pixi-live2d-display@0.4.0` 的 peer 依赖是 `@pixi/*@^6`，
> 用 Pixi 7 会渲染出黑剪影重影。

**Cubism Core 是专有库，手动下载**：到 live2d.com 取 Cubism SDK for Web，把
`Core/live2dcubismcore.min.js` 拷到 `vendor/live2dcubismcore.min.js`。
（这一步本身就是授权条款的执行，无法用 curl 替代。）

## 3. 模型资产（avatars/*/model/）

模型包不可分发，从官方样例克隆后拷入：

```bash
git clone --depth 1 https://github.com/Live2D/CubismWebSamples /tmp/cws
mkdir -p avatars/hiyori/model
cp -R /tmp/cws/Samples/Resources/Hiyori/. avatars/hiyori/model/
```

每个 `avatars/<name>/model/` 里必须有一个 `*.model3.json`（入口）。

## 4. 标记安装

```bash
moss nodes install nodes/live2d/avatar
```

## 5. 运行

```bash
moss nodes run nodes/live2d/avatar -- --avatar hiyori
```

打开启动日志里的页面地址即可看见形象。
