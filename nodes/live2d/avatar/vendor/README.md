# vendor/ — 浏览器侧依赖（本地自建，不入库）

这个目录放渲染页面的第三方 JS。**全部 gitignore**（见 node 根 `.gitignore`），原因是
授权：`live2dcubismcore.min.js` 是 Live2D 专有库（条款明写不得向第三方再分发），
其余文件是第三方产物、按项目约定也不入库。每台机器按 `../INSTALL.md` 自建。

页面按顺序加载三个文件：

| 文件 | 作用 | 来源 |
|---|---|---|
| `pixi.min.js` | WebGL 渲染器 | npm `pixi.js@6.5.x`（MIT） |
| `live2dcubismcore.min.js` | Cubism Core（moc3 运行时） | Live2D Cubism SDK for Web（**专有，手动下载**） |
| `cubism4.min.js` | pixi-live2d-display，Cubism 4 的薄封装 | npm `pixi-live2d-display@0.4.0`（MIT） |

`cubism4.min.js` 内部引用全局 `Live2DCubismCore`，所以 Core 必须排在它前面。

> **Pixi 必须 6.x**：`pixi-live2d-display@0.4.0` 的 peer 依赖是 `@pixi/*@^6`，
> 用 Pixi 7 会渲染出黑剪影重影（渲染器 API 不兼容）。

> 注：选 pixi-live2d-display 而非官方 CubismWebFramework，是因为后者是 TS 源码、需要
> esbuild 打包一步；前者是预构建 UMD，页面端零构建。两者对外暴露的是同一份 WS 协议，
> 页面是协议背后的实现细节，以后想换官方 SDK 只改 `app.js`，驱动侧无感。
