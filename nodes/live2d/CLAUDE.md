# Live2D Avatar

MOSS 的 Live2D 虚拟形象躯体：给 ghost 一个可交互、可被 CTML 驱动的数字人身体，
人类在浏览器里看见它。

**所属 workstream**: `ghost-live2d-avatar`（FEATURE.md 是设计真相；本文件是范式真相）。

## 首次进入

1. 读 `avatar/NODE.md` —— 节点机制（Ghost 挂载后读的也是它）
2. 读 `avatar/avatars/README.md` —— 形象套件契约（怎么加/改一套形象）
3. 读 `avatar/INSTALL.md` 与 `avatar/skills/` —— 自建环境与验证
4. 回溯设计决策 → `git log -- .ai_partners/features/workstreams/2026/09/ghost-live2d-avatar/`

## 方法论（范式真相）

以下决策不由代码自然表达，每个进入本目录的实例必须内化：

### 驱动与模型解耦，粘合面是约定

驱动是代码（随仓库分发），模型是数据（本地持有、不分发）。一套形象 = `avatars/<name>/`
一个自包含目录：`channel.py`（命令面）+ `model/`（资产）。发现规则两级、显式接管不合并：
有 `channel.py` 用它，没有则从模型自带的 `cdi3.json` / `model3.json` 自动映射一份示范面。

### 模型是唯一开发者

写 `channel.py` 的是模型（AI），不是人类。所以驱动把能力**事件化封装**在 `Avatar`
一个对象上（`param`/`motion`/`expression`/`reset`/`backdrop`），作者不碰 WS、页面、
渲染。自动映射只是兜底示范，证明"标准包丢进来就能驱动"，不是主路径。

### 模型资产与官方 JS 不可分发

Live2D 条款明写不得向第三方再分发。因此 `avatars/*/model/` 与 `vendor/` 全 gitignore，
每台机器按 `INSTALL.md` 自建。Core（`live2dcubismcore.min.js`）是专有库，必须手动下载；
`pixi` / `pixi-live2d-display` 是 MIT 可 curl。这是授权条款的执行，不是规避。

### command 即真相

驱动不向页面回读参数状态。`Avatar.state()` 返回"被命令过的值"，就是真相。这避免了
双驱动抖动，也让命令面在没有浏览器时仍可测（moss-shell 直接验证）。

### 换形象 = 重启 node

没有运行期 `switch_model`——状态太重。形象身份是 argument（`--avatar`），走启动参数。

### 页面必须同源

页面、模型资产、WebSocket 全由 node 的一个 aiohttp server 提供，否则 WS 跨域被浏览器拦。
页面是协议背后的实现细节：对外暴露的只有 WS 协议，换渲染库只改 `app.js`。

## 目录

```
nodes/live2d/
├── CLAUDE.md          # 本文件 — AI 认知入口（范式真相）
├── README.md          # 面向开发者
└── avatar/            # node
    ├── NODE.md        # 节点机制（Ghost 挂载后读）
    ├── INSTALL.md     # 自建步骤（venv + vendor JS + 模型）
    ├── main.py        # 装配：select → Avatar → bridge → provide_channel
    ├── src/avatar_node/   # 驱动框架
    │   ├── cubism.py      #   model3/cdi3 解析 + 噪声过滤
    │   ├── lexicon.py     #   共享词表（分组名/参数名归一 + 量纲提示）
    │   ├── avatar.py      #   Avatar 事件面（command 即真相）
    │   ├── bridge.py      #   同源 server（页面/资产/WS）
    │   ├── mapper.py      #   自动映射（示范路径）
    │   ├── discovery.py   #   套件发现 + channel.py 加载
    │   └── web/           #   页面（index.html/app.js/style.css）
    ├── avatars/       # 形象套件（每套自包含）
    ├── backdrop/      # 通用背板图（与模型包解耦）
    ├── vendor/        # 浏览器侧 JS（gitignore，见 vendor/README.md）
    └── skills/        # setup / add-avatar
```

## 已知问题

- **背板/渲染库是简化的第一版**：选 pixi-live2d-display 而非官方 CubismWebFramework，
  因为后者要 esbuild 打包一步。对外 WS 协议不变，换库只改 `web/app.js`。
- **唇动包络未接**：`model3.json` 的 `Groups.LipSync` 声明已解析进 `spec.lip_sync`，
  但音频包络 → 唇形驱动还没接（依赖 speech 侧的 volume 流，属后续集成）。
