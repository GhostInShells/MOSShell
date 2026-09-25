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

### command 有时间轨迹

命令占时, 不是 fire-and-forget: 参数命令占默认缓动时长 (0.3s), 动作命令占
`motion3.json` 的 `Meta.Duration` (可用 `hold` 覆盖), 结束在 `finally` 里复原动作。
这是"时间第一公民"的落地 —— 同轨命令因此串行, 异轨并行。动作文件全部 `Loop:True`
(实测 hiyori), 所以"结束"是 driver 自己计时后发 `clear_motion` 停掉, 不靠页面回报。

### idle 是 driver 仲裁的待机循环, 跑在 build.idle

`build.idle` 在无 blocking 命令 (含子命令) 时进入, 新 blocking 命令到达即取消 ——
子命令会取消父 idle (内核契约, 见 tests/ghoshell_moss/default/core/channels/test_py_channel.py)。
待机仲裁循环跑在这个生命周期里: 空闲超过 `idle.delay` 才进待机。说话 (speaking) 与
点按 (on_tap 后台 play) 不是命令, 内核看不到, 由 `speaking` / `_foreground` 在循环里
额外让位。部件级 idle (眨眼/呼吸) 是 SDK 原生, 由 `AVATAR.md` 的 `idle.parts` 开关。

### animation 轨迹编程

模型在 `animations.py` 里写 N 个 `async def`, 每个是一条动画轨迹: 编译 (codex Compiler)
时注入 `get_avatar()` + `asyncio`, 反射协程函数经 `channel_builder.new_command` 变成
command, 打包成 `ChannelModule` 挂到主 channel (`with_module`)。同名 module 覆盖 =
热更新, 由 `reload_animations` 命令触发。命令 blocking=True, 它的 await 序列就是时间轨迹。

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
- **SDK 默认眨眼与 idle motion 自带眨眼会叠加**（已用 `idle.parts.blink: false` 规避）：
  hiyori 的 idle 动作本身就驱动眼开闭参数，SDK 的自动眨眼在动作间隙又驱动一遍，
  两次眨眼贴太近会闪一下。模型包自带眨眼时，把它在 `AVATAR.md` 里关掉。
- **待机动作无条件覆写它驱动的参数**：idle 循环动作逐帧覆写头/眼/嘴等参数。blocking
  参数命令现在会打断待机（内核取消 `build.idle`）、让参数短暂生效，但待机回来后动作
  曲线又覆写回去 —— "参数遮挡/权重"仍是要讨论的问题，当前未做。
