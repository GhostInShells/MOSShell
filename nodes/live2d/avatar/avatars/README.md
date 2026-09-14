# avatars/ — 形象套件契约

一套形象 = `avatars/<name>/` 一个自包含目录：

```
avatars/<name>/
├── channel.py     可选：这个形象的命令面（模型是唯一的作者）
└── model/         模型资产（不入库，见 INSTALL.md），必须有 *.model3.json
```

**没有 `channel.py`** → 驱动从模型的 `cdi3.json` / `model3.json` 自动映射一份示范命令面
（`src/avatar_node/mapper.py`）。自动映射只是兜底示范，不是主路径——真想给一个形象设计
好用的命令面，写 `channel.py`。

## channel.py 的入口

驱动发现套件后，若存在 `channel.py`，会调用它唯一要求的入口：

```python
# avatars/<name>/channel.py
from ghoshell_moss.core.blueprint.channel_builder import new_channel

async def build(avatar):            # avatar: avatar_node.Avatar
    chan = new_channel(name="hiyori", description="...")

    @chan.build.command()
    async def look(x: float = 0.0, y: float = 0.0) -> None:
        """让眼睛转向 (x, y)，取值 -1 到 1"""
        avatar.params({"ParamEyeBallX": x, "ParamEyeBallY": y})

    return chan
```

`build` 返回的 channel 就是这个形象的膜，直接挂进 Matrix。命令签名即接口，不用再写
清单（channel_builder 的红线：不在 instruction 里复述命令）。

## Avatar 事件面（形象作者只用这一个对象）

驱动把"我要它做什么"事件化封装在 `Avatar` 上，作者不需要碰 WS、页面、渲染：

| 方法 | 语义 |
|---|---|
| `avatar.param(id, value, manual=False)` | 推一个参数到某值。`manual=True` 表示模型显式命令，若目标是唇形参数会关掉自动唇动（模型输出优先） |
| `avatar.params({id: value, ...})` | 批量推参数 |
| `avatar.motion(group, index=0, loop=False)` | 播一个动作组里第 index 个 |
| `avatar.expression(name)` | 切一个表情 |
| `avatar.reset()` | 表情/参数回默认，并恢复自动唇动 |
| `avatar.set_backdrop(url)` | 换背板（通用，与模型包解耦） |
| `avatar.set_idle(group, index=0)` | 设定待机动作（空闲循环播）；传 None 回自动 |
| `avatar.set_lip_sync(on)` | 开关自动唇动 |
| `avatar.zoom(factor)` / `avatar.move(x, y)` | 缩放 / 平移（视口层，非模型参数） |
| `avatar.spec` | `ModelSpec`：`.groups`、`.motions`、`.expressions`、`.lip_sync`、`.eye_blink` |

约束（见 NODE.md）：**command 即真相**——驱动不向页面回读，`avatar.state()` 返回的是
被命令过的参数。

## 默认待机、唇动与交互感知

这三件是框架的默认行为，作者在 `build()` 里可覆盖：

- **默认待机**：`build()` 里调 `avatar.set_idle("Idle", 0)` 即把该动作设为这个形象的
  默认 idle。不设则自动（优先名字含 Idle 的动作组，否则第一个）。
- **自动唇动**：驱动默认订阅 `audio/sample`（说侧）驱动唇形。模型显式对唇形参数下
  命令（`avatar.param(..., manual=True)`）即关闭自动唇动，`set_lip_sync(True)` 恢复。
- **交互感知面**：人类点击/拖拽形象会被记录进 `avatar.interactions()`（有界 buffer），
  notice 自动 tail 给模型。点击还会触发 Tap 反馈动作（模型有 `Tap*` 动作组才播）。
  其余状态不进模型 context，按需用命令查。

## 命名与发现

- 套件名 = 目录名。启动用 `--avatar <name>` 选定，换形象 = 重启 node。
- 参数命令名由 `lexicon.param_ident` 从参数 id（必要时从 cdi3 的 Name）归一，共享词表
  见 `src/avatar_node/lexicon.py`——它描述 Cubism 参数词汇本身，不是 per-model 配置。
