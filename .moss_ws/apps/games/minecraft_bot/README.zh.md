# Minecraft Bot App

一个 MOSS App，通过 [mineflayer](https://github.com/PrismarineJS/mineflayer) 将 Minecraft 机器人桥接到 MOSS 运行时，作为一个标准的 Channel App。

机器人连接到 Minecraft 服务器后，将移动、挖掘、方块探测和聊天等行为暴露为 Ghost 可调用的 CTML 命令。

## 外部依赖

### 1. Minecraft 服务器（Docker）

启动 App 之前，必须先运行一个 Minecraft 服务器。

```bash
cd server
docker compose up -d
```

首次启动会自动下载 Minecraft 服务端文件。

**检查服务器状态：**

```bash
docker compose logs -f
```

等待出现 `Done!` 日志：

```
[Server thread/INFO]: Done (5.274s)! For help, type "help"
```

### 2. Minecraft 客户端（可选）

如果你想在游戏中与机器人互动，需要安装 Minecraft 客户端。

**下载 HMCL 启动器**

访问 [HMCL 官网](https://hmcl.huangyuhui.net/download/) 下载启动器。

**安装 Minecraft 1.21.8**

1. 启动 HMCL 启动器
2. 点击"安装游戏"
3. 选择 "Minecraft 1.21.8"
4. 点击"安装"

**加入服务器**

1. 启动 Minecraft
2. 选择"多人游戏"
3. 添加服务器，地址填写 `127.0.0.1`
4. 加入服务器

## 配置

编辑 `configs/minecraft_bot.yml`：

```yaml
host: "127.0.0.1"
port: 25565
bot_name: "Jarvis"
```

- `host` / `port`：Minecraft 服务器地址
- `bot_name`：机器人在游戏内的用户名（必须在服务器上唯一）

## CTML 命令

App 运行后，Ghost 可以通过 CTML 控制机器人：

```xml
<!-- 移动到指定坐标 -->
<apps.games_minecraft_bot:move x="10" y="64" z="20" />

<!-- 前往玩家位置 -->
<apps.games_minecraft_bot:come sender="Steve" />

<!-- 在游戏聊天栏回复 -->
<apps.games_minecraft_bot:reply>Hello from MOSS!</apps.games_minecraft_bot:reply>

<!-- 获取当前坐标 -->
<apps.games_minecraft_bot:where_i_am />

<!-- 查找附近方块 -->
<apps.games_minecraft_bot:find_blocks block_name="diamond_ore" max_distance="64" count="5" />

<!-- 挖掘脚下方块 -->
<apps.games_minecraft_bot:dig_under />

<!-- 跟随玩家 -->
<apps.games_minecraft_bot:set_follow_player sender="Steve" />
<apps.games_minecraft_bot:stop_follow_player />
```

**重要**：所有对玩家的回复必须通过 `reply` 命令发送。Ghost 的默认输出不会自动流入 Minecraft 聊天栏。

## 架构

- **输入**：Minecraft 聊天消息被转换为 `Signal(name="input")`，上报给 Mindflow。
- **输出**：Ghost 必须显式调用 `apps.games_minecraft_bot:reply` 才能将消息发送到 Minecraft 聊天栏。
- **Channel 名称**：`minecraft`（完整 CTML 前缀：`apps.games_minecraft_bot`）

## 故障排除

| 问题 | 原因 | 解决方法 |
|---------|-------|-----|
| 无法连接服务器 | Docker 容器未运行 | `cd server && docker compose up -d` |
| 连接超时（30s） | 服务器未就绪或地址/端口错误 | 检查 `configs/minecraft_bot.yml` 和服务器日志 |
| 被服务器踢出 | `online-mode=true` | 在 `server/data/server.properties` 中设置 `online-mode=false` 并重启 |
| 机器人名称已被占用 | 其他玩家/机器人使用了相同名字 | 修改配置中的 `bot_name` |
| 命令返回"未连接到服务器" | 机器人与服务器断开连接 | 检查服务器状态并重启 App |

## 项目结构

```
.
├── main.py              # App 入口、Channel 定义、mineflayer 桥接
├── APP.md               # MOSS App 元数据
├── pyproject.toml       # Python 依赖（javascript, ghoshell-moss）
├── configs/             # App 配置
│   └── minecraft_bot.yml
├── server/              # Minecraft 服务器（Docker）
│   └── docker-compose.yml
└── tests/               # App 测试
```
