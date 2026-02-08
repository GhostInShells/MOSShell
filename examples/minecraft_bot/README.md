# Minecraft Bot 示例

## 项目概述

minecraft_bot 示例提供了一个可对话、可交互的 Minecraft 机器人，用于展示 MOSS 架构在游戏 NPC 应用的前景。

机器人基于开源项目 [mineflayer](https://github.com/PrismarineJS/mineflayer) 构建，这是一个用于控制 Minecraft 机器人的 Node.js 库。通过 Python 的 javascript 库进行桥接，实现了在 Python 中调用 JavaScript 库的功能。

## 功能特性

当前已集成的机器人行为能力包括：
- 🚶‍♂️ **移动控制** - 在游戏世界中自由移动
- 👥 **跟随玩家** - 自动跟随指定玩家
- 🔍 **物品查找** - 寻找游戏中的特定物品
- ⛏️ **资源采集** - 自动采集指定物品
- 💬 **智能对话** - 通过 MOSS 架构实现自然语言交互

## 项目结构

```
minecraft_bot/
├── main.py              # 主程序文件
├── package.json         # Node.js 依赖配置
├── package-lock.json    # 依赖锁定文件
├── server/              # Minecraft 服务器配置
│   └── docker-compose.yml
└── README.md           # 项目说明文档
```

# 环境准备
## Minecraft 客户端安装

### 下载 HMCL 启动器
访问 [HMCL 官网](https://hmcl.huangyuhui.net/download/) 下载启动器。

### 安装 Minecraft 1.21.8 版本
1. 启动 HMCL 启动器
2. 点击"安装游戏"
3. 选择 "Minecraft 1.21.8" 版本
4. 点击"安装"完成客户端安装

## 安装 Minecraft Server

### 使用 Docker 启动服务器
```bash
cd server
docker-compose up -d
```
第一次启动容器时会自动下载 Minecraft 1.21.8 版本的服务器文件。

### 检查服务状态
等待服务启动完成，检查端口是否正常监听：
```bash
docker-compose logs -f 
```

出现以下日志表示服务启动成功：
```
[03:30:22] [Server thread/INFO]: Done (5.274s)! For help, type "help"
[03:30:22] [Server thread/INFO]: Starting remote control listener
[03:30:22] [Server thread/INFO]: Thread RCON Listener started
[03:30:22] [Server thread/INFO]: RCON running on 0.0.0.0:25575
```

### 配置服务器
服务器启动后，在 `server/data` 目录下会生成配置文件：
- `server.properties` - 服务器主配置文件
- `whitelist.json` - 白名单配置
- ...

**重要配置修改：**
编辑 `server.properties`，将 `online-mode` 设置为 `false`：
```properties
online-mode=false
```

编辑 `server.properties`，将 `gamemode` 设置为 `creative`（防止被小怪攻击）：
```properties
gamemode=creative
```

修改后重启容器使配置生效：
```bash
docker-compose restart
```

**注意：** 如果不修改此配置，将无法加入本地服务器。

# 使用指南

## 加入服务器

1. **启动 Minecraft 游戏**
2. **选择"多人游戏"**
3. **添加服务器**，地址设置为 `127.0.0.1`
4. **加入服务器**

## 启动机器人

配置环境变量
在项目根目录下创建 `.workspace/.env` 文件，配置环境变量：

```bash
cp .workspace/.env.example .workspace/.env
```

编辑 `.workspace/.env` 文件，填写您的 模型 API 密钥等信息：

```bash
MOSS_LLM_API_KEY=your_api_key_here
MOSS_LLM_BASE_URL=your_llm_base_url_here
MOSS_LLM_MODEL=your_llm_model_here

VOLCENGINE_STREAM_TTS_APP=your_volcengine_stream_tts_app_id_here
VOLCENGINE_STREAM_TTS_ACCESS_TOKEN=your_volcengine_stream_tts_access_token_here
```

在项目根目录下运行以下命令启动机器人：
```bash
python main.py [--speech]
```
第一次执行会自动安装 Node.js 依赖

机器人将自动连接到服务器并出现在游戏中。默认机器人用户名为 `Jarvis`。
添加`--speech`会开启语音，需要运行前配置.workspace/.env文件中的tts相关变量

## 交互方式

机器人启动后，您可以通过游戏内的聊天功能（按t）与机器人交互：
- 发送消息给机器人
- 使用指令控制机器人行为
- 机器人会根据 MOSS 架构进行智能响应

# 配置说明

## 修改机器人设置

在 `main.py` 中可以修改以下配置：

```python
# 服务器配置
BOT_USERNAME = 'Jarvis'  # 机器人用户名
HOST = '127.0.0.1'       # 服务器地址
PORT = 25565             # 服务器端口
```

## 自定义行为

可以通过修改 `main.py` 中的事件处理函数来自定义机器人行为，例如添加新的移动模式或交互逻辑。

# 故障排除

## 常见问题

1. **无法连接服务器**
   - 检查 Docker 容器是否正常运行：`docker-compose ps`
   - 确认服务器端口 25565 是否被占用
   - 验证 `online-mode=false` 配置是否正确

2. **机器人无法启动**
   - 检查 Python 和 Node.js 依赖是否安装完整
   - 确认 Minecraft 服务器已启动并运行正常
   - 查看控制台错误日志进行排查

3. **机器人无响应**
   - 检查网络连接状态
   - 确认机器人是否成功连接到服务器
   - 验证游戏内聊天功能是否正常

# 技术架构

本项目基于以下技术栈：
- **MOSS 架构** - 提供智能对话和决策能力
- **Mineflayer** - Minecraft 机器人控制库
- **Python-JavaScript 桥接** - 实现跨语言调用
- **Docker** - 容器化部署 Minecraft 服务器
