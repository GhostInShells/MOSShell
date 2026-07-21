# Minecraft Bot App

[中文文档](README.zh.md)

A MOSS App that bridges a Minecraft bot (via [mineflayer](https://github.com/PrismarineJS/mineflayer)) into the MOSS runtime as a standard Channel App.

The bot connects to a Minecraft server and exposes movement, mining, block discovery, and chat as CTML commands for a Ghost.

## External Dependencies

### 1. Minecraft Server (Docker)

A Minecraft server must be running before the App starts.

```bash
cd server
docker compose up -d
```

First start will download the Minecraft server files automatically.

**Check server status:**

```bash
docker compose logs -f
```

Wait for the `Done!` message:

```
[Server thread/INFO]: Done (5.274s)! For help, type "help"
```

### 2. Minecraft Client (optional)

To interact with the bot in-game, you need a Minecraft client.

**Download HMCL Launcher**

Visit [HMCL official site](https://hmcl.huangyuhui.net/download/) to download the launcher.

**Install Minecraft 1.21.8**

1. Open HMCL launcher
2. Click "Install Game"
3. Select "Minecraft 1.21.8"
4. Click "Install"

**Join the server**

1. Launch Minecraft
2. Select "Multiplayer"
3. Add server with address `127.0.0.1`
4. Join the server

## Configuration

Edit `configs/minecraft_bot.yml`:

```yaml
host: "127.0.0.1"
port: 25565
bot_name: "Jarvis"
```

- `host` / `port`: Minecraft server address
- `bot_name`: Bot username in game (must be unique on the server)

## CTML Commands

Once the App is running, a Ghost can control the bot via CTML:

```xml
<!-- Move to coordinates -->
<apps.games_minecraft_bot:move x="10" y="64" z="20" />

<!-- Go to a player -->
<apps.games_minecraft_bot:come sender="Steve" />

<!-- Reply in Minecraft chat -->
<apps.games_minecraft_bot:reply>Hello from MOSS!</apps.games_minecraft_bot:reply>

<!-- Get current position -->
<apps.games_minecraft_bot:where_i_am />

<!-- Find nearby blocks -->
<apps.games_minecraft_bot:find_blocks block_name="diamond_ore" max_distance="64" count="5" />

<!-- Dig under feet -->
<apps.games_minecraft_bot:dig_under />

<!-- Follow a player -->
<apps.games_minecraft_bot:set_follow_player sender="Steve" />
<apps.games_minecraft_bot:stop_follow_player />
```

**Important**: All replies to players must go through the `reply` command. Ghost output does not automatically flow into Minecraft chat.

## Architecture

- **Input**: Minecraft chat messages are converted to `Signal(name="input")` and sent to Mindflow.
- **Output**: The Ghost must explicitly call `apps.games_minecraft_bot:reply` to send messages to Minecraft chat.
- **Channel name**: `minecraft` (full CTML prefix: `apps.games_minecraft_bot`)

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Cannot connect to server | Docker container not running | `cd server && docker compose up -d` |
| Connection timeout (30s) | Server not ready or wrong host/port | Check `configs/minecraft_bot.yml` and server logs |
| Kicked by server | `online-mode=true` | Set `online-mode=false` in `server/data/server.properties` and restart |
| Bot name already taken | Another player/bot using the same name | Change `bot_name` in config |
| Commands return "not connected" | Bot disconnected from server | Check server status and restart the App |

## Project Structure

```
.
├── main.py              # App entry, channel definition, mineflayer bridge
├── APP.md               # MOSS App metadata
├── pyproject.toml       # Python dependencies (javascript, ghoshell-moss)
├── configs/             # App configuration
│   └── minecraft_bot.yml
├── server/              # Minecraft server (Docker)
│   └── docker-compose.yml
└── tests/               # App tests
```
