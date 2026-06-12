# Minecraft Bot App — AI Developer Context

You are working on a **Minecraft Bot MOSS App**. It bridges a Minecraft bot
(mineflayer + Python-JS bridge) into MOSS as a standard Channel App.

## What This App Is

A Channel-based MOSS App that lets a Ghost control a Minecraft bot. The bot
connects to a Minecraft server, senses the environment, moves, mines, chats,
and follows players — all driven by CTML commands from a Ghost.

## Mineflayer 基础

Mineflayer is a **Node.js** library for creating Minecraft bots. In this app we
call it through the `javascript` Python package (a Python-JS bridge).

- **mineflayer**: https://github.com/PrismarineJS/mineflayer
- **mineflayer-pathfinder**: https://github.com/PrismarineJS/mineflayer-pathfinder
- **prismarine-viewer** (for visual perception): https://github.com/PrismarineJS/prismarine-viewer

### Core requires

```python
mineflayer = require("mineflayer")
pathfinder = require("mineflayer-pathfinder")
blockfinder = require("mineflayer-blockfinder")
```

### Common APIs

| API | What it does |
|-----|-------------|
| `mineflayer.createBot({"host": ..., "port": ..., "username": ...})` | Create and connect bot |
| `bot.loadPlugin(plugin)` | Load a mineflayer plugin |
| `bot.entity.position` | Bot's current `Vec3` position |
| `bot.players[name]` | Get a player object by name |
| `bot.blockAt(position)` | Get block at a `Vec3` position |
| `bot.findBlocks({"matching": ids, "maxDistance": d, "count": n})` | Find blocks by ID |
| `bot.dig(block)` | Mine a block |
| `bot.chat(text)` | Send a chat message |
| `bot.canDigBlock(block)` | Check if block is diggable |
| `bot.lookAt(position)` | Look at a `Vec3` position |
| `bot.pathfinder.setGoal(goal)` | Set pathfinding goal |
| `pathfinder.goals.GoalNear(x, y, z, range)` | Goal: stand near a point |
| `pathfinder.Movements(bot)` | Movement constraints for pathfinder |
| `bot.registry.blocksByName[name]` | Block registry lookup |

### JS events (wired via `@On`)

```python
@On(bot, "spawn")
def _on_spawn(): ...

@On(bot, "chat")
def _on_chat(this, sender, message, *args): ...

@On(bot, "error")
def _on_error(this, err): ...

@On(bot, "end")
def _on_end(*args): ...

@On(bot, "login")
def _on_login(*args): ...
```

**Critical**: JS callbacks run on a different thread. Never call asyncio or
`matrix.session.add_signal()` directly inside a JS callback. Use
`asyncio.Queue` or `asyncio.Event` as a thread bridge, then consume from
the Python asyncio main loop.

## Current Architecture

### Channel

- Name: `minecraft` (CTML prefix: `<apps.games_minecraft_bot:...>`)
- Commands: `move`, `come`, `where_i_am`, `where_player_is`, `find_blocks`,
  `dig_under`, `dig_target`, `set_follow_player`, `stop_follow_player`, `reply`
- Idle hook: continuous player following via `pathfinder`
- Context messages: connection status, position, nearby blocks, follow target

### State

All bot state is module-level (single App instance = single bot):

```python
_bot           # mineflayer bot instance
_bot_ready     # bool: bot spawned and pathfinder ready
_to_follow_player  # str: player name to follow
_conn_event    # asyncio.Event: signals login / error / end
_conn_error    # str | None: connection failure reason
_chat_queue    # asyncio.Queue[Message]: JS chat → Python bridge
```

### Config

`MinecraftBotConfig(ConfigType)` — loaded via `matrix.cell_workspace.configs().read_yaml()`.
File lives at `configs/minecraft_bot.yml`.

```yaml
host: "127.0.0.1"
port: 25565
bot_name: "Jarvis"
```

### Output channel

Ghost replies to players **must** go through the `reply` command. The channel
description enforces this routing. `reply` calls `bot.chat(text)`.

### Input channel

Minecraft chat messages become `Signal(name="input", priority=Priority.NOTICE)`
via `_chat_bridge_task()`. The JS `@On(bot, "chat")` callback pushes to
`_chat_queue`; the Python task consumes and calls `matrix.session.add_signal()`.

## How to Add a New Command

1. Define the command on `chan` in `main.py`:

```python
@chan.build.command()
async def my_command(arg: str) -> str:
    """Description for the Ghost.

    Args:
        arg: Description of arg.
    """
    if not _bot_ready or _bot is None:
        return "未连接到服务器。"
    try:
        # ... use mineflayer API ...
        return "结果"
    except Exception as e:
        return f"失败：{e}"
```

2. If the command needs new state, add a module-level variable.
3. If the command emits output to Minecraft chat, consider whether it should
   go through `reply` or return a string.

## Future Directions

This app is **open-ended**. There is no fixed roadmap — any capability mineflayer
(or its plugin ecosystem) supports can be exposed as a command, context message,
or idle hook. Some natural extensions:

- **Inventory management**: list items, craft, equip armor, drop items
- **Combat**: attack entities, dodge, use shields/bows
- **Building**: place blocks from inventory, build structures
- **Farming**: plant/harvest crops, breed animals
- **Exploration**: long-distance navigation, map recording, waypoint system
- **Entity interaction**: trade with villagers, ride horses/minecarts
- **Redstone**: interact with levers, buttons, pressure plates
- **Multi-server**: connect to different servers via config profiles
- **Plugin integration**: `prismarine-viewer`, `mineflayer-auto-auth`, etc.

**You are not limited to the above.** If mineflayer can do it, this app can
expose it. The only constraint is the thread-safety bridge between JS callbacks
and Python asyncio.

## Known Improvement: Visual Perception

The current `context_messages()` reports the bot's surroundings as a **text list
of nearby blocks** (a 5x5x4 volume around the bot). This works but is low-bandwidth
and hard for the Ghost to interpret spatially.

**The desired upgrade**: capture the bot's **visual view** as an image and send
it as an image message in `context_messages()`.

Mineflayer plugins that can help:
- `mineflayer-screenshot` — render the bot's first-person view to a PNG
- `prismarine-viewer` — headless or viewer-based rendering
- Custom: use mineflayer's internal block/mesh data to generate a top-down
  or isometric view

When implementing visual perception, wire it similarly to the existing context
message system: generate the image in `context_messages()`, wrap it in a
`Message` with an image content type, and return it alongside (or instead of)
the block list.

## Testing

```bash
# Start the local Minecraft server
cd server && docker compose up -d

# Test the app (foreground, logs to console)
moss apps test games/minecraft_bot

# Run unit tests
uv run pytest tests/ -v
```

## Common Pitfalls

- **JS thread bridge**: Never call `matrix.session.add_signal()` inside a JS
  `@On` callback. Always use `asyncio.Queue` / `asyncio.Event`.
- **Connection state**: Check `_bot_ready` before every command. Commands should
  fail gracefully with a Chinese error message when disconnected.
- **Pathfinder goal override**: `bot.pathfinder.setGoal()` replaces the current
  goal. The `on_idle` follow loop and `move`/`come` commands compete — this is
  intentional (idle is low-priority background behavior).
- **Config path**: Use `matrix.cell_workspace.configs().read_yaml(...)` — never
  hardcode `Path(__file__).parent / ...`.

---
*Last updated 2026-06-11.*
