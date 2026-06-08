# MOSS App: im/feishu

Feishu IM integration — WebSocket long connection via lark-channel-sdk.
Push lightweight Signals to Ghost, Ghost pulls messages and replies via Channel.

## Key files

- `APP.md` — App metadata (frontmatter: uv, respawn=true)
- `main.py` — All logic: SDK lifecycle, MessageBuffer, Channel commands, Signal emission
- `pyproject.toml` — Dependencies: lark-channel-sdk + ghoshell-moss[host]
- `doc/FEATURE_in_process.md` — Design decisions, phase plan, implementation steps

## Architecture

```
Feishu Server <──WSS──> lark-channel-sdk (background thread)
                              │
                     _on_message() [SDK thread]
                              │
                     MessageBuffer.put() → dedup + store
                              │
                     loop.call_soon_threadsafe() → main event loop
                              │
                     matrix.session.add_input_signal()
                              │
                     Ghost receives Signal → CTML pull/send
                              │
                     Ghost → im_feishu:pull_messages / send_message / get_status
```

## Commands

| Command | Args | Returns | Notes |
|---------|------|---------|-------|
| `pull_messages` | chat_id, limit=20, before="" | list[dict] | Pull buffered messages |
| `send_message` | chat_id, content, reply_to="" | dict | Send text, optionally inline-reply |
| `get_status` | — | dict | Connection state + bot identity + buffer stats |
| `mark_read` | chat_id, message_id="" | bool | Reset unread count |

## Signal metadata (description field)

Format: `飞书[{sender}][{chat_type_label}]: {text[:50]} | chat_id=xxx | msg_id=xxx | chat_type=xxx | sender_id=xxx [| mentioned_bot=true] [| reply_to=xxx]`

Ghost parses this to extract chat_id/sender_id for routing.

## Environment variables

- `FEISHU_APP_ID` — Feishu app ID (required)
- `FEISHU_APP_SECRET` — Feishu app secret (required)
- `FEISHU_DOMAIN` — "feishu" (default), "lark", or custom URL

## Testing

```bash
# Install dependencies
cd .moss_ws/apps/im/feishu && uv sync

# Verify app discovery
cd /path/to/MOSShell && .venv/bin/moss --ai apps list --mode default

# Test run (foreground)
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
.venv/bin/moss apps test im/feishu

# In Feishu client
# Send a message to the bot → check TUI for Signal → verify Ghost can reply
```
