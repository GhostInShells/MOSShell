# Web Trafilatura App

Web content extraction app — fetch any URL and return clean Markdown via trafilatura.

## 依赖

- `trafilatura` — local HTML→Markdown extraction, no external API
- `ghoshell-moss[host]` — Matrix host

## Channel

`web_trafilatura` — CTML 调用路径 `apps.web_trafilatura:<command>`

### Commands

| Command | Signature | Behavior |
|---------|-----------|----------|
| `extract` | `(url: str, output_format: str = "markdown") -> str` | Fetch URL, extract readable content |
| `extract_batch` | `(urls: list[str], output_format: str = "markdown") -> dict[str, str]` | Concurrent multi-URL extract |
All commands use `always_observe=True` — results flow into the observe stream.

### CTML 示例

```
<apps.web_trafilatura:extract url="https://github.com/adbar/trafilatura" />
<apps.web_trafilatura:extract_batch urls=["https://example.com", "https://python.org"] />
```
