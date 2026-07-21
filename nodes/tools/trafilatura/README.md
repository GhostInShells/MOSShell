# trafilatura

Web content extraction — fetch any URL and get clean Markdown text.

## Setup

```bash
cd nodes/tools/trafilatura
uv sync
moss nodes install nodes/tools/trafilatura
```

## Usage

```bash
moss nodes run nodes/tools/trafilatura
```

## Dependencies

- `trafilatura` — local HTML→Markdown extraction
- `ghoshell-moss[host]` — Matrix host
