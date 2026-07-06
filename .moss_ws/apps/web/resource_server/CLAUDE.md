# Resource HTTP Endpoint

FastAPI-based HTTP server that exposes Matrix ResourceRegistry resources to browsers.

## Purpose

Matrix resources use `scheme://host/path` locators. This app maps them to HTTP URLs
so humans can access them in a browser — play videos, view images, download files.

## Architecture

- **FastAPI** — HTTP framework with automatic FileResponse Range support
- **No Channel** — this app serves browsers, not models
- **ConfigStore** — configuration via `ResourceServerConfig` ConfigType, stored in cell workspace

## Routes

| Method | Path | Behavior |
|--------|------|----------|
| GET | /health | Health check |
| GET | /resources | List all schemes and their hosts |
| GET | /resources/{scheme} | List hosts for scheme |
| GET | /resources/{scheme}/{host} | List resources for host |
| GET | /resources/{scheme}/{host}/{path} | Serve resource data |

## Configuration

`ResourceServerConfig` (ConfigType), persisted to cell workspace `configs/resource_server.yml`:
- `host` — bind address (default: 127.0.0.1)
- `port` — bind port (default: 20880)

## Testing

```bash
# Start the app
moss apps test web/resource_server

# Health check
curl http://127.0.0.1:20880/health

# List all schemes
curl http://127.0.0.1:20880/resources

# Access a resource
curl http://127.0.0.1:20880/resources/{scheme}/{host}/{path}
```

## Dependencies

- `fastapi[standard]` — HTTP framework with FileResponse Range support
- `uvicorn` — ASGI server
