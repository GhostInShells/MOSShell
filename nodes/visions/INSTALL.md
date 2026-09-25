# Install — shared visions venv

`nodes/visions/` 是 vision 家族 node 的共享 venv 父目录。所有子 node 共用这一个
venv，无 per-node install（不穿透 — 子 node 不携带自己的 INSTALL/.venv）。

```bash
cd nodes/visions
uv sync
```

当前服务的 node：camera、stream、push。后续 vision node 加入时，在 `pyproject.toml` 追加
对应依赖，无需新建 venv。
