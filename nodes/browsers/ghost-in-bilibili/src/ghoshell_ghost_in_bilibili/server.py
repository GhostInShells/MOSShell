"""Local HTTP server for ghost-in-bilibili.

Endpoints:
  GET  /            instruction page
  GET  /events      recent received events
  GET  /pages       page registry (label -> {bvid,url,title,state})
  GET  /config      { ghostName }
  GET  /js/drain    drain the whole pending-JS queue for a page (frontend polls)
  POST /ingest      receive an event: dump temp file -> read back -> delete -> log
  POST /js/dispatch model puts a JS command for a page (placeholder return)

State lives in the shared ``BridgeModel`` (injected) — this handler only moves bytes.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .model import BridgeModel

_MODEL: BridgeModel | None = None
_HTML_PATH: Path | None = None


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "*"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html(self, path: Path):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        assert _MODEL is not None
        if self.path == "/events":
            return self._json(_MODEL.recent)
        if self.path == "/pages":
            return self._json(_MODEL.pages)
        if self.path == "/config":
            return self._json({"ghostName": _MODEL.ghost_name})
        if self.path.startswith("/js/drain"):
            page = (parse_qs(urlparse(self.path).query).get("page") or [""])[0]
            return self._json(_MODEL.drain_js(page))
        if self.path in ("/", "/index.html"):
            if _HTML_PATH is None:
                return self._json({"error": "no index.html"})
            return self._html(_HTML_PATH)
        return self._json({"error": "not found"})

    def do_POST(self):
        assert _MODEL is not None
        raw = self._read_body()

        if self.path == "/js/dispatch":
            try:
                obj = json.loads(raw)
            except Exception:
                return self._json({"ok": False, "error": "bad json"})
            page, action = obj.get("page"), obj.get("action")
            if not page or not action:
                return self._json({"ok": False, "error": "need page + action"})
            cmd = _MODEL.dispatch_action(page, action, obj.get("value"), obj.get("id"))
            print(f"[ghost] dispatch {action} #{cmd['id']} -> {page}")
            return self._json({"ok": True, "id": cmd["id"], "page": page,
                               "queued": len(_MODEL.pending_js[page])})

        if self.path != "/ingest":
            return self._json({"ok": False, "error": f"unknown path {self.path}"})

        tmp_dir = os.path.join(tempfile.gettempdir(), "moss-bilibili-bridge")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, f"dump-{uuid.uuid4().hex[:8]}.json")

        with open(path, "wb") as f:
            f.write(raw)
        size = os.path.getsize(path)

        with open(path, "rb") as f:
            back = f.read()
        read_back_identical = back == raw

        parsed = None
        try:
            parsed = json.loads(back.decode("utf-8"))
            parsed_as = "json"
        except Exception:
            parsed_as = "raw"

        os.remove(path)

        if parsed is not None:
            _MODEL.add_event(parsed)
            if parsed.get("type") == "toggle":
                _MODEL.apply_toggle(parsed)

        print(
            f"[ghost] ingest {size}B parsed={parsed_as} deleted={not os.path.exists(path)}\n"
            f"  {raw[:600].decode('utf-8', 'replace')}"
        )
        self._json({
            "ok": True, "bytes": size, "path": path,
            "read_back_identical": read_back_identical,
            "parsed_as": parsed_as, "deleted": not os.path.exists(path),
            "echo": raw[:200].decode("utf-8", "replace"),
        })

    def log_message(self, fmt, *args):
        print("[ghost]", fmt % args)


def run_server(host: str, port: int, html_path: Path, model: BridgeModel) -> None:
    global _MODEL, _HTML_PATH
    _MODEL = model
    _HTML_PATH = html_path
    print(f"[ghost] listening on http://{host}:{port}", flush=True)
    print(f"[ghost] instructions: http://{host}:{port}/", flush=True)
    HTTPServer((host, port), Handler).serve_forever()
