"""The node's own audit page — the human's face on everything the ghost did.

A tiny zero-dependency HTTP server (stdlib ``http.server``, daemon thread) that
serves one self-contained page and a JSON log endpoint the page polls. It is the
generalization of co_browser's surface: every dispatch, every authorization flip,
every human push lands here, so "I can see" replaces "I approve".

The page doubles as the first-run install guide: opening it shows the human how to
load the extension, so the "how do I use this body" answer is on the page, not
buried in the node's stdout.

What appears here is **the ghost's behavior and the human's authorizations** — not
the human's own browsing. That line is the privacy rule of this body; the audit
page must never become a browsing history. Served on 127.0.0.1 only.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import PageModel

__all__ = ["AuditServer"]


def _page(extension_dir: str) -> str:
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>ghost-in-web · audit</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0b0e12; color:#d7dee8;
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }}
  header {{ padding:14px 18px; border-bottom:1px solid #1d242e; display:flex;
           align-items:baseline; gap:12px; }}
  header h1 {{ font-size:14px; margin:0; font-weight:600; letter-spacing:.04em; }}
  header .hint {{ color:#66707d; font-size:11px; }}
  #install {{ padding:12px 18px; border-bottom:1px solid #1d242e; background:#0d1418;
             color:#9fb2c8; font-size:12px; line-height:1.7; }}
  #install b {{ color:#2dd4bf; font-weight:600; }}
  #install code {{ background:#141d26; padding:1px 5px; border-radius:3px; color:#d7dee8; }}
  #rows {{ padding:6px 0; }}
  .row {{ display:grid; grid-template-columns:64px 78px 56px 1fr; gap:10px;
         padding:5px 18px; border-bottom:1px solid #12171e; }}
  .row:hover {{ background:#10151c; }}
  .t {{ color:#5a6673; }}
  .pg {{ color:#7f8b99; }}
  .kind {{ color:#8b97a6; }}
  .k-behavior {{ color:#6fb1ff; }}
  .k-event {{ color:#c792ea; }}
  .k-perception {{ color:#3ddc8f; }}
  .k-dialog {{ color:#e5c07b; }}
  .ok {{ color:#3ddc8f; }} .no {{ color:#e06c75; }}
  .pending {{ color:#66707d; }}
  .empty {{ padding:40px 18px; color:#4c555f; }}
  .shot img {{ max-width:360px; max-height:200px; display:block; margin-top:6px;
              border-radius:4px; border:1px solid #2a3340; }}
</style></head>
<body>
<header>
  <h1>ghost-in-web · audit</h1>
  <span class="hint">模型对页面做过的每一次动作 · 人类的每一次授权 · 只记录 ghost 的行为,不记录你自己的浏览</span>
</header>
<div id="install">
  <b>装扩展</b> → <code>chrome://extensions</code> 打开开发者模式 → 加载已解压的扩展程序 →
  选目录 <code>{extension_dir}</code><br>
  装好后把扩展 id pin 进 <code>MOSS_GHOST_IN_WEB_ORIGINS</code> (<code>chrome-extension://&lt;id&gt;</code>);
  然后打开任意非 local 网页,点右上角图标授权感知。
</div>
<div id="rows"><div class="empty">等待中…</div></div>
<script>
let lastKey = '';
async function tick() {{
  let data;
  try {{ data = await (await fetch('/log')).json(); }} catch (e) {{ return; }}
  // 没变化就别重渲染,否则截图每秒被重绘会闪。
  const key = JSON.stringify(data.map(e => [e.detail, e.ok, e.image ? e.image.length : 0]));
  if (key === lastKey) return;
  lastKey = key;
  const rows = document.getElementById('rows');
  if (!data.length) {{ rows.innerHTML = '<div class="empty">还没有动作</div>'; return; }}
  rows.innerHTML = data.reverse().map(e => {{
    const ok = e.ok === null ? '<span class="pending">⋯</span>'
             : (e.ok ? '<span class="ok">✓</span>' : '<span class="no">✗</span>');
    const t = new Date(e.created * 1000).toTimeString().slice(0, 8);
    const img = e.image ? `<div class="shot"><img src="${{e.image}}"></div>` : '';
    return `<div class="row"><span class="t">${{t}}</span>` +
           `<span class="pg">${{e.label}}</span>` +
           `<span class="kind k-${{e.kind}}">${{e.kind}}</span>` +
           `<span>${{ok}} ${{escapeHtml(e.detail)}}${{img}}</span></div>`;
  }}).join('');
}}
function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => (
    {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
tick(); setInterval(tick, 1000);
</script>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    model: PageModel  # set on the subclass below
    extension_dir: str

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.startswith("/log"):
            body = json.dumps(
                [
                    {
                        "label": e.label,
                        "kind": e.kind,
                        "detail": e.detail,
                        "ok": e.ok,
                        "created": e.created,
                        "image": f"data:image/jpeg;base64,{e.image_b64}" if e.image_b64 else None,
                    }
                    for e in self.model.audit()
                ],
                ensure_ascii=False,
            ).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        else:
            body = _page(self.extension_dir).encode("utf-8")
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass  # keep the node's stderr clean


class AuditServer:
    def __init__(
        self,
        model: PageModel,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        extension_dir: str = "nodes/browsers/ghost-in-web/extension",
    ) -> None:
        handler = type(
            "_BoundHandler", (_Handler,), {"model": model, "extension_dir": extension_dir}
        )
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        self.host = host
        self.port = self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
