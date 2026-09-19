---
name: 'webview_server'
description: 'webview service kind serve-side probe — declares a url and drives notify/touch on a timer'
singleton: true
exec:
  command: python
  args: main.py
---

Serve-side probe for the `webview` service kind. Provides a `WebViewServer` with
a synthetic url and, every couple of seconds, exercises the kind's two
self-claims — `notify` (unread + activity) and `touch` (activity only) — so a
consumer can observe discovery, the change stream and ordering with no model
involved.

    moss nodes run .moss/system_test_nodes/webview_server --mode system_test
