---
name: 'webview_consumer'
description: 'webview service kind consume-side probe — prints changes, activates the first unread view'
singleton: true
exec:
  command: python
  args: main.py
---

Consume-side probe for the `webview` service kind. Connects a `WebViewClient`,
prints a one-line summary on every change, and — once any view carries an unread
count — activates it and prints the settled condition. Proves discovery, the
change stream, ordering and focus convergence across two processes.

    moss nodes run .moss/system_test_nodes/webview_consumer --mode system_test
