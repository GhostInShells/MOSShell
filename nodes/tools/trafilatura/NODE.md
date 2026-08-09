---
name: 'trafilatura'
description: 'Local web content extraction via trafilatura — fetch URL and extract clean Markdown, no external API required.'
category: tools
singleton: true
exec:
  command: python
  args: main.py
---

Use when you need to read the content of a web page. Prefer this over
playwright when you only need the text — it's faster and has no browser
overhead. For pages that require JavaScript rendering or interaction,
use the playwright node instead.
