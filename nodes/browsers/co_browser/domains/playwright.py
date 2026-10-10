"""Co-browser playwright domain — headed by construction.

The human is watching; there is no silent variant. Top-level names
(playwright, browser, context, page, json, urllib) are the runtime's
pre-loaded objects; the model drives them via exec/aexec.
"""

import json
import urllib.parse

from playwright.sync_api import sync_playwright

playwright = sync_playwright().start()
# Maximized, and with **no pinned viewport**: the content follows the window.
#
# Playwright's default is a 1280x720 viewport that ignores the window size, and
# that default quietly fights the human — resizing or splitting the window leaves
# the page at 1280x720, which reads as "the thing is just small". The human owns
# the window here; the model pins a viewport explicitly (`page.set_viewport_size`)
# only when it needs a deterministic frame for a measurement.
browser = playwright.chromium.launch(headless=False, args=["--start-maximized"])
context = browser.new_context(no_viewport=True)
page = context.new_page()
