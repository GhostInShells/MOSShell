"""Playwright domain — materializes a live browser at child startup.

Top-level names (playwright, browser, context, page, json, urllib) are the
runtime's pre-loaded objects; the model drives them via exec/aexec.
"""

import json
import urllib.parse

from playwright.sync_api import sync_playwright

playwright = sync_playwright().start()
# headless — the browser is a standard protocol; MOSS surfaces it through its own
# GUI (screen node), not a native window.
browser = playwright.chromium.launch(headless=True)
context = browser.new_context()
page = context.new_page()
