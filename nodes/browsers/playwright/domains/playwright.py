"""Playwright domain — materializes a live browser at child startup.

Top-level names (playwright, browser, context, page, json, urllib) are the
runtime's pre-loaded objects; the model drives them via exec/aexec.
"""

import json
import urllib.parse

from playwright.sync_api import sync_playwright

playwright = sync_playwright().start()
# Headed (temporary): human-visible dogfooding of the screen-manager lab.
# MEMO (2026-09-19): headed must be a FORCED, confirmed mechanism — the human
# must know a visible window opens; never silent. Default headless; gate headed
# behind a node arg / env var + confirmation step, not a code default.
browser = playwright.chromium.launch(headless=False)
context = browser.new_context()
page = context.new_page()
