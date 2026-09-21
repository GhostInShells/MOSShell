"""Co-browser playwright domain — headed by construction.

The human is watching; there is no silent variant. Top-level names
(playwright, browser, context, page, json, urllib) are the runtime's
pre-loaded objects; the model drives them via exec/aexec.
"""

import json
import urllib.parse

from playwright.sync_api import sync_playwright

playwright = sync_playwright().start()
browser = playwright.chromium.launch(headless=False)
context = browser.new_context()
page = context.new_page()
