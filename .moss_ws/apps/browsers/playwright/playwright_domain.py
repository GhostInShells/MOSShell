"""Playwright domain module — initializes browser and injects page/context.

Compiled by ModuleEval's eval_server at child process startup.
All top-level names (playwright, browser, context, page, json, urllib)
are injected into the sandbox — AI code sees them directly.
"""

import json
import urllib.parse

from playwright.sync_api import sync_playwright

playwright = sync_playwright().start()
browser = playwright.chromium.launch(headless=False)
context = browser.new_context()
page = context.new_page()
