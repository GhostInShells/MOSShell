---
name: 'desktop-gui'
description: 'Desktop GUI — human observation and approval interface for the desktop channel'
singleton: true
exec:
  command: python
  args: main.py
---

Desktop GUI is the human-facing window into Ghost's desktop activity. Ghost
operates desktop.bash / desktop.file_editor as usual — this GUI renders the
command stream, shows status via breathing-light indicators, and lets the
human approve, reject, or dialogue with the Ghost about each action.

Dual-pane layout: sidebar lists commands (active / stale toggle), detail
panel shows command content, approval dialog, and execution results.

This is NOT a Ghost tool. Ghost does not know this GUI exists. It's a human
perception space shared with Ghost through the desktop channel.
