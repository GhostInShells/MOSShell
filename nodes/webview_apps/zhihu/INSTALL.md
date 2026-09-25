# Install

This node depends on the official `zhihu-cli` binary (the data chassis). The
skill's install mechanism is vendored under `skill/` (only `manifest.json` +
`scripts/setup.sh` + `scripts/run.sh` — the install chassis, not the skill docs).

## 1. Install the CLI binary

```bash
bash nodes/webview_apps/zhihu/skill/scripts/run.sh setup
```

This downloads the official darwin/linux binary for your platform, verifies
host + size + SHA-256 + archive structure + self-reported version, and installs
to the user data dir (`~/Library/Application Support/zhihu-cli/` on macOS,
`$XDG_DATA_HOME/zhihu-cli/` on Linux). No sudo, no PATH change.

Check it:

```bash
bash nodes/webview_apps/zhihu/skill/scripts/run.sh status
```

## 2. Authorize (Access Secret)

The node's web surface has an auth card. Or pre-authorize from the CLI:

```bash
printf '%s' '<your-access-secret>' | \
  "$HOME/Library/Application Support/zhihu-cli/current/zhihu-cli" auth set --secret-stdin
```

Get the Access Secret at https://developer.zhihu.com/profile. It goes into the
macOS Keychain (or Secret Service on Linux) — not into any file.

## 3. Mark installed

```bash
moss nodes install nodes/webview_apps/zhihu
```
