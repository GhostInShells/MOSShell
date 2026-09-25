#!/bin/sh
set -eu
umask 077

say() { printf '%s\n' "$*" >&2; }
fail_json() {
  message=$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')
  printf '{"ok":false,"error":{"code":"%s","message":"%s"}}\n' "$2" "$message"
  exit 1
}
json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SKILL_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
LOCAL_MANIFEST="$SKILL_DIR/manifest.json"
[ -f "$LOCAL_MANIFEST" ] || fail_json "manifest.json is missing" "INVALID_PACKAGE"

# Skill 只声明最低兼容版本和发布 manifest 地址，不再携带任何平台二进制。
MIN_VERSION=$(sed -n 's/.*"min_version":[[:space:]]*"\([^"]*\)".*/\1/p' "$LOCAL_MANIFEST" | head -n 1)
UPDATE_MANIFEST_URL=$(sed -n 's/.*"update_manifest_url":[[:space:]]*"\([^"]*\)".*/\1/p' "$LOCAL_MANIFEST" | head -n 1)
[ -n "$MIN_VERSION" ] || fail_json "manifest cli.min_version is empty" "INVALID_PACKAGE"
printf '%s' "$MIN_VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$' || fail_json "manifest cli.min_version is invalid" "INVALID_PACKAGE"

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)
case "$OS_NAME" in
  Darwin)
    case "$ARCH_NAME" in
      arm64) PLATFORM=darwin-arm64 ;;
      x86_64) PLATFORM=darwin-amd64 ;;
      *) fail_json "Unsupported macOS architecture: $ARCH_NAME" "UNSUPPORTED_PLATFORM" ;;
    esac
    ;;
  Linux)
    case "$ARCH_NAME" in
      x86_64|amd64) PLATFORM=linux-amd64 ;;
      aarch64|arm64) PLATFORM=linux-arm64 ;;
      *) fail_json "Unsupported Linux architecture: $ARCH_NAME" "UNSUPPORTED_PLATFORM" ;;
    esac
    ;;
  *) fail_json "Unsupported operating system: $OS_NAME" "UNSUPPORTED_PLATFORM" ;;
esac

if [ -n "${ZHIHU_CLI_HOME:-}" ]; then
  CLI_HOME=$ZHIHU_CLI_HOME
elif [ "$OS_NAME" = Darwin ]; then
  [ -n "${HOME:-}" ] || fail_json "HOME is required" "INVALID_CONFIGURATION"
  CLI_HOME="$HOME/Library/Application Support/zhihu-cli"
else
  [ -n "${HOME:-}" ] || fail_json "HOME is required" "INVALID_CONFIGURATION"
  DATA_HOME=${XDG_DATA_HOME:-"$HOME/.local/share"}
  case "$DATA_HOME" in /*) ;; *) fail_json "XDG_DATA_HOME must be an absolute path" "INVALID_CONFIGURATION" ;; esac
  CLI_HOME="$DATA_HOME/zhihu-cli"
fi
if [ "$OS_NAME" = Linux ]; then
  case "$CLI_HOME" in /*) ;; *) fail_json "ZHIHU_CLI_HOME must be an absolute path" "INVALID_CONFIGURATION" ;; esac
fi
CURRENT_DIR="$CLI_HOME/current"
CURRENT="$CURRENT_DIR/zhihu-cli"

read_version() { "$1" version 2>/dev/null | sed -n 's/.*"version":"\([^"]*\)".*/\1/p'; }
version_ge() { awk -v a="$1" -v b="$2" 'BEGIN { split(a,x,/[.-]/); split(b,y,/[.-]/); for(i=1;i<=3;i++){if(x[i]+0>y[i]+0)exit 0;if(x[i]+0<y[i]+0)exit 1} if(a~/-/&&b!~/-/)exit 1; exit 0 }'; }

# 已安装且兼容时直接复用；setup 不负责日常更新，也不会把新版 CLI 降级。
CURRENT_VERSION=""
if [ -x "$CURRENT" ]; then CURRENT_VERSION=$(read_version "$CURRENT"); fi
if [ -n "$CURRENT_VERSION" ] && version_ge "$CURRENT_VERSION" "$MIN_VERSION"; then
  AUTH_CONFIGURED=false
  NEXT_ACTION=request_access_secret
  if AUTH_JSON=$("$CURRENT" auth status 2>/dev/null) && printf '%s' "$AUTH_JSON" | grep -q '"source":"'; then AUTH_CONFIGURED=true; NEXT_ACTION=ready; fi
  ABS_PATH=$(CDPATH= cd -- "$CURRENT_DIR" && pwd)/zhihu-cli
  printf '{"ok":true,"installed":false,"reused_cli":true,"installed_cli_version":"%s","binary_path":"%s","auth_configured":%s,"next_action":"%s"}\n' \
    "$CURRENT_VERSION" "$(json_escape "$ABS_PATH")" "$AUTH_CONFIGURED" "$NEXT_ACTION"
  exit 0
fi

[ -n "$UPDATE_MANIFEST_URL" ] || fail_json "CLI download manifest is not configured" "UPDATE_NOT_CONFIGURED"

# 正式安装只接受 HTTPS。localhost HTTP 仅供仓库内安装测试显式开启。
case "$UPDATE_MANIFEST_URL" in
  https://*) ;;
  http://127.0.0.1:*|http://localhost:*)
    [ "${ZHIHU_CLI_ALLOW_INSECURE_TEST_URL:-}" = "1" ] || fail_json "CLI download manifest must use HTTPS" "UPDATE_INTEGRITY_FAILED"
    ;;
  *) fail_json "CLI download manifest must use HTTPS" "UPDATE_INTEGRITY_FAILED" ;;
esac
MANIFEST_AUTHORITY=${UPDATE_MANIFEST_URL#*://}
MANIFEST_AUTHORITY=${MANIFEST_AUTHORITY%%/*}
case "$MANIFEST_AUTHORITY" in ''|*@*) fail_json "CLI download manifest URL is invalid" "UPDATE_INTEGRITY_FAILED" ;; esac

for required_command in curl tar; do
  command -v "$required_command" >/dev/null 2>&1 || fail_json "$required_command is required to install zhihu-cli" "INSTALL_PREREQUISITE_MISSING"
done
if command -v plutil >/dev/null 2>&1 && [ "$OS_NAME" = Darwin ]; then
  JSON_PARSER=plutil
elif command -v jq >/dev/null 2>&1; then
  JSON_PARSER=jq
elif command -v python3 >/dev/null 2>&1; then
  JSON_PARSER=python3
else
  fail_json "jq or python3 is required to install zhihu-cli" "INSTALL_PREREQUISITE_MISSING"
fi
if command -v sha256sum >/dev/null 2>&1; then
  SHA256_TOOL=sha256sum
elif command -v shasum >/dev/null 2>&1; then
  SHA256_TOOL=shasum
else
  fail_json "sha256sum or shasum is required to install zhihu-cli" "INSTALL_PREREQUISITE_MISSING"
fi

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/zhihu-cli-setup.XXXXXX")
cleanup() { rm -rf -- "$TMP_ROOT"; }
trap cleanup EXIT HUP INT TERM
REMOTE_MANIFEST="$TMP_ROOT/manifest.json"

say "Reading the official zhihu-cli release manifest..."
curl --fail --silent --show-error --connect-timeout 10 --max-time 60 \
  --output "$REMOTE_MANIFEST" "$UPDATE_MANIFEST_URL" || fail_json "Unable to download CLI release manifest" "NETWORK_ERROR"
[ "$(wc -c < "$REMOTE_MANIFEST" | tr -d '[:space:]')" -le 1048576 ] || fail_json "CLI release manifest is too large" "UPDATE_INTEGRITY_FAILED"

manifest_value() {
  case "$JSON_PARSER" in
    plutil) plutil -extract "$2" raw -o - "$1" 2>/dev/null || true ;;
    jq) jq -er --arg path "$2" 'getpath($path | split("."))' "$1" 2>/dev/null || true ;;
    python3)
      python3 - "$1" "$2" 2>/dev/null <<'PY' || true
import json
import sys

with open(sys.argv[1], encoding="utf-8") as manifest_file:
    value = json.load(manifest_file)
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
      ;;
  esac
}
SCHEMA_VERSION=$(manifest_value "$REMOTE_MANIFEST" schema_version)
LATEST_VERSION=$(manifest_value "$REMOTE_MANIFEST" cli.latest_version)
ARTIFACT_URL=$(manifest_value "$REMOTE_MANIFEST" "cli.artifacts.$PLATFORM.url")
EXPECTED_SHA=$(manifest_value "$REMOTE_MANIFEST" "cli.artifacts.$PLATFORM.sha256")
EXPECTED_SIZE=$(manifest_value "$REMOTE_MANIFEST" "cli.artifacts.$PLATFORM.size")
[ "$SCHEMA_VERSION" = "1" ] || fail_json "CLI release manifest schema is unsupported" "UPSTREAM_ERROR"
printf '%s' "$LATEST_VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$' || fail_json "CLI release version is invalid" "UPSTREAM_ERROR"
version_ge "$LATEST_VERSION" "$MIN_VERSION" || fail_json "Latest CLI does not satisfy this Skill" "INCOMPATIBLE_VERSION"
printf '%s' "$EXPECTED_SHA" | grep -Eq '^[0-9a-f]{64}$' || fail_json "CLI artifact SHA-256 is invalid" "UPSTREAM_ERROR"
case "$EXPECTED_SIZE" in ''|*[!0-9]*) fail_json "CLI artifact size is invalid" "UPSTREAM_ERROR" ;; esac
[ "$EXPECTED_SIZE" -gt 0 ] && [ "$EXPECTED_SIZE" -le 134217728 ] || fail_json "CLI artifact size is invalid" "UPSTREAM_ERROR"

case "$ARTIFACT_URL" in
  https://"$MANIFEST_AUTHORITY"/*) ;;
  http://"$MANIFEST_AUTHORITY"/*)
    [ "${ZHIHU_CLI_ALLOW_INSECURE_TEST_URL:-}" = "1" ] || fail_json "CLI artifact host differs from manifest host" "UPDATE_INTEGRITY_FAILED"
    ;;
  *) fail_json "CLI artifact host differs from manifest host" "UPDATE_INTEGRITY_FAILED" ;;
esac

ARCHIVE="$TMP_ROOT/$(basename "${ARTIFACT_URL%%\?*}")"
STAGED="$TMP_ROOT/zhihu-cli"
say "Downloading zhihu-cli $LATEST_VERSION for $PLATFORM..."
curl --fail --silent --show-error --connect-timeout 10 --max-time 60 \
  --output "$ARCHIVE" "$ARTIFACT_URL" || fail_json "Unable to download CLI artifact" "NETWORK_ERROR"
ACTUAL_SIZE=$(wc -c < "$ARCHIVE" | tr -d '[:space:]')
[ "$ACTUAL_SIZE" = "$EXPECTED_SIZE" ] || fail_json "CLI artifact size does not match manifest" "UPDATE_INTEGRITY_FAILED"
if [ "$SHA256_TOOL" = sha256sum ]; then
  ACTUAL_SHA=$(sha256sum "$ARCHIVE" | awk '{print $1}')
else
  ACTUAL_SHA=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
fi
[ "$ACTUAL_SHA" = "$EXPECTED_SHA" ] || fail_json "CLI artifact SHA-256 does not match manifest" "CHECKSUM_MISMATCH"

# 归档必须且只能包含根目录下的单一 zhihu-cli，拒绝额外成员和路径穿越。
ARCHIVE_LIST=$(tar -tzf "$ARCHIVE" 2>/dev/null) || fail_json "CLI archive is invalid" "UPDATE_INTEGRITY_FAILED"
[ "$ARCHIVE_LIST" = "zhihu-cli" ] || fail_json "CLI archive structure is invalid" "UPDATE_INTEGRITY_FAILED"
tar -xOzf "$ARCHIVE" zhihu-cli > "$STAGED" 2>/dev/null || fail_json "CLI archive does not contain zhihu-cli" "UPDATE_INTEGRITY_FAILED"
chmod 755 "$STAGED"
STAGED_VERSION=$(read_version "$STAGED")
[ "$STAGED_VERSION" = "$LATEST_VERSION" ] || fail_json "Downloaded CLI version does not match manifest" "BINARY_INVALID"

VERSION_DIR="$CLI_HOME/versions/$LATEST_VERSION"
DEST="$VERSION_DIR/zhihu-cli"
mkdir -p "$CLI_HOME" "$CLI_HOME/versions" "$VERSION_DIR" "$CURRENT_DIR"
chmod 700 "$CLI_HOME" "$CLI_HOME/versions" "$VERSION_DIR" "$CURRENT_DIR"
cp "$STAGED" "$VERSION_DIR/.zhihu-cli.new.$$"
chmod 755 "$VERSION_DIR/.zhihu-cli.new.$$"
mv -f "$VERSION_DIR/.zhihu-cli.new.$$" "$DEST"
cp "$DEST" "$CURRENT_DIR/.zhihu-cli.new.$$"
chmod 755 "$CURRENT_DIR/.zhihu-cli.new.$$"
mv -f "$CURRENT_DIR/.zhihu-cli.new.$$" "$CURRENT"

AUTH_CONFIGURED=false
NEXT_ACTION=request_access_secret
if AUTH_JSON=$("$CURRENT" auth status 2>/dev/null); then
  if printf '%s' "$AUTH_JSON" | grep -q '"source":"'; then AUTH_CONFIGURED=true; NEXT_ACTION=ready; fi
fi

ABS_PATH=$(CDPATH= cd -- "$CURRENT_DIR" && pwd)/zhihu-cli
say "[OK] zhihu-cli $LATEST_VERSION is ready at $ABS_PATH"
printf '{"ok":true,"installed":true,"downloaded_cli_version":"%s","binary_path":"%s","auth_configured":%s,"next_action":"%s"}\n' \
  "$LATEST_VERSION" "$(json_escape "$ABS_PATH")" "$AUTH_CONFIGURED" "$NEXT_ACTION"
