#!/bin/sh
set -eu

fail_package() {
  printf '{"ok":false,"error":{"code":"INVALID_PACKAGE","message":"%s"}}\n' "$1"
  exit 7
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SKILL_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
MANIFEST="$SKILL_DIR/manifest.json"
[ -f "$MANIFEST" ] || fail_package "manifest.json is missing"
SKILL_VERSION=$(sed -n 's/^[[:space:]]*"version":[[:space:]]*"\([^"]*\)".*/\1/p' "$MANIFEST" | head -n 1)
MIN_VERSION=$(sed -n 's/.*"min_version":[[:space:]]*"\([^"]*\)".*/\1/p' "$MANIFEST" | head -n 1)
[ -n "$SKILL_VERSION" ] || fail_package "manifest version is empty"
[ -n "$MIN_VERSION" ] || fail_package "manifest cli.min_version is empty"
if [ -n "${ZHIHU_CLI_HOME:-}" ]; then
  CLI_HOME=$ZHIHU_CLI_HOME
  if [ "$(uname -s)" = Linux ]; then
    case "$CLI_HOME" in /*) ;; *) fail_package "ZHIHU_CLI_HOME must be an absolute path" ;; esac
  fi
else
  case "$(uname -s)" in
    Darwin)
      [ -n "${HOME:-}" ] || fail_package "HOME is required"
      CLI_HOME="$HOME/Library/Application Support/zhihu-cli"
      ;;
    Linux)
      [ -n "${HOME:-}" ] || fail_package "HOME is required"
      DATA_HOME=${XDG_DATA_HOME:-"$HOME/.local/share"}
      case "$DATA_HOME" in /*) ;; *) fail_package "XDG_DATA_HOME must be an absolute path" ;; esac
      CLI_HOME="$DATA_HOME/zhihu-cli"
      ;;
    *) fail_package "unsupported operating system" ;;
  esac
fi
BINARY="$CLI_HOME/current/zhihu-cli"

installed_version() {
  "$BINARY" version 2>/dev/null | sed -n 's/.*"version":"\([^"]*\)".*/\1/p'
}

version_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    split(a, x, /[.-]/); split(b, y, /[.-]/)
    for (i=1; i<=3; i++) {
      if ((x[i]+0) > (y[i]+0)) exit 0
      if ((x[i]+0) < (y[i]+0)) exit 1
    }
    if (a ~ /-/ && b !~ /-/) exit 1
    exit 0
  }'
}

is_ready() {
  [ -x "$BINARY" ] || return 1
  current=$(installed_version)
  [ -n "$current" ] && version_ge "$current" "$MIN_VERSION"
}

# 使用安装器同样支持的解析器验证 JSON；不把解析器缺失误报为 CLI 未安装。
status_field() {
  case "$status_parser" in
    plutil) printf '%s' "$status_output" | plutil -extract "$1" raw -expect "$2" -o - - 2>/dev/null ;;
    jq)
      printf '%s' "$status_output" | jq -ers --arg path "$1" --arg kind "$2" '
        if length != 1 then error("multiple values") else .[0] end |
        getpath($path | split(".")) |
        if type == (if $kind == "bool" then "boolean" else $kind end)
        then tostring else error("invalid type") end' 2>/dev/null
      ;;
    python3)
      printf '%s' "$status_output" | python3 -c '
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1].split("."):
    value = value[key]
expected = bool if sys.argv[2] == "bool" else str
if type(value) is not expected:
    sys.exit(1)
print(str(value).lower() if expected is bool else value)
' "$1" "$2" 2>/dev/null
      ;;
  esac
}

valid_status() {
  if [ "$(uname -s)" = Darwin ] && command -v plutil >/dev/null 2>&1; then
    status_parser=plutil
  elif command -v jq >/dev/null 2>&1; then
    status_parser=jq
  elif command -v python3 >/dev/null 2>&1; then
    status_parser=python3
  else
    return 2
  fi
  # plutil 也接受 plist；状态协议仅接受 JSON 对象。
  case "$(printf '%s' "$status_output" | sed 's/^[[:space:]]*//' | head -n 1)" in
    \{*) ;; *) return 1 ;;
  esac
  [ "$(status_field ok bool)" = true ] || return 1
  [ "$(status_field installed bool)" = true ] || return 1
  status_compatible=$(status_field cli.compatible bool) || return 1
  status_auth=$(status_field auth.configured bool) || return 1
  [ -n "$(status_field cli.current_version string)" ] || return 1
  [ -n "$(status_field cli.binary_path string)" ] || return 1
  status_action=$(status_field next_action string) || return 1
  if [ "$status_compatible" = false ]; then
    [ "$status_action" = request_cli_upgrade_consent ]
  elif [ "$status_auth" = false ]; then
    [ "$status_action" = request_access_secret ]
  else
    [ "$status_action" = ready ]
  fi
}

if [ "${1:-}" = "status" ]; then
  # 先确认 CLI 能报告版本，保留损坏安装的修复入口；此处不要求版本兼容。
  if [ -x "$BINARY" ] && [ -n "$(installed_version)" ]; then
    if status_output=$("$BINARY" status --skill-version "$SKILL_VERSION" --min-cli-version "$MIN_VERSION" 2>/dev/null) && [ -n "$status_output" ]; then
      if valid_status; then
        printf '%s\n' "$status_output"
        exit 0
      else
        validation_code=$?
        if [ "$validation_code" -eq 2 ]; then
          printf '%s\n' '{"ok":false,"error":{"code":"STATUS_VALIDATOR_UNAVAILABLE","message":"jq or python3 is required to validate CLI status"}}'
          exit 7
        fi
      fi
    fi
  fi
  printf '{"ok":true,"installed":false,"skill":{"current_version":"%s","min_cli_version":"%s"},"auth":{"configured":false},"update_check":{"status":"not_applicable"},"next_action":"request_install_consent"}\n' "$SKILL_VERSION" "$MIN_VERSION"
  exit 0
fi

if [ "${1:-}" = "setup" ]; then
  shift
  exec "$SCRIPT_DIR/setup.sh" "$@"
fi

if ! is_ready; then
  printf '%s\n' '{"ok":false,"error":{"code":"CLI_NOT_INSTALLED","message":"Run scripts/setup.sh after the user approves installation"}}'
  exit 7
fi

exec "$BINARY" "$@"
