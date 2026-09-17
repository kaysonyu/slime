#!/usr/bin/env bash
set -euo pipefail
case "${1:---dry-run}" in
  --dry-run) action=plan ;;
  --submit) action=submit ;;
  *) echo 'Usage: submit_inspire.sh [--dry-run|--submit]' >&2; exit 2 ;;
esac
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/launch.py" "$action"
