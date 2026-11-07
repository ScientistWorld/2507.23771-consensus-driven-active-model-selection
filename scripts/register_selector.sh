#!/bin/bash
# Register a selector policy so the evaluators run it.
#
# Usage: bash scripts/register_selector.sh <name> <command> [args...]
# Writes outputs/selectors/<name>.json = {"command": [<command>, args...]}.
# The command runs from the repository root and must speak the label-oracle
# protocol documented in data/selection_protocol.py.

set -euo pipefail

cd /home/user

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <name> <command> [args...]" >&2
  exit 2
fi
NAME="$1"
shift
SELECTORS_DIR="${SELECTORS_DIR:-outputs/selectors}"
mkdir -p "$SELECTORS_DIR"
python3 -c 'import json, sys; json.dump({"command": sys.argv[2:]}, open(sys.argv[1], "w"), indent=2)' \
  "$SELECTORS_DIR/$NAME.json" "$@"
echo "registered selector '$NAME' -> $SELECTORS_DIR/$NAME.json"
