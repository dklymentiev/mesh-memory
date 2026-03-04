#!/usr/bin/env bash
# Sync Papers documents to Mesh Memory
# Intended to run via cron every 2 hours
#
# Only imports new documents (deduplication by papers-guid tag)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/sync-papers.log"
VENV="$PROJECT_DIR/venv"

mkdir -p "$LOG_DIR"

{
    echo "=== $(date -Iseconds) ==="

    # Activate venv
    source "$VENV/bin/activate"

    # Run import
    python3 "$SCRIPT_DIR/import_papers.py" 2>&1

    echo "--- done ---"
    echo ""
} >> "$LOG_FILE" 2>&1

# Keep log under 1MB
if [ -f "$LOG_FILE" ] && [ "$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE")" -gt 1048576 ]; then
    tail -n 500 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi
