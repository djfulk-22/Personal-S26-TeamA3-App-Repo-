#!/bin/zsh
set -euo pipefail

REPO="/Users/danielfulk/GitHub/Personal-S26-TeamA3-App-Repo-"
PYTHON_BIN="/Users/danielfulk/Downloads/miniconda3/envs/tfenv/bin/python"
LOG_DIR="$REPO/automation_logs"

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/run_refresh_launchd.log" 2>&1

echo "=================================================="
echo "Starting refresh: $(date)"
echo "Repo: $REPO"

cd "$REPO"
"$PYTHON_BIN" SCRAPERS/run_refresh_and_push.py

echo "Finished refresh: $(date)"
echo "=================================================="