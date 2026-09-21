#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"

# Backend
if [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi

export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/backend/data"

echo "Starting API on http://127.0.0.1:8472"
(
  cd "$ROOT/backend"
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8472
) &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
echo "Starting UI on http://127.0.0.1:5284"
cd "$ROOT/frontend"
npm run dev -- --host 0.0.0.0 --port 5284
