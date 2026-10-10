#!/usr/bin/env bash
# One-command run for AutoGuru.
#   ./run.sh            -> trains (first time only), then serves API + UI on :8000
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"

if [ ! -d "models/autogluon" ]; then
  echo "[run.sh] no trained model found -> running train.py first (CPU, a few minutes)"
  "$PY" train.py
fi

echo "[run.sh] running tests"
"$PY" -m pytest tests/ -q

echo "[run.sh] serving on http://0.0.0.0:8000  (API: /predict  UI: /  docs: /docs)"
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8000
