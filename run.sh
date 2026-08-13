#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
exec uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
