#!/usr/bin/env bash
# start_demo.sh — palaiž Fraud Detection demo lokāli (bez Docker)
#   API: http://127.0.0.1:5050   UI: http://localhost:3000
# Lietošana:  ./start_demo.sh      (Ctrl+C aptur abus)

set -e
cd "$(dirname "$0")"

# ── Python vide API ───────────────────────────────────────────────────────────
if [ ! -x .venv/bin/python ]; then
    echo "📦 Veidoju Python vidi (.venv)..."
    PY=$(command -v python3.11 || command -v python3)
    "$PY" -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    # scikit-learn 1.8.0 — ar šo versiju saglabāts modelis un scaler
    .venv/bin/pip install -q flask flask-cors joblib numpy pandas "scikit-learn==1.8.0" lightgbm
fi

# ── Node atkarības UI ─────────────────────────────────────────────────────────
if [ ! -x fraud-ui/node_modules/.bin/next ]; then
    echo "📦 Instalēju UI atkarības..."
    (cd fraud-ui && npm ci --no-audit --no-fund)
fi

# ── Atbrīvo portus, ja kaut kas jau darbojas ──────────────────────────────────
for port in 5050 3000; do
    pids=$(lsof -ti tcp:$port -sTCP:LISTEN || true)
    [ -n "$pids" ] && kill $pids && sleep 1
done

cleanup() { echo; echo "🛑 Apturu demo..."; kill $API_PID $UI_PID 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# ── Palaiž API ────────────────────────────────────────────────────────────────
.venv/bin/python api.py &
API_PID=$!
for _ in $(seq 1 60); do curl -sf http://127.0.0.1:5050/health >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:5050/health >/dev/null || { echo "❌ API nepalaidās"; exit 1; }
echo "✅ API darbojas: http://127.0.0.1:5050"

# ── Palaiž UI ─────────────────────────────────────────────────────────────────
(cd fraud-ui && NEXT_TELEMETRY_DISABLED=1 ./node_modules/.bin/next dev) &
UI_PID=$!
for _ in $(seq 1 60); do curl -sf -o /dev/null http://localhost:3000 && break; sleep 1; done
echo "✅ UI darbojas: http://localhost:3000"
open http://localhost:3000 2>/dev/null || true

wait
