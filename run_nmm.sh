#!/usr/bin/env bash
# run_nmm.sh — Launch Nine Men's Morris web server and open the browser.

set -euo pipefail

NMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$NMM_DIR/.venv/bin/python"
VENV_UV="$NMM_DIR/.venv/bin/uvicorn"
HOST="127.0.0.1"
PORT="8000"
URL="http://$HOST:$PORT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[NMM]${NC} $*"; }
warn()  { echo -e "${YELLOW}[NMM]${NC} $*"; }
error() { echo -e "${RED}[NMM]${NC} $*" >&2; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[ -f "$VENV_PY" ]  || error "Virtual environment not found. Run ./install.sh first."
[ -f "$VENV_UV" ]  || error "uvicorn not found in venv. Run ./install.sh first."

# ── Ensure Ollama is running ──────────────────────────────────────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    warn "Ollama not running — starting it in the background..."
    ollama serve &>/dev/null &
    for i in $(seq 1 15); do
        sleep 1
        curl -s http://localhost:11434/api/tags &>/dev/null && break
        [ "$i" -eq 15 ] && warn "Ollama did not respond — LLM features will be disabled."
    done
fi

# ── Check port availability ───────────────────────────────────────────────────
if lsof -i :"$PORT" &>/dev/null 2>&1; then
    warn "Port $PORT is already in use. Trying port 8080..."
    PORT="8080"
    URL="http://$HOST:$PORT"
fi

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    info "Shutting down..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Launch server ─────────────────────────────────────────────────────────────
info "Starting Nine Men's Morris at $URL ..."
cd "$NMM_DIR"
"$VENV_UV" web.app:app --host "$HOST" --port "$PORT" --reload &
SERVER_PID=$!

# Wait for the server to be ready (poll /api/ping; -f fails on HTTP error codes)
info "Waiting for server to be ready..."
for i in $(seq 1 60); do
    sleep 0.5
    curl -sf "$URL/api/ping" &>/dev/null && break
    [ "$i" -eq 60 ] && warn "Server took too long to respond — opening browser anyway."
done

# ── Open browser ──────────────────────────────────────────────────────────────
info "Opening browser at $URL"
if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" &>/dev/null &
elif command -v open &>/dev/null; then
    open "$URL" &
elif command -v wslview &>/dev/null; then
    wslview "$URL" &
else
    warn "Could not detect a browser opener. Open $URL manually."
fi

echo ""
echo -e "${GREEN}Nine Men's Morris is running at $URL${NC}"
echo "  Press Ctrl-C to stop."
echo ""

# ── Keep running until interrupted ───────────────────────────────────────────
wait "$SERVER_PID"
