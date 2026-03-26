#!/usr/bin/env bash
# watch-autonomous.sh — Monitor a ForgeOS autonomous session
# Usage: ./scripts/watch-autonomous.sh [port] [log_file]

set -euo pipefail

PORT="${1:-3001}"
LOG_FILE="${2:-adapsisos.log}"
API="http://127.0.0.1:${PORT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

STUCK_THRESHOLD=60  # seconds before "stuck" alert
last_status_hash=""
last_change_time=$(date +%s)

header() {
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  ForgeOS Autonomous Monitor — :${PORT}${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
    echo
}

poll_status() {
    local resp
    resp=$(curl -sf "${API}/api/status" 2>/dev/null) || { echo -e "${RED}[!] ForgeOS not responding${NC}"; return 1; }

    local revision functions types
    revision=$(echo "$resp" | jq -r '.revision // 0')
    functions=$(echo "$resp" | jq -r '.functions // [] | length')
    types=$(echo "$resp" | jq -r '.types // [] | length')
    local fn_list
    fn_list=$(echo "$resp" | jq -r '.functions // [] | join(", ")')

    echo -e "${GREEN}[status]${NC} rev=${BOLD}${revision}${NC}  fns=${functions}  types=${types}"
    if [ -n "$fn_list" ]; then
        echo -e "  ${DIM}functions: ${fn_list}${NC}"
    fi

    # Detect stuck state
    local current_hash
    current_hash=$(echo "$resp" | md5sum | cut -d' ' -f1)
    local now
    now=$(date +%s)
    if [ "$current_hash" = "$last_status_hash" ]; then
        local elapsed=$(( now - last_change_time ))
        if [ "$elapsed" -gt "$STUCK_THRESHOLD" ]; then
            echo -e "${RED}${BOLD}[STUCK]${NC}${RED} No state change for ${elapsed}s!${NC}"
        fi
    else
        last_status_hash="$current_hash"
        last_change_time=$now
    fi
}

poll_tasks() {
    local resp
    resp=$(curl -sf "${API}/api/tasks" 2>/dev/null) || return 0

    local count
    count=$(echo "$resp" | jq -r '.tasks // [] | length')
    if [ "$count" -gt 0 ]; then
        echo -e "${BLUE}[tasks]${NC} ${count} active:"
        echo "$resp" | jq -r '.tasks[] | "  \(.id): \(.function_name) — \(.status)"' 2>/dev/null || true
    fi
}

poll_plan() {
    # Query for the current plan via the query endpoint
    local resp
    resp=$(curl -sf -X POST "${API}/api/query" -H 'Content-Type: application/json' -d '{"query":"?plan"}' 2>/dev/null) || return 0

    local plan
    plan=$(echo "$resp" | jq -r '.response // ""')
    if [ -n "$plan" ] && [ "$plan" != "No plan set." ] && [ "$plan" != "null" ]; then
        echo -e "${YELLOW}[plan]${NC}"
        echo "$plan" | head -20 | sed 's/^/  /'
    fi
}

tail_log() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${DIM}--- Live log (${LOG_FILE}) ---${NC}"
        tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
            # Color errors red, mutations green, AI responses cyan
            if echo "$line" | grep -qi 'error\|fail\|panic'; then
                echo -e "${RED}${line}${NC}"
            elif echo "$line" | grep -qi 'mutation\|applied\|OK:'; then
                echo -e "${GREEN}${line}${NC}"
            elif echo "$line" | grep -qi 'llm\|response\|thinking'; then
                echo -e "${CYAN}${line}${NC}"
            else
                echo -e "${DIM}${line}${NC}"
            fi
        done &
        TAIL_PID=$!
    fi
}

cleanup() {
    [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null || true
    echo -e "\n${YELLOW}Monitor stopped.${NC}"
    exit 0
}
trap cleanup EXIT INT TERM

# Main
header

# Wait for ForgeOS to be reachable
echo -e "${YELLOW}Waiting for ForgeOS on :${PORT}...${NC}"
for i in $(seq 1 30); do
    if curl -sf "${API}/api/status" >/dev/null 2>&1; then
        echo -e "${GREEN}Connected.${NC}"
        break
    fi
    sleep 1
done

# Start tailing log in background
tail_log

echo
echo -e "${YELLOW}Polling every 10s. Ctrl+C to stop.${NC}"
echo

# Poll loop
while true; do
    echo -e "${DIM}--- $(date '+%H:%M:%S') ---${NC}"
    poll_status
    poll_tasks
    poll_plan
    echo
    sleep 10
done
