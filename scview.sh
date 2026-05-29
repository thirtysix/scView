#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ICON="$DIR/frontend/public/icon.png"
LOG_FILE="$DIR/log/scview-launch.log"

# Mode: env var > flag > default (dev)
MODE="${SCVIEW_MODE:-dev}"

# ── helpers ──────────────────────────────────────────────────────────

compose_args() {
    if [ "$MODE" = "dev" ]; then
        echo "-f docker-compose.dev.yml"
    fi
}

frontend_url() {
    if [ "$MODE" = "dev" ]; then
        echo "http://localhost:5173"
    else
        echo "http://localhost:3000"
    fi
}

is_running() {
    local count
    count=$(cd "$DIR" && docker compose $(compose_args) ps --status running -q 2>/dev/null | wc -l)
    [ "$count" -gt 0 ]
}

container_count() {
    cd "$DIR" && docker compose $(compose_args) ps --status running -q 2>/dev/null | wc -l
}

check_docker() {
    if ! command -v docker &>/dev/null; then
        notify "scView" "Docker is not installed."
        exit 1
    fi
    if ! docker info &>/dev/null 2>&1; then
        notify "scView" "Docker daemon is not running.\nStart Docker Desktop or the Docker service."
        exit 1
    fi
}

wait_for_health() {
    local url
    url="$(frontend_url)"
    local max_wait=90
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        # Bail early if containers exited (build failure)
        local running
        running=$(container_count)
        if [ "$running" -eq 0 ] && [ $waited -gt 10 ]; then
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

notify() {
    local title="$1"
    local body="$2"
    echo -e "$title: $body"
    if command -v notify-send &>/dev/null; then
        notify-send -i "$ICON" "$title" "$body"
    fi
}

# ── commands ─────────────────────────────────────────────────────────

do_start() {
    for arg in "$@"; do
        case "$arg" in
            --prod)     MODE="prod" ;;
            --desktop)  ;;  # marker for .desktop file, same as default
        esac
    done

    local url
    url="$(frontend_url)"

    # Already running? Just open the browser.
    if is_running; then
        local count
        count=$(container_count)
        echo "scView is already running ($count containers)"
        notify "scView" "Already running — opening browser"
        xdg-open "$url" 2>/dev/null &
        exit 0
    fi

    check_docker

    # Bootstrap .env if missing
    if [ ! -f "$DIR/.env" ]; then
        cp "$DIR/.env.example" "$DIR/.env"
    fi

    mkdir -p "$DIR/log"

    echo "Starting scView ($MODE mode) at $url"

    if ! (cd "$DIR" && docker compose $(compose_args) up --build -d) >> "$LOG_FILE" 2>&1; then
        notify "scView" "Failed to start. See log/scview-launch.log"
        exit 1
    fi

    if wait_for_health; then
        notify "scView" "Services started ($MODE mode)\n$url"
        xdg-open "$url" 2>/dev/null &
    else
        notify "scView" "Services started but frontend not yet responding.\nCheck: docker compose logs -f"
    fi
}

do_stop() {
    if ! is_running; then
        notify "scView" "Not running."
        exit 0
    fi

    echo "Stopping scView..."
    (cd "$DIR" && docker compose $(compose_args) down)

    notify "scView" "All services stopped."
}

do_status() {
    local msg
    if is_running; then
        local count
        count=$(container_count)
        local url
        url="$(frontend_url)"
        msg="Running ($count/3 containers)\n$url\nLogs: docker compose logs -f"
    else
        msg="Not running."
    fi

    echo -e "scView\n$msg"
    notify "scView" "$msg"
}

# ── main dispatch ────────────────────────────────────────────────────

case "${1:-start}" in
    start)   shift || true; do_start "$@" ;;
    stop)    do_stop ;;
    status)  do_status ;;
    restart) do_stop; sleep 2; shift || true; do_start "$@" ;;
    *)
        echo "Usage: $(basename "$0") {start|stop|status|restart} [--prod]"
        exit 1
        ;;
esac
