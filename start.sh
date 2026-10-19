#!/usr/bin/env bash
#
# scView — Quick Start Script
#
# Checks prerequisites, configures the environment, builds Docker images,
# starts all services, waits for health checks, and opens the browser.
#
# Usage:
#   ./start.sh            # Development mode (hot-reload)
#   ./start.sh --prod     # Production mode (optimized, Nginx; prebuilt images)
#   ./start.sh --prod --build  # Production mode, images built from source
#   ./start.sh --stop     # Stop all services
#   ./start.sh --clean    # Stop and remove containers, images, volumes (deletes datasets)
#

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[error]${NC} $*"; exit 1; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Container engine ────────────────────────────────────────────────────────
# Docker when it is running, else Podman, else Docker (to say how to start it).
# SCVIEW_ENGINE=docker|podman picks one when both are installed.
if [ -n "${SCVIEW_ENGINE:-}" ]; then
    ENGINE="$SCVIEW_ENGINE"
elif command -v docker &>/dev/null && docker info &>/dev/null; then
    ENGINE=docker
elif command -v podman &>/dev/null; then
    ENGINE=podman
else
    ENGINE=docker
fi
COMPOSE="$ENGINE compose"

# True when version $1 is at least $2 (major.minor).
ver_ge() {
    local a1 a2 b1 b2
    IFS=. read -r a1 a2 _ <<< "$1"; IFS=. read -r b1 b2 _ <<< "$2"
    a2=${a2%%[!0-9]*}
    [ "${a1:-0}" -gt "$b1" ] || { [ "${a1:-0}" -eq "$b1" ] && [ "${a2:-0}" -ge "$b2" ]; }
}

# True when compose tool $1 can run scView's compose files (for Podman).
compose_ok() {
    local out v
    out=$("$1" version 2>/dev/null) || return 1
    case "$out" in
        *"Docker Compose version"*)
            v=$(echo "$out" | sed -n 's/.*version v\{0,1\}\([0-9][0-9.]*\).*/\1/p' | head -1)
            ver_ge "${v:-0}" 2.24 ;;
        *podman-compose*)
            v=$(echo "$out" | sed -n 's/.*podman-compose version:\{0,1\} *\([0-9][0-9.]*\).*/\1/p' | head -1)
            ver_ge "${v:-0}" 1.2 ;;
        *) return 1 ;;
    esac
}

# `podman compose` runs the first compose tool it finds, and an old
# docker-compose v1 on the PATH can't read scView's compose files. Pick one
# that can: Docker Compose >= 2.24, or podman-compose >= 1.2 (1.3 waits for
# the health checks; 1.2 starts everything at once).
pick_podman_compose() {
    [ -n "${PODMAN_COMPOSE_PROVIDER:-}" ] && return 0
    for c in "$HOME/.docker/cli-plugins/docker-compose" \
             /usr/local/lib/docker/cli-plugins/docker-compose \
             /usr/libexec/docker/cli-plugins/docker-compose \
             /usr/lib/docker/cli-plugins/docker-compose \
             "$(command -v docker-compose || true)" \
             "$(command -v podman-compose || true)" "$HOME/.local/bin/podman-compose"; do
        if [ -n "$c" ] && [ -x "$c" ] && compose_ok "$c"; then
            export PODMAN_COMPOSE_PROVIDER="$c"
            break
        fi
    done
}

# ── Parse arguments ─────────────────────────────────────────────────────────
MODE="dev"
BUILD_FROM_SOURCE=false
for arg in "$@"; do
    case "$arg" in
        --prod|--production) MODE="prod" ;;
        --build) BUILD_FROM_SOURCE=true ;;
        --stop)
            step "Stopping scView"
            [ "$ENGINE" = "podman" ] && pick_podman_compose
            $COMPOSE down
            ok "All services stopped."
            exit 0
            ;;
        --clean)
            step "Cleaning up scView"
            echo -e "  ${RED}This permanently deletes every dataset you have uploaded or converted.${NC}"
            if [ -t 0 ]; then
                read -r -p "  Type 'delete' to continue: " CONFIRM
                [ "$CONFIRM" = "delete" ] || { info "Cancelled — nothing removed."; exit 0; }
            fi
            [ "$ENGINE" = "podman" ] && pick_podman_compose
            $COMPOSE down -v --rmi all 2>/dev/null || true
            rm -rf frontend/node_modules frontend/dist
            find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            ok "Cleaned containers, volumes, and build artifacts."
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./start.sh [--prod [--build]] [--stop] [--clean] [--help]"
            echo ""
            echo "  (default)   Start in development mode (hot-reload)"
            echo "  --prod      Start in production mode (Nginx, optimized), pulling"
            echo "              prebuilt images; anything not published is built"
            echo "  --build     With --prod: build the images from source instead"
            echo "  --stop      Stop all running services"
            echo "  --clean     Stop and remove containers, images, and volumes"
            echo "              (DELETES all uploaded datasets; asks first)"
            exit 0
            ;;
        *) fail "Unknown argument: $arg (use --help for usage)" ;;
    esac
done

# ── Banner ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "  ┌─────────────────────────────────────┐"
echo "  │         scView Quick Start           │"
echo "  │   Single-Cell RNA-Seq Visualizer     │"
echo "  └─────────────────────────────────────┘"
echo -e "${NC}"
info "Mode: ${BOLD}${MODE}${NC}"

# ── 1. Check prerequisites ──────────────────────────────────────────────────
step "Checking prerequisites"

if [ "$ENGINE" = "podman" ]; then
    ok "Podman found: $(podman --version | head -1)"
    # macOS and Windows run Podman in a virtual machine: start it, or make one.
    if ! podman info &>/dev/null && [ "$(uname -s)" != "Linux" ]; then
        if [ -z "$(podman machine list --format '{{.Name}}' 2>/dev/null)" ]; then
            info "Creating the Podman machine (first time only, a few minutes)..."
            podman machine init
        fi
        podman machine start || true
    fi
    if ! podman info &>/dev/null; then
        fail "Podman is not answering. Run 'podman info' to see why (on macOS: 'podman machine start')."
    fi
    # docker-compose v2 as the compose provider talks to Podman's socket.
    if [ "$(uname -s)" = "Linux" ] && command -v systemctl &>/dev/null; then
        systemctl --user start podman.socket 2>/dev/null || true
    fi
    pick_podman_compose
    if [ -z "${PODMAN_COMPOSE_PROVIDER:-}" ]; then
        echo ""
        echo -e "  ${RED}Podman needs a compose tool for scView, and none that works is installed.${NC}"
        echo ""
        echo -e "  Install one of these, then re-run ${BOLD}./start.sh${NC}:"
        echo -e "    ${BOLD}Docker Compose v2 (2.24+):${NC} https://docs.docker.com/compose/install/standalone/"
        echo -e "    ${BOLD}podman-compose 1.3+:${NC}       pip install --user 'podman-compose>=1.3'"
        echo -e "    ${BOLD}macOS:${NC}                     Podman Desktop -> Settings -> Resources -> Compose -> Setup"
        exit 1
    fi
    ok "Compose tool: ${PODMAN_COMPOSE_PROVIDER}"
else
    # Docker
    if ! command -v docker &>/dev/null; then
        fail "Neither Docker nor Podman is installed. Install Docker from https://docs.docker.com/get-docker/ or Podman from https://podman.io/docs/installation"
    fi
    ok "Docker found: $(docker --version | head -1)"

    # Docker Compose v2 (as a plugin: "docker compose")
    if docker compose version &>/dev/null; then
        COMPOSE_CMD="docker compose"
        COMPOSE_VER=$(docker compose version --short 2>/dev/null | sed 's/^v//')
        ok "Docker Compose found: ${COMPOSE_VER:-$(docker compose version)}"
        # The compose files use `env_file: required: false`, added in Compose 2.24.
        IFS=. read -r C_MAJ C_MIN _ <<< "${COMPOSE_VER:-0.0}"
        if [ "${C_MAJ:-0}" -eq 2 ] && [ "${C_MIN%%[!0-9]*}" -lt 24 ] 2>/dev/null; then
            fail "Docker Compose ${COMPOSE_VER} is too old; scView needs 2.24 or newer. Update Docker (https://docs.docker.com/compose/install/)."
        fi
    else
        echo ""
        echo -e "  ${RED}Docker Compose v2 plugin is required but not installed.${NC}"
        echo ""
        echo -e "  Install it with one of these commands:"
        echo ""
        echo -e "    ${BOLD}Ubuntu/Debian:${NC}  sudo apt install docker-compose-v2"
        echo -e "    ${BOLD}Fedora/RHEL:${NC}    sudo dnf install docker-compose-plugin"
        echo -e "    ${BOLD}Arch:${NC}           sudo pacman -S docker-compose"
        echo -e "    ${BOLD}macOS/Windows:${NC}  Included with Docker Desktop"
        echo ""
        echo -e "  Then re-run: ${BOLD}./start.sh${NC}"
        exit 1
    fi

    # Docker daemon running?
    if ! docker info &>/dev/null; then
        fail "Docker daemon is not running. Start Docker Desktop or the Docker service."
    fi
    ok "Docker daemon is running."
fi

# Check available memory (warn if <6GB)
if command -v free &>/dev/null; then
    TOTAL_MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM_MB" -lt 6144 ]; then
        warn "System has ${TOTAL_MEM_MB}MB RAM. scView recommends at least 8GB for smooth operation."
    else
        ok "System memory: ${TOTAL_MEM_MB}MB"
    fi
fi

# curl (needed for health checks)
if ! command -v curl &>/dev/null; then
    warn "curl not found — health checks will be skipped."
    HAS_CURL=false
else
    HAS_CURL=true
fi

# ── 2. Environment configuration ────────────────────────────────────────────
step "Configuring environment"

if [ ! -f .env ]; then
    cp .env.example .env
    ok "Created .env from .env.example"

    # Check if user has a DeepInfra key to set
    if [ -t 0 ]; then
        echo ""
        echo -e "  ${YELLOW}Optional:${NC} Enter your DeepInfra API key for AI-powered analysis suggestions."
        echo -e "  Press Enter to skip (the app works fully without it)."
        echo -n "  DEEPINFRA_API_KEY: "
        read -r API_KEY
        if [ -n "$API_KEY" ]; then
            sed -i "s/^DEEPINFRA_API_KEY=.*/DEEPINFRA_API_KEY=${API_KEY}/" .env
            ok "DeepInfra API key saved to .env"
        else
            info "Skipped — rule-based suggestions will be used instead."
        fi
    fi
else
    ok ".env already exists — using existing configuration."
fi

# ── 3. Build and start services ─────────────────────────────────────────────
if [ "$MODE" = "dev" ]; then
    step "Building and starting services (development mode)"
    info "This may take a few minutes on first run (downloading Docker images)..."
    echo ""

    $COMPOSE -f docker-compose.dev.yml up --build -d

    FRONTEND_URL="http://localhost:5173"
    BACKEND_URL="http://localhost:8080"
    CONVERTER_URL="http://localhost:8001"
else
    step "Starting services (production mode)"
    if [ "$BUILD_FROM_SOURCE" = true ]; then
        info "Building images from source; the first build can take tens of minutes..."
        echo ""
        $COMPOSE build
    else
        info "Downloading prebuilt images (about 1 GB on the first run)."
        info "Any image that isn't published is built from source instead, which is slower."
        echo ""
        if [ "$ENGINE" = "docker" ]; then
            $COMPOSE pull --ignore-pull-failures
        else
            $COMPOSE pull || warn "Some images could not be downloaded; they will be built from source."
        fi
    fi
    $COMPOSE up -d

    FRONTEND_URL="http://localhost:3000"
    BACKEND_URL="http://localhost:8080"
    CONVERTER_URL="http://localhost:8001"
fi

# ── 4. Wait for services to be healthy ──────────────────────────────────────
step "Waiting for services to start"

wait_for_service() {
    local name="$1"
    local url="$2"
    local max_attempts="${3:-30}"
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if $HAS_CURL && curl -sf "$url" >/dev/null 2>&1; then
            ok "$name is ready ($url)"
            return 0
        fi
        printf "  Waiting for %s... (%d/%d)\r" "$name" "$attempt" "$max_attempts"
        sleep 2
        attempt=$((attempt + 1))
    done

    warn "$name did not respond within $((max_attempts * 2))s — it may still be starting."
    return 1
}

if $HAS_CURL; then
    echo ""
    wait_for_service "R Converter" "${CONVERTER_URL}/health" 45
    wait_for_service "Backend API" "${BACKEND_URL}/health" 30
    wait_for_service "Frontend"    "${FRONTEND_URL}" 20
else
    info "Skipping health checks (curl not available). Waiting 30s for services..."
    sleep 30
fi

# ── 5. Summary ──────────────────────────────────────────────────────────────
step "scView is running!"
echo ""
echo -e "  ${BOLD}Frontend${NC}    ${GREEN}${FRONTEND_URL}${NC}"
echo -e "  ${BOLD}Backend API${NC} ${GREEN}${BACKEND_URL}${NC}"
echo -e "  ${BOLD}API Docs${NC}    ${GREEN}${BACKEND_URL}/docs${NC}"
echo -e "  ${BOLD}Converter${NC}   ${GREEN}${CONVERTER_URL}${NC}"
echo ""
echo -e "  ${BOLD}Useful commands:${NC}"
if [ "$ENGINE" = "podman" ]; then
    echo -e "    export PODMAN_COMPOSE_PROVIDER=${PODMAN_COMPOSE_PROVIDER}  ${CYAN}# once per terminal${NC}"
fi
echo -e "    $COMPOSE logs -f          ${CYAN}# View all logs${NC}"
echo -e "    $COMPOSE logs -f backend  ${CYAN}# View backend logs${NC}"
echo -e "    ./start.sh --stop               ${CYAN}# Stop all services${NC}"
echo -e "    ./start.sh --clean              ${CYAN}# Remove everything, including datasets${NC}"
echo ""

# ── 6. Open browser ─────────────────────────────────────────────────────────
open_browser() {
    local url="$1"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$url" 2>/dev/null &
    elif command -v open &>/dev/null; then
        open "$url" 2>/dev/null &
    elif command -v wslview &>/dev/null; then
        wslview "$url" 2>/dev/null &
    else
        return 1
    fi
    return 0
}

if open_browser "$FRONTEND_URL"; then
    ok "Opened ${FRONTEND_URL} in your browser."
else
    info "Open ${BOLD}${FRONTEND_URL}${NC} in your browser to get started."
fi

echo ""
info "Upload a .h5ad or .rds file to begin exploring your data."
echo ""
