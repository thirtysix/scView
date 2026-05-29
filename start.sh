#!/usr/bin/env bash
#
# scView — Quick Start Script
#
# Checks prerequisites, configures the environment, builds Docker images,
# starts all services, waits for health checks, and opens the browser.
#
# Usage:
#   ./start.sh            # Development mode (hot-reload)
#   ./start.sh --prod     # Production mode (optimized, Nginx)
#   ./start.sh --stop     # Stop all services
#   ./start.sh --clean    # Stop and remove containers, volumes, images
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

# ── Parse arguments ─────────────────────────────────────────────────────────
MODE="dev"
for arg in "$@"; do
    case "$arg" in
        --prod|--production) MODE="prod" ;;
        --stop)
            step "Stopping scView"
            docker compose down
            ok "All services stopped."
            exit 0
            ;;
        --clean)
            step "Cleaning up scView"
            docker compose down -v --rmi local 2>/dev/null || true
            rm -rf frontend/node_modules frontend/dist
            find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            ok "Cleaned containers, volumes, and build artifacts."
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./start.sh [--prod] [--stop] [--clean] [--help]"
            echo ""
            echo "  (default)   Start in development mode (hot-reload)"
            echo "  --prod      Start in production mode (Nginx, optimized)"
            echo "  --stop      Stop all running services"
            echo "  --clean     Stop and remove containers, volumes, images"
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

# Docker
if ! command -v docker &>/dev/null; then
    fail "Docker is not installed. Install it from https://docs.docker.com/get-docker/"
fi
ok "Docker found: $(docker --version | head -1)"

# Docker Compose v2 (as a plugin: "docker compose")
if docker compose version &>/dev/null; then
    COMPOSE_CMD="docker compose"
    ok "Docker Compose found: $(docker compose version --short 2>/dev/null || docker compose version)"
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

    docker compose -f docker-compose.dev.yml up --build -d

    FRONTEND_URL="http://localhost:5173"
    BACKEND_URL="http://localhost:8080"
    CONVERTER_URL="http://localhost:8001"
else
    step "Building and starting services (production mode)"
    info "This may take a few minutes on first run (downloading Docker images)..."
    echo ""

    docker compose build
    docker compose up -d

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
echo -e "    docker compose logs -f          ${CYAN}# View all logs${NC}"
echo -e "    docker compose logs -f backend  ${CYAN}# View backend logs${NC}"
echo -e "    ./start.sh --stop               ${CYAN}# Stop all services${NC}"
echo -e "    ./start.sh --clean              ${CYAN}# Remove everything${NC}"
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
