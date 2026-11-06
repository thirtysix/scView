#!/usr/bin/env bash
# Checks start.sh's memory suggestion: its arithmetic, and its "Memory limit" section run as
# start.sh runs it (set -euo pipefail) with docker, podman, uname and free memory stood in for.
# Run: bash scripts/test_start_memory.sh
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
section=$(sed -n '/^# ── Memory limit/,/^# ── 3\./p' "$here/start.sh")
[ -n "$section" ] || { echo "FAIL: no Memory limit section in start.sh"; exit 1; }
# Free memory as given, not this computer's.
[[ "$section" == *'FREE_GB=$(free_memory_gb)'* ]] || { echo "FAIL: start.sh no longer sets FREE_GB=\$(free_memory_gb)"; exit 1; }
section=${section//'FREE_GB=$(free_memory_gb)'/'FREE_GB=$FREE'}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fails=0
out=""

check() {  # check NAME EXPECTED ACTUAL
    if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: expected [$2], got [$3]"; fails=$((fails + 1)); fi
}

contains() {  # contains NAME NEEDLE HAYSTACK
    if [[ "$3" == *"$2"* ]]; then echo "ok    $1"; else echo "FAIL  $1: no [$2] in:"; echo "$3"; fails=$((fails + 1)); fi
}

lacks() {  # lacks NAME NEEDLE HAYSTACK
    if [[ "$3" != *"$2"* ]]; then echo "ok    $1"; else echo "FAIL  $1: [$2] in:"; echo "$3"; fails=$((fails + 1)); fi
}

# run ENGINE OS FREE_GB ENGINE_BYTES [ENV] [DOCKER_OS]: the section run in a bash of its own
# (set -e holds whatever calls it): its output; it leaves $work/.env. FREE_GB "" = free memory
# not known; ENGINE_BYTES "" = engine info fails; ENV = .env before it (default: the empty
# line a fresh .env gets from .env.example).
run() {
    printf '%s\n' "${5-SCVIEW_MEMORY_GB=}" > "$work/.env"
    {
        echo 'set -euo pipefail'
        printf 'ENGINE=%q OS=%q FREE=%q BYTES=%q DOCKER_OS=%q\n' "$1" "$2" "$3" "$4" "${6:-Ubuntu 24.04}"
        echo 'info() { echo "[info] $*"; }'
        echo 'ok() { echo "[ok] $*"; }'
        echo 'warn() { echo "[warn] $*"; }'
        echo 'uname() { echo "$OS"; }'
        echo 'engine_info() {'
        echo '    case "$*" in'
        echo '        *MemTotal*) if [ -n "$BYTES" ]; then echo "$BYTES"; else return 1; fi ;;'
        echo '        *OperatingSystem*) echo "$DOCKER_OS" ;;'
        echo '    esac'
        echo '}'
        echo 'docker() { engine_info "$@"; }'
        echo 'podman() { engine_info "$@"; }'
        echo "$section"
        echo 'echo "[end]"'
    } > "$work/run.sh"
    (cd "$work" && bash run.sh < /dev/null 2>&1)
}

go() {  # go NAME ARGS...: run ARGS into $out; a FAIL when the section stops before its end
    local name=$1
    shift
    if ! out=$(run "$@") || [[ "$out" != *"[end]"* ]]; then
        echo "FAIL  $name: the section stopped:"; echo "$out"; fails=$((fails + 1))
    fi
}

gb() { awk -v g="$1" 'BEGIN {printf "%d", g * 1073741824}'; }

# -- the arithmetic -----------------------------------------------------------------------
eval "$(sed -n '/^engine_cap_gb()/,/^}/p; /^suggest_memory_gb()/,/^}/p' "$here/start.sh")"
check "VM finding: 3.3 GB free, Podman machine 3.8 GB -> 3, wanted 4" "3 4" "$(suggest_memory_gb 3.3 3.8 4)"
check "engine not known: the 4 GB floor" "4 4" "$(suggest_memory_gb 3.3 "" 4)"
check "engine 0: not known" "4 4" "$(suggest_memory_gb 3.3 0 4)"
check "floor kept when the engine has room for it (4.5 GB)" "4 4" "$(suggest_memory_gb 3.3 4.5 4)"
check "4.4 GB engine: 3" "3 4" "$(suggest_memory_gb 3.3 4.4 4)"
check "big free memory, Docker Desktop 7.7 GB: 7" "7 17" "$(suggest_memory_gb 20 7.7 4)"
check "big free memory, big engine: free - 3" "17 17" "$(suggest_memory_gb 20 31.2 4)"
check "rounding: 7.4 free -> 4" "4 4" "$(suggest_memory_gb 7.4 31 4)"
check "rounding: 7.6 free -> 5" "5 5" "$(suggest_memory_gb 7.6 31 4)"
check "tiny engine: at least 1" "1 4" "$(suggest_memory_gb 3.3 1.2 4)"
check "free not known" "4 4" "$(suggest_memory_gb "" 16 4)"
check "cap of 3.8" "3" "$(engine_cap_gb 3.8)"
check "cap not known" "" "$(engine_cap_gb "")"

# -- the section as start.sh runs it ------------------------------------------------------
go "macOS Podman 3.8 GB" podman Darwin 3.3 "$(gb 3.8)"
[ -z "${SHOW:-}" ] || echo "$out"  # SHOW=1: print what this one says
check "macOS Podman 3.8 GB: saved 3" "SCVIEW_MEMORY_GB=3" "$(cat "$work/.env")"
contains "says the machine is small" "The Podman machine has only 3.8 GB of memory, so scView can use at most 3 GB" "$out"
contains "how to give it more" "podman machine stop; podman machine set --memory 5120; podman machine start" "$out"
contains "under-4 warning kept" "Under 4 GB: processing a 14,000-cell dataset needs about 3.5 GB." "$out"

go "macOS Docker 7.7 GB" docker Darwin 20 "$(gb 7.7)"
check "macOS Docker 7.7 GB: saved 7" "SCVIEW_MEMORY_GB=7" "$(cat "$work/.env")"
contains "Docker Desktop advice" "open Docker Desktop, Settings > Resources, set Memory to" "$out"
lacks "no under-4 warning at 7" "Under 4 GB" "$out"

go "Linux Docker Engine" docker Linux 20 "$(gb 31.2)"
check "Linux Docker Engine, big machine: saved 17" "SCVIEW_MEMORY_GB=17" "$(cat "$work/.env")"
lacks "no engine message when it is not the limit" "has only" "$out"

go "Linux Docker Engine, small" docker Linux 3.3 "$(gb 3.8)"
check "Linux Docker Engine on a 3.8 GB computer: saved 3" "SCVIEW_MEMORY_GB=3" "$(cat "$work/.env")"
contains "says the limit" "Docker has only 3.8 GB of memory, so scView can use at most 3 GB" "$out"
lacks "no Desktop advice for Docker Engine" "Docker Desktop" "$out"

go "Linux Docker Desktop" docker Linux 20 "$(gb 7.7)" "SCVIEW_MEMORY_GB=" "Docker Desktop"
contains "Docker Desktop on Linux gets the advice" "open Docker Desktop, Settings > Resources" "$out"

go "engine info fails" podman Linux 3.3 ""
check "engine info fails: the 4 GB floor" "SCVIEW_MEMORY_GB=4" "$(cat "$work/.env")"

go "existing 4 GB" podman Darwin 3.3 "$(gb 3.8)" "SCVIEW_MEMORY_GB=4"
check "an existing value is kept" "SCVIEW_MEMORY_GB=4" "$(cat "$work/.env")"
contains "an existing value above the machine is warned about" "Set SCVIEW_MEMORY_GB in .env to 3 or less" "$out"
contains "with how to give it more" "podman machine set --memory 5120" "$out"

go "existing 6 GB, 8 GB machine" podman Darwin 3.3 "$(gb 8)" "SCVIEW_MEMORY_GB=6"
lacks "an existing value that fits: no warning" "has only" "$out"

go "existing 6 GB, engine not known" podman Darwin 3.3 "" "SCVIEW_MEMORY_GB=6"
contains "existing value, engine not known" "[ok] Memory limit: 6 GB" "$out"

go "an older .env without the line" docker Linux 20 "$(gb 31.2)" "DEEPINFRA_API_KEY="
check "it gets the line" "DEEPINFRA_API_KEY=,SCVIEW_MEMORY_GB=17" "$(paste -sd, "$work/.env")"

echo
if [ "$fails" -gt 0 ]; then echo "$fails failed"; exit 1; fi
echo "all passed"
