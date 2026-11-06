#!/usr/bin/env bash
# Checks the PowerShell expression start-windows.bat suggests SCVIEW_MEMORY_GB with (its
# "for /f ... powershell -Command" line), with free memory and the engine's bytes stood in.
# Needs pwsh, or Docker to run it in mcr.microsoft.com/powershell; skipped without either.
# The batch lines around it (for /f tokens, if/goto) are not run here: only on Windows.
# Run: bash scripts/test_start_windows_memory.sh
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
expr=$(grep -o 'powershell -NoProfile -Command "[^"]*FreePhysicalMemory[^"]*"' "$here/start-windows.bat" | head -1)
expr=${expr#powershell -NoProfile -Command \"}
expr=${expr%\"}
[[ "$expr" == *'(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'* && "$expr" == *'%ENGINE_BYTES%'* ]] \
    || { echo "FAIL: the memory expression in start-windows.bat changed shape"; exit 1; }
expr=${expr//'(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'/'$FREEKB'}

# name|free GB|engine bytes|expected "free engine suggested wanted cap"
cases=(
    "VM finding: 3.3 GB free, Podman machine 3.8 GB|3.3|$((3891 * 1048576))|3.3 3.8 3 4 3"
    "engine not known: the 4 GB floor|3.3|0|3.3 0.0 4 4 0"
    "floor kept when the engine has room for it|3.3|$((4608 * 1048576))|3.3 4.5 4 4 4"
    "Docker Desktop 7.7 GB, 20 GB free|20|$((7885 * 1048576))|20.0 7.7 7 17 7"
    "big engine: free - 3|20|$((31949 * 1048576))|20.0 31.2 17 17 30"
    "tiny engine: at least 1|3.3|$((1229 * 1048576))|3.3 1.2 1 4 1"
)

script=""
for c in "${cases[@]}"; do
    IFS='|' read -r _ free bytes _ <<< "$c"
    kb=$(awk -v g="$free" 'BEGIN {printf "%d", g * 1048576}')
    script+="\$FREEKB=$kb; ${expr//'%ENGINE_BYTES%'/$bytes}"$'\n'
done

if command -v pwsh >/dev/null; then
    got=$(pwsh -NoProfile -Command "$script")
elif command -v docker >/dev/null && docker image inspect mcr.microsoft.com/powershell:lts-7.4-ubuntu-22.04 >/dev/null 2>&1; then
    got=$(docker run --rm --cpus 1 mcr.microsoft.com/powershell:lts-7.4-ubuntu-22.04 pwsh -NoProfile -Command "$script")
else
    echo "skipped: no pwsh, and no mcr.microsoft.com/powershell:lts-7.4-ubuntu-22.04 image"
    exit 0
fi

mapfile -t lines <<< "$got"
fails=0
for i in "${!cases[@]}"; do
    IFS='|' read -r name _ _ want <<< "${cases[$i]}"
    line=${lines[$i]:-}
    line=${line%$'\r'}
    if [ "$line" = "$want" ]; then echo "ok    $name"; else echo "FAIL  $name: expected [$want], got [$line]"; fails=$((fails + 1)); fi
done
echo
if [ "$fails" -gt 0 ]; then echo "$fails failed"; exit 1; fi
echo "all passed"
