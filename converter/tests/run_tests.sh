#!/bin/sh
# Converter tests, run inside the converter image with this directory's parent
# mounted at /app:
#   docker run --rm --cpus 2 -v "$PWD/converter:/app" -w /app \
#     scview-converter sh tests/run_tests.sh
# Builds small R objects, converts them with convert.R / convert_sce.R, and
# checks the h5ad files from Python.
set -e
OUT=${1:-/tmp/scview-converter-tests}
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$OUT"
mkdir -p "$OUT"
Rscript "$HERE/make_fixtures.R" "$OUT"
# a few cells per block, so the dense assays are written in many blocks
SCVIEW_CONVERT_BLOCK_VALUES=100 Rscript "$HERE/convert_fixtures.R" "$OUT"
python3 "$HERE/check_h5ad.py" "$OUT"
