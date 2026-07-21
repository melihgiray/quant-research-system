#!/usr/bin/env bash
# One command that reproduces every number this project claims about itself.
#
#   ./scripts/verify.sh
#
# Prints the test count and line coverage. Any figure quoted in the README or on
# a CV should come from here, not from memory. Exits non-zero if the suite fails,
# so it is safe to wire into CI.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="python3"
fi

echo "=============================================="
echo " quant-research-system verification"
echo " $($PY --version 2>&1)"
echo "=============================================="
echo

$PY -m pytest -q \
    --cov=quant_system \
    --cov-report=term-missing:skip-covered \
    --cov-report=json:.coverage.json \
    2>&1 | tail -40

echo
echo "----------------------------------------------"
$PY - <<'PY'
import json
import pathlib

data = json.loads(pathlib.Path(".coverage.json").read_text())
totals = data["totals"]
covered = totals["covered_lines"]
total = totals["num_statements"]
print(f" Line coverage : {totals['percent_covered']:.1f}%  ({covered}/{total} statements)")

worst = sorted(
    ((v["summary"]["percent_covered"], k) for k, v in data["files"].items()),
)[:5]
print(" Least covered modules:")
for pct, name in worst:
    print(f"   {pct:5.1f}%  {name}")
PY
echo "----------------------------------------------"
