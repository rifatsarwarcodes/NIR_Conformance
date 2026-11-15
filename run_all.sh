#!/usr/bin/env bash
# Regenerate fixtures, run every framework in both environments, build the report.
# Environments (see README): .venv-latest (nir 1.0.8) and .venv-pinned (nir 1.0.4,
# needed because sinabs and rockpool[nir] pin nir<=1.0.4).
set -euo pipefail
cd "$(dirname "$0")"
# nirtorch's legacy loader is not deterministic for graphs with unsupported nodes
# (finding NT-3; node sets ordered by object identity). The runner repeats such
# frameworks 30x and marks those cells FLAKY. Fixing the hash seed does not help,
# it only removes one source of variation.
export PYTHONHASHSEED=0
LATEST=.venv-latest/bin/python
PINNED=.venv-pinned/bin/python
$LATEST -m pytest tests -q
$LATEST -m nirconf.fixtures fixtures > /dev/null
rm -rf results/raw results/logs && mkdir -p results/logs
for fw in snntorch norse spyx; do
  $LATEST -m nirconf.run_framework "$fw" fixtures results > "results/logs/$fw-nir1.0.8.log" 2>&1
done
for fw in snntorch norse sinabs rockpool; do
  $PINNED -m nirconf.run_framework "$fw" fixtures results > "results/logs/$fw-nir1.0.4.log" 2>&1
done
$LATEST -m nirconf.compare fixtures results
rm -rf results/verified
$LATEST -m nirconf.verify_findings results > results/logs/verify-nir1.0.8.log 2>&1
$PINNED -m nirconf.verify_findings results > results/logs/verify-nir1.0.4.log 2>&1
grep -hE "^[A-Z]{2,3}-[0-9]" results/logs/verify-*.log | awk '{print $2}' | sort | uniq -c
