#!/usr/bin/env bash
# Reproduces the experiments reported in the paper: iris + breast_cancer,
# IID and Dirichlet non-IID partitions (alpha in {0.5, 0.05}), 128-bit
# security. Thin wrapper around config/run_campaign.sh's defaults so this
# entry point can't drift from what the paper actually ran.
#
# For a fast, non-secure pipeline smoke test instead, see the README.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$REPO_ROOT/config/run_campaign.sh" "$@"
