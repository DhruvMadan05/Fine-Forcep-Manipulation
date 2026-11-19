#!/usr/bin/env bash
# Regenerate the per-arm ADFs and launch ambf_simulator with psm1, psm2,
# mtm1, mtm2 loaded. Any extra args are passed through to ambf_simulator.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

ADF_PATHS="$(python3 "$SCRIPT_DIR/generate_adfs.py")"

echo "Launching ambf_simulator with: $ADF_PATHS"
exec ambf_simulator -a "$ADF_PATHS" "$@"
