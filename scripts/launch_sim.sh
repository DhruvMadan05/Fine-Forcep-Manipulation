#!/usr/bin/env bash
# Regenerate the per-arm ADFs and launch ambf_simulator with psm1, psm2,
# mtm1, mtm2 plus the Task 1 scene objects (needle L-bracket) loaded. Any
# extra args are passed through to ambf_simulator.
#
#   --no-scene   load only the arms (e.g. for kinematic calibration, so the
#                tool can't collide with the bracket). Must be the first arg.
set -e

LOAD_SCENE=1
if [ "${1:-}" = "--no-scene" ]; then
    LOAD_SCENE=0
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

ARM_ADF_PATHS="$(python3 "$SCRIPT_DIR/generate_adfs.py")"

# Static scene ADFs committed in the repo. Resolved from REPO_ROOT at runtime
# so this works from any clone location.
SCENE_ADFS=(
    "$REPO_ROOT/adf/task1/needle_bracket.yaml"
)
for adf in "${SCENE_ADFS[@]}"; do
    if [ ! -f "$adf" ]; then
        echo "ERROR: scene ADF not found: $adf" >&2
        exit 1
    fi
done
SCENE_ADF_PATHS="$(IFS=,; echo "${SCENE_ADFS[*]}")"

if [ "$LOAD_SCENE" = 1 ]; then
    ADF_PATHS="$ARM_ADF_PATHS,$SCENE_ADF_PATHS"
else
    ADF_PATHS="$ARM_ADF_PATHS"
fi

echo "Launching ambf_simulator with: $ADF_PATHS"
exec ambf_simulator -a "$ADF_PATHS" "$@"
