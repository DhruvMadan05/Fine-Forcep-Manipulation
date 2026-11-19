#!/usr/bin/env bash
# Source this script (`source scripts/env.sh`) to load .env and source the
# ROS 2 + AMBF workspace it points at. Do not execute it directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
else
    echo "WARNING: no .env found. Copy .env.example to .env and set AMBF_WS." >&2
fi

if [ -z "$AMBF_WS" ]; then
    echo "ERROR: AMBF_WS is not set (expected in .env or the environment)." >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$AMBF_WS/install/setup.bash" ]; then
    echo "ERROR: '$AMBF_WS/install/setup.bash' not found. Check AMBF_WS and that the workspace is built." >&2
    return 1 2>/dev/null || exit 1
fi

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    source "/opt/ros/$ROS_DISTRO/setup.bash"
fi

source "$AMBF_WS/install/setup.bash"

export AMBF_WS
export REPO_ROOT
echo "Environment ready: AMBF_WS=$AMBF_WS, ROS_DISTRO=$ROS_DISTRO"
