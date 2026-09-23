#!/bin/bash
# Copy actor weights from the newest base-policy run into a starting-policy directory.
#
# Usage:
#   ./stage_policy.sh <dest_dir> <task> <algorithm> <exp_name> <src_robot>:<dst_robot> [<src_robot>:<dst_robot> ...]
#
# The newest run under results/isaaclab/<task>/<algorithm>/<exp_name>/seed-* is used, and each
# models/actor_agent_<src_robot>.pt is copied to <dest_dir>/actor_agent_<dst_robot>.pt. Only actors
# are copied (matching the HuggingFace starting policies), so critics from other stages are never loaded.
# Multiple calls can fill the same <dest_dir>, e.g. to build heterogeneous teams.

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <dest_dir> <task> <algorithm> <exp_name> <src_robot>:<dst_robot> [...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dest_dir=$1
task=$2
algorithm=$3
exp_name=$4
shift 4

exp_dir="$SCRIPT_DIR/results/isaaclab/$task/$algorithm/$exp_name"
# seed-<seed>-<YYYY-MM-DD-HH-MM-SS> sorts chronologically for a fixed seed
run_dir=$(ls -d "$exp_dir"/seed-* 2>/dev/null | sort | tail -n 1 || true)
if [ -z "$run_dir" ]; then
    echo "ERROR: no runs found in $exp_dir" >&2
    exit 1
fi

models_dir="$run_dir/models"
mkdir -p "$dest_dir"

for mapping in "$@"; do
    src=${mapping%%:*}
    dst=${mapping##*:}
    src_file="$models_dir/actor_agent_${src}.pt"
    if [ ! -f "$src_file" ]; then
        echo "ERROR: missing actor $src_file" >&2
        exit 1
    fi
    cp "$src_file" "$dest_dir/actor_agent_${dst}.pt"
    echo "Staged $src_file -> $dest_dir/actor_agent_${dst}.pt"
done
