#!/bin/bash
# Submit all HARL slurm training jobs in this directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

for job in *.slurm; do
    echo "Submitting $job"
    sbatch "$job"
done
