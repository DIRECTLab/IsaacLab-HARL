#!/bin/bash
# Submit all base-policy training jobs, chained with slurm dependencies:
#
#   anymal_go_to_point_sumo   -> anymal_push_blocks
#   anymal_go_to_point_soccer -> anymal_go_to_ball -> anymal_score_goals
#
# Once the base policies needed by the adversarial tasks finish, an assemble job builds the
# starting-policy directories and every ../adversarial/*.slurm job is submitted to start from them.
#
# Usage:
#   ./run_all_slurm_jobs.sh                   # base policies + adversarial jobs
#   ./run_all_slurm_jobs.sh --no_adversarial  # base policies + assemble only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

submit_adversarial=true
if [ "${1:-}" = "--no_adversarial" ]; then
    submit_adversarial=false
elif [ -n "${1:-}" ]; then
    echo "Unknown argument: $1" >&2
    exit 1
fi

mkdir -p logs starting_policies

# submit <slurm_file> [dependency job ids...]
submit() {
    local job=$1
    shift
    local dep_args=()
    if [ "$#" -gt 0 ]; then
        local deps
        deps=$(IFS=:; echo "$*")
        dep_args=(--dependency="afterok:$deps" --kill-on-invalid-dep=yes)
    fi
    local job_id
    job_id=$(sbatch --parsable "${dep_args[@]}" "$job")
    echo "Submitted $job -> $job_id${*:+ (after $*)}" >&2
    echo "$job_id"
}

# independent base policies
drone=$(submit run_training_drone_go_to_point.slurm)
minitank=$(submit run_training_minitank_point.slurm)
lb_push=$(submit run_training_leatherback_push_blocks.slurm)
lb_score=$(submit run_training_leatherback_score_goals.slurm)
anymal_point_sumo=$(submit run_training_anymal_go_to_point_sumo.slurm)
anymal_point_soccer=$(submit run_training_anymal_go_to_point_soccer.slurm)

# chained base policies
anymal_push=$(submit run_training_anymal_push_blocks.slurm "$anymal_point_sumo")
anymal_ball=$(submit run_training_anymal_go_to_ball.slurm "$anymal_point_soccer")
anymal_score=$(submit run_training_anymal_score_goals.slurm "$anymal_ball")

# starting policies for the adversarial tasks
assemble=$(submit assemble_adversarial_starting_policies.slurm \
    "$drone" "$minitank" "$anymal_push" "$lb_push" "$anymal_score" "$lb_score")

if [ "$submit_adversarial" = true ]; then
    (
        cd ../adversarial
        mkdir -p logs
        for job in *.slurm; do
            submit "$job" "$assemble" > /dev/null
        done
    )
fi
