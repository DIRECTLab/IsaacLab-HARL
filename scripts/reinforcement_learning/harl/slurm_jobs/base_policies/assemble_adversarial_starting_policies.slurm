#!/bin/bash

#SBATCH --job-name=assemble-adversarial-starting-policies
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=./logs/%x-%j.log
#SBATCH --error=./logs/%x-%j.log

set -euo pipefail

# Build the starting-policy directories used by ../adversarial/*.slurm from the trained base policies.
# Each adversarial agent slot gets the robot_0 actor of the matching base policy.

rm -rf ./starting_policies/minitank_adversarial \
    ./starting_policies/sumo_hetero_by_team \
    ./starting_policies/soccer_anymal_vs_leatherback

# Minitank-Adversarial-Direct-v0: robot_0 = tank, robot_1 = drone
./stage_policy.sh ./starting_policies/minitank_adversarial \
    Minitank-Direct-v0 happo base_minitank_point \
    robot_0:robot_0
./stage_policy.sh ./starting_policies/minitank_adversarial \
    Isaac-Quadcopter-Direct-HARL-v0 happo base_drone_go_to_point \
    robot_0:robot_1

# Sumo-Stage2-Hetero-By-Team-v0 (and Same-Critic variants): robot_0/1 = anymal, robot_2/3 = leatherback
./stage_policy.sh ./starting_policies/sumo_hetero_by_team \
    Anymal-C-Sumo-Stage1-Blocks-Push-v0 happo_adv base_anymal_push_blocks \
    robot_0:robot_0 robot_0:robot_1
./stage_policy.sh ./starting_policies/sumo_hetero_by_team \
    leatherback-Sumo-Direct-MA-Stage1-v0 happo_adv base_leatherback_push_blocks \
    robot_0:robot_2 robot_0:robot_3

# AnymalC-VS-Leatherback-Soccer-v0: anymal_0 vs leatherback_0
./stage_policy.sh ./starting_policies/soccer_anymal_vs_leatherback \
    AnymalC_Soccer_Score_Goals_Stage_2 happo base_anymal_score_goals \
    robot_0:anymal_0
./stage_policy.sh ./starting_policies/soccer_anymal_vs_leatherback \
    Leatherback-Stage1-Soccer-v0 happo base_leatherback_score_goals \
    robot_0:leatherback_0
