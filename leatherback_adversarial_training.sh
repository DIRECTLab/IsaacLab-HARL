eval "$(conda shell.bash hook)"

conda activate env_isaaclab

base_policy="$(readlink -f "./source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/start_policy_for_ma_leatherback_sumo_stage2")"
cd ./scripts/reinforcement_learning/harl

python train.py --num_envs 5000 \
 --task "leatherback-Sumo-Direct-MA-Stage2-v0" --seed 1 --save_interval 10000 \
 --log_interval 1 --exp_name "leatherback_ladder_training" --num_env_steps 1_000_000_000 \
 --algorithm happo_adv --headless --dir "$base_policy" --adversarial_training_mode ladder \
 --save_checkpoints --checkpoint_interval 30 --adversarial_training_iterations 100_000_000

python train.py --num_envs 5000 \
 --task "leatherback-Sumo-Direct-MA-Stage2-v0" --seed 1 --save_interval 10000 \
 --log_interval 1 --exp_name "leatherback_parallel_training" --num_env_steps 1_000_000_000 \
 --algorithm happo_adv --headless --dir "$base_policy" --adversarial_training_mode parallel \
 --save_checkpoints --checkpoint_interval 30 --adversarial_training_iterations 100_000_000

 python train.py --num_envs 5000 \
 --task "leatherback-Sumo-Direct-MA-Stage2-v0" --seed 1 --save_interval 10000 \
 --log_interval 1 --exp_name "leatherback_leapfrog_training" --num_env_steps 1_000_000_000 \
 --algorithm happo_adv --headless --dir "$base_policy" --adversarial_training_mode leapfrog \
 --save_checkpoints --checkpoint_interval 30 --adversarial_training_iterations 100_000_000