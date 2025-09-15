eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# echo -e "\e[32mRunning: anymal_leatherback_vs_anymal_leatherback...\e[0m"
# python ./scripts/reinforcement_learning/harl/paper_results.py \
#   --checkpoint_path ./results/sumo_training_runs/anymal_leatherback_vs_anymal_leatherback/models/checkpoints \
#   --outdir ./results/adversarial_paper_results/anymal_leatherback_vs_anymal_leatherback \
#   --learned_agents_names robot_0,robot_2 \
#   --unlearned_agents_names robot_1,robot_3 \
#   --eval_script ./scripts/reinforcement_learning/harl/get_adversarial_results.py \
#   --algo happo_adv \
#   --num_envs 1_000 \
#   --num_envs_episode_length 4000,200 \
#   --task Sumo-Stage2-Hetero-v0 \
#   --seed 1 \
#   --num_env_steps 31_000_000 \
#   --headless 


echo -e "\e[32mRunning: trained_anymals_vs_untrained_leatherbacks...\e[0m"
python ./scripts/reinforcement_learning/harl/paper_results.py \
  --checkpoint_path ./results/sumo_training_runs/anymals_vs_leatherbacks/models/checkpoints \
  --outdir ./results/adversarial_paper_results/trained_anymals_vs_untrained_leatherbacks \
  --learned_agents_names robot_0,robot_1 \
  --unlearned_agents_names robot_2,robot_3 \
  --eval_script ./scripts/reinforcement_learning/harl/get_adversarial_results.py \
  --algo happo_adv \
  --num_envs 1_000 \
  --num_envs_episode_length 4000,200 \
  --task Sumo-Stage2-Hetero-By-Team-v0 \
  --seed 1 \
  --num_env_steps 31_000_000 \
  --headless

echo -e "\e[32mRunning: trained_leatherbacks_vs_untrained_anymals...\e[0m"
python ./scripts/reinforcement_learning/harl/paper_results.py \
  --checkpoint_path ./results/sumo_training_runs/anymals_vs_leatherbacks/models/checkpoints \
  --outdir ./results/adversarial_paper_results/trained_leatherbacks_vs_untrained_anymals \
  --learned_agents_names robot_2,robot_3 \
  --unlearned_agents_names robot_0,robot_1 \
  --eval_script ./scripts/reinforcement_learning/harl/get_adversarial_results.py \
  --algo happo_adv \
  --num_envs 1_000 \
  --num_envs_episode_length 4000,200 \
  --task Sumo-Stage2-Hetero-By-Team-v0 \
  --seed 1 \
  --num_env_steps 31_000_000 \
  --headless
