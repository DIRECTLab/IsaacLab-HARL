#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def extract_config_values(config_path):
    """Extract num_envs, episode_length, seed, algo, task, and num_env_steps from configs.json file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        extracted = {}
        
        # Extract from Args section
        args = config_data.get('Args', {})
        
        if 'num_envs' in args:
            extracted['num_envs'] = args['num_envs']
        
        if 'seed' in args:
            extracted['seed'] = args['seed']
        
        if 'algo' in args:
            extracted['algo'] = args['algo']
        
        if 'task' in args:
            extracted['task'] = args['task']
        
        if 'num_env_steps' in args:
            extracted['num_env_steps'] = args['num_env_steps']
        
        # Extract episode_length from Algo Args -> train section
        algo_args = config_data.get('Algo Args', {})
        train_config = algo_args.get('train', {})
        if 'episode_length' in train_config:
            extracted['episode_length'] = train_config['episode_length']
        
        return extracted
    except Exception as e:
        print(f"Warning: Could not extract values from config: {e}")
    
    return {}


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    checkpoint_parent_path = Path(config['checkpoint_path'])
    checkpoints_path = checkpoint_parent_path / "models" / "checkpoints"
    checkpoint_1_path = checkpoints_path / "episode_1"

    # outdir logic: use argument if provided, otherwise default
    if config.get('outdir'):
        outdir = Path(config['outdir'])
    else:
        outdir = checkpoint_parent_path.parent.parent / "results_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    learned_agent_names = list(config['learned_agents_names'].split(','))
    unlearned_agent_names = list(config['unlearned_agents_names'].split(','))
    
    # Extract num_envs and episode_length from configs.json
    config_json_path = checkpoint_parent_path / "configs.json"
    if not config_json_path.exists():
        raise FileNotFoundError(f"configs.json not found at {config_json_path}")
    
    extracted_config = extract_config_values(config_json_path)
    if not extracted_config or 'num_envs' not in extracted_config or 'episode_length' not in extracted_config:
        raise ValueError("Failed to extract required num_envs and episode_length from configs.json")
    
    num_envs = extracted_config['num_envs']
    num_steps_per_episode = extracted_config['episode_length']
    
    # Use extracted values or fallback to YAML config
    seed = extracted_config.get('seed', config.get('seed', 1))
    algo = extracted_config.get('algo', config.get('algo', 'happo_adv'))
    task = extracted_config.get('task', config.get('task'))
    num_env_steps = config.get('num_env_steps')
    
    print(f"Extracted from configs.json: num_envs={num_envs}, episode_length={num_steps_per_episode}")
    print(f"Using seed={seed}, algo={algo}, task={task}, num_env_steps={num_env_steps}")
    
    results = {}
    
    # Get all available episodes
    available_episodes = sorted([
        int(folder.split("_")[-1]) 
        for folder in os.listdir(checkpoints_path) 
        if "episode" in folder
    ])
    
    # Determine which checkpoints to evaluate
    if config.get('target_step'):
        # Calculate target episode number from target step
        target_episode = config['target_step'] // (num_envs * num_steps_per_episode)
        if config['target_step'] % (num_envs * num_steps_per_episode) != 0:
            target_episode += 1  # Round up if not exact
        
        print(f"Target step: {config['target_step']}")
        print(f"Calculated target episode: {target_episode}")
        
        # Find the closest episode if exact match doesn't exist
        if target_episode not in available_episodes:
            closest_episode = min(available_episodes, key=lambda x: abs(x - target_episode))
            print(f"Episode {target_episode} not found. Using closest episode: {closest_episode}")
            episodes_to_evaluate = [closest_episode]
        else:
            episodes_to_evaluate = [target_episode]
    else:
        # Evaluate all episodes
        episodes_to_evaluate = available_episodes

    try:
        for episode_num in episodes_to_evaluate:
            curr_folder = f"episode_{episode_num}"
            curr_checkpoint_path = checkpoints_path / curr_folder
            
            if not curr_checkpoint_path.exists():
                print(f"Warning: Checkpoint folder {curr_folder} not found, skipping...")
                continue
            
            num_steps = episode_num * num_steps_per_episode * num_envs
            results.setdefault("episode_num", []).append(episode_num)
            results.setdefault("num_steps", []).append(num_steps)
            
            new_checkpoint_path = checkpoints_path / "curr_checkpoint"
            os.makedirs(new_checkpoint_path, exist_ok=True)

            # learned agents from current checkpoint
            for learned_agent in learned_agent_names:
                shutil.copy(
                    curr_checkpoint_path / f"actor_agent_{learned_agent}.pt",
                    new_checkpoint_path / f"actor_agent_{learned_agent}.pt",
                )

            # unlearned agents fixed from episode_1
            for unlearned_agent in unlearned_agent_names:
                shutil.copy(
                    checkpoint_1_path / f"actor_agent_{unlearned_agent}.pt",
                    new_checkpoint_path / f"actor_agent_{unlearned_agent}.pt",
                )

            npz_save_path = outdir / f"ep{episode_num}.npz"

            cmd = [
                "python",
                config['eval_script'],
                "--num_envs",
                str(config.get('num_envs', num_envs)),
                "--algo",
                algo,
                "--task",
                task,
                "--seed",
                str(seed),
                "--num_env_steps",
                str(num_env_steps),
                "--dir",
                str(new_checkpoint_path),
                "--save_path",
                str(npz_save_path),
            ]
            if config.get('debug'):
                cmd.append("--debug")
            if config.get('headless'):
                cmd.append("--headless")

            subprocess.run(cmd, check=True)

            res = np.load(npz_save_path)
            for k, v in res.items():
                results.setdefault(k, []).append(v.item())

            os.remove(npz_save_path)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if "new_checkpoint_path" in locals() and os.path.exists(new_checkpoint_path):
            shutil.rmtree(new_checkpoint_path)

    # ---- Save collected results ----
    all_results_path = outdir / "all_results.npz"
    np.savez(all_results_path, **results)
    print(f"Saved results to {all_results_path}")

    # ---- Plot results ----
    if "episode_num" in results and len(results["episode_num"]) > 1:
        plt.figure(figsize=(14, 8))  # bigger figure
        episodes = results["episode_num"]

        for metric, values in results.items():
            if metric == "episode_num":
                continue
            plt.plot(episodes, values, marker="o", label=metric)

        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Metric Value", fontsize=14)
        plt.title("Adversarial Training Results", fontsize=16)
        plt.grid(True)

        # legend outside
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize=12
        )

        plt.tight_layout(rect=[0, 0, 0.8, 1])

        plot_path = outdir / "training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {plot_path}")
    elif config.get('target_step'):
        print("Skipping plot generation for single checkpoint evaluation")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate adversarial checkpoints from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)
