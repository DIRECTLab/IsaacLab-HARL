#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main(args):
    checkpoints_path = Path(args.checkpoint_path)
    checkpoint_1_path = checkpoints_path / "episode_1"

    # outdir logic: use argument if provided, otherwise default
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = checkpoints_path.parent.parent / "results_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    learned_agent_names = list(args.learned_agents_names.split(','))
    unlearned_agent_names = list(args.unlearned_agents_names.split(','))
    num_envs, num_steps_per_episode = args.num_envs_episode_length.split(',')

    results = {}

    try:
        for curr_folder in sorted(
            os.listdir(checkpoints_path),
            key=lambda x: int(x.split("_")[-1]) if "episode" in x else -1,
        ):
            if "episode" in curr_folder:
                episode_num = int(curr_folder.split("_")[-1])
                num_steps = episode_num * int(num_steps_per_episode) * int(num_envs)
                results.setdefault("episode_num", []).append(episode_num)
                results.setdefault("num_steps", []).append(num_steps)
                curr_checkpoint_path = checkpoints_path / curr_folder
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
                    args.eval_script,
                    "--num_envs",
                    str(args.num_envs),
                    "--algo",
                    args.algo,
                    "--task",
                    args.task,
                    "--seed",
                    str(args.seed),
                    "--num_env_steps",
                    str(args.num_env_steps),
                    "--dir",
                    str(new_checkpoint_path),
                    "--save_path",
                    str(npz_save_path),
                ]
                if args.debug:
                    cmd.append("--debug")
                if args.headless:
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
    if "episode_num" in results:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate adversarial checkpoints")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoints directory")
    parser.add_argument("--learned_agents_names", type=str, required=True, help="Agents to evaluate adversarial learning, separated by commas i.e 'agent_1_name,agent_2_name'")
    parser.add_argument("--unlearned_agents_names", type=str, required=True, help="Agents to evaluate against, separated by commas 'agent_1_name,agent_2_name'")
    parser.add_argument("--num_envs_episode_length", type=str, required=True, help="For calculating the number of steps given the episodes, this parameters should be" \
    "formatted as 'num_envs,episode_length'")
    parser.add_argument("--eval_script", type=str, default="./get_adversarial_results.py", help="Evaluation script path")

    parser.add_argument("--outdir", type=str, help="Output directory for results and plots")

    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments")
    parser.add_argument("--algo", type=str, default="happo_adv", help="Algorithm name")
    parser.add_argument("--task", type=str, default="leatherback-Sumo-Direct-MA-Stage2-v0", help="Task name")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_env_steps", type=int, default=3_100_000, help="Number of environment steps")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--headless", action="store_true", help="Run headless")

    args = parser.parse_args()
    main(args)
