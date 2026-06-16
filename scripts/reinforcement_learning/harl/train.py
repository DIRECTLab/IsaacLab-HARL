# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with HARL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys


from isaaclab.app import AppLauncher

# local imports
import cli_args # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment")
# append HARL cli arguments
cli_args.add_harl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from harl.runners import RUNNER_REGISTRY

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from huggingface_hub import snapshot_download
from harl.utils.hf_policies import HF_POLICY_MAP, HF_REPO_ID

import time

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"harl_{algorithm}_cfg_entry_point"


def _configure_model_dir(args: dict, algo_args: dict) -> None:
    """Apply HF/local policy loading rules and set algo_args['train']['model_dir']."""
    task_name = args.get("task")
    provided_dir = bool(args.get("dir"))
    load_trained = bool(args.get("load_trained_policy"))
    load_starting = bool(args.get("load_starting_policy"))

    # mutual exclusivity:
    if load_trained and load_starting:
        raise ValueError("Cannot set both --load_trained_policy and --load_starting_policy.")
    if provided_dir and (load_trained or load_starting):
        raise ValueError("Cannot combine --dir with --load_trained_policy/--load_starting_policy.")

    # default: use --dir (if provided)
    if provided_dir:
        algo_args["train"]["model_dir"] = args["dir"]
        return

    # HF load
    if load_trained:
        entry = HF_POLICY_MAP.get(task_name, {})
        policy_location = entry.get("trained")
        if not policy_location:
            print("Sorry, a trained policy doesn't exist for this env.")
            return

        base = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision="main",
            allow_patterns=[f"{policy_location}/**"],
        )
        algo_args["train"]["model_dir"] = os.path.join(base, policy_location)
        return

    if load_starting:
        entry = HF_POLICY_MAP.get(task_name, {})
        policy_location = entry.get("starting")
        if not policy_location:
            print("Sorry, a starting policy doesn't exist for this env.")
            return

        base = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision="main",
            allow_patterns=[f"{policy_location}/**"],
        )
        algo_args["train"]["model_dir"] = os.path.join(base, policy_location)
        return

    # If nothing specified, leave whatever hydra config provides (or None).
    # This mirrors the old behavior where user might rely on the default model_dir in config.
    return


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    args = args_cli.__dict__

    args["env"] = "isaaclab"

    args["algo"] = args["algorithm"]

    algo_args = agent_cfg

    algo_args["eval"]["use_eval"] = False
    algo_args["train"]["n_rollout_threads"] = args["num_envs"]
    algo_args["train"]["num_env_steps"] = args["num_env_steps"]
    algo_args["train"]["eval_interval"] = args["save_interval"]
    algo_args["train"]["save_checkpoints"] = args["save_checkpoints"]
    algo_args["train"]["checkpoint_interval"] = args["checkpoint_interval"]
    algo_args["train"]["log_interval"] = args["log_interval"]
    algo_args["train"]["model_dir"] = args["dir"]
    algo_args["seed"]["specify_seed"] = True
    algo_args["seed"]["seed"] = args["seed"]
    algo_args["algo"]["adversarial_training_mode"] = args["adversarial_training_mode"]
    algo_args["algo"]["adversarial_training_iterations"] = args["adversarial_training_iterations"]

    algo_args.setdefault("train", {})
    _configure_model_dir(args, algo_args)

    env_args = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    env_args["video_settings"] = {
        "video": bool(args["video"]),
        "video_length": args["video_length"],
        "video_interval": args["video_interval"],
        "log_dir": os.path.join(
            algo_args["logger"]["log_dir"],
            "isaaclab",
            args["task"],
            args["algorithm"],
            args["exp_name"],
            "-".join(["seed-{:0>5}".format(agent_cfg["seed"]["seed"]), hms_time]),
            "videos",
        ),
    }

    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]

    # create runner

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()  # type: ignore
    simulation_app.close()
