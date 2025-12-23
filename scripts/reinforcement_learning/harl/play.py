# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

HF_REPO_ID = "isaacwilliam4/isaaclab-harl-dataset"

HF_POLICY_MAP = {
    "Leatherback-Stage1-Soccer-v0": {
        "starting": None,
        "trained": "adversarial_policies/policy_for_sa_score_goal_leatherback",
    },
    "Leatherback-Stage2-Soccer-v0": {
        "starting": "adversarial_policies/stage_2_leatherback_start_policy_soccer",
        "trained": "adversarial_policies/trained_vs_trained_soccer_leatherback",
    },
    "AnymalC_Soccer_Hetero_By_Team-v0": {
        "starting": "adversarial_policies/anymals_vs_leatherback_start_policy_soccer",
        "trained": None,
    },
    "Sumo-Stage2-Hetero-By-Team-v0": {
        "starting": "adversarial_policies/hetero_by_team_start_policies_sumo",
        "trained": None,
    },
    "Sumo-Stage2-Hetero-v0": {
        "starting": "adversarial_policies/hetero_within_team_start_policies_sumo",
        "trained": None,
    },
    "Minitank-Adversarial-Direct-v0": {"starting": "adversarial_policies/3dg_model", "trained": None},
    "Isaac-Multi-Agent-Flat-Sumo-Stage1-Blocks-Push-v0": {
        "starting": "adversarial_policies/anymal_c_walk_to_point_policy",
        "trained": None,
    },
    "Anymal-C-Go-To-Point-Sumo": {"starting": None, "trained": "adversarial_policies/anymal_c_go_to_point_sumo"},
    "AnymalC_Soccer_Go_To_Point-v1": {"starting": None, "trained": "adversarial_policies/anymal_c_go_to_point_soccer"},
    "AnymalC_Soccer_Stage2-v0": {"starting": "adversarial_policies/anymal_c_go_to_point_soccer", "trained": None},
    "Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0": {
        "starting": "adversarial_policies/anymal_c_velocity_model",
        "trained": "adversarial_policies/bar_carrying_trained",
    },
    "Anymal-C-Sumo-Stage1-Blocks-Push-v0": {
        "starting": "adversarial_policies/anymal_c_go_to_point_sumo",
        "trained": None,
    },
    "AnymalC_Soccer_Score_Goals": {
        "starting": "adversarial_policies/anymal_c_go_to_point_soccer",
        "trained": None,
    },
}

"""Play an algorithm (supports both coordination + adversarial HARL runners)."""

import argparse
import os
import pprint
import sys
import torch
from tqdm import tqdm

from huggingface_hub import snapshot_download

from isaaclab.app import AppLauncher

def policies_summary(policy_map: dict) -> str:
    # Build rows
    rows = []
    for env, info in policy_map.items():
        has_start = info.get("starting") is not None
        has_train = info.get("trained") is not None
        rows.append((env, "YES" if has_start else "NO", "YES" if has_train else "NO"))

    headers = ("Environment", "Starting", "Trained")

    # Compute column widths
    w_env = max(len(headers[0]), *(len(r[0]) for r in rows)) if rows else len(headers[0])
    w_start = max(len(headers[1]), *(len(r[1]) for r in rows)) if rows else len(headers[1])
    w_train = max(len(headers[2]), *(len(r[2]) for r in rows)) if rows else len(headers[2])

    def sep(char="-", cross="+"):
        return (
            f"{cross}{char*(w_env+2)}"
            f"{cross}{char*(w_start+2)}"
            f"{cross}{char*(w_train+2)}{cross}"
        )

    def fmt_row(a, b, c):
        return f"| {a:<{w_env}} | {b:<{w_start}} | {c:<{w_train}} |"

    lines = []
    lines.append("Policy availability by environment:")
    lines.append(sep("-"))
    lines.append(fmt_row(*headers))
    lines.append(sep("="))
    for r in rows:
        lines.append(fmt_row(*r))
    lines.append(sep("-"))

    return "\n" + "\n".join(lines)

parser = argparse.ArgumentParser(description="Play an RL agent with HARL.", formatter_class=argparse.RawTextHelpFormatter, epilog=policies_summary(HF_POLICY_MAP))

parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "haddpg",
        "hatd3",
        "hasac",
        "had3qn",
        "maddpg",
        "matd3",
        "mappo",
        "happo_adv",
    ],
    help="Algorithm name.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_env_steps", type=int, default=None, help="Total environment steps to play.")
parser.add_argument("--dir", type=str, default=None, help="Folder with trained models (local path).")
parser.add_argument("--debug", action="store_true", help="Run in debug mode for visualization.")
parser.add_argument(
    "--load_starting_policy",
    action="store_true",
    help="If set, load the starting policy for this env from HuggingFace (if one exists).",
)
parser.add_argument(
    "--load_trained_policy",
    action="store_true",
    help="If set, load the trained policy for this env from HuggingFace (if one exists).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# --------------------------------------------------------------------------------------
# Launch Omniverse
# --------------------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------------
# Imports that require the app
# --------------------------------------------------------------------------------------

from harl.runners import RUNNER_REGISTRY  # noqa: E402

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"harl_{algorithm}_cfg_entry_point"



def _max_action_dim(action_space) -> int:
    """Recursively find the maximum action dimension across nested dict action spaces."""
    if isinstance(action_space, dict):
        if len(action_space) == 0:
            return 0
        return max(_max_action_dim(v) for v in action_space.values())
    # assume a gymnasium space-like object with .shape
    shape = getattr(action_space, "shape", None)
    if not shape:
        return 0
    return int(shape[0])


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
    return

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    policies_summary(HF_POLICY_MAP)

    args = args_cli.__dict__

    # HARL runner args
    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"

    is_adv = "adv" in str(args["algo"]).lower()

    algo_args = agent_cfg
    algo_args["eval"]["use_eval"] = False
    algo_args["render"]["use_render"] = True

    # apply model_dir logic (HF and/or local dir)
    algo_args.setdefault("train", {})
    _configure_model_dir(args, algo_args)

    # Env config
    env_args: dict = {}
    if args.get("num_envs") is not None:
        env_cfg.scene.num_envs = args["num_envs"]
    else:
        # ensure downstream uses an int
        args["num_envs"] = int(env_cfg.scene.num_envs)

    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["video_settings"] = {"video": False}
    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]

    # create runner
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)

    obs, _, _ = runner.env.reset()

    # determine max action dim (supports nested dict action spaces for adv)
    max_action_space = 0
    for _, sp in runner.env.action_space.items():
        max_action_space = max(max_action_space, _max_action_dim(sp))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    actions = torch.zeros(
        (args["num_envs"], runner.num_agents, max_action_space),
        dtype=torch.float32,
        device=device,
    )
    rnn_states = torch.zeros(
        (args["num_envs"], runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
        dtype=torch.float32,
        device=device,
    )
    masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float32, device=device)

    # logging + progress bar (from adv script)
    log_infos: dict[str, float] = {}
    envs_completed = 0
    pbar = tqdm(total=args["num_env_steps"], desc="Playing")

    while simulation_app.is_running():
        with torch.inference_mode():
            if is_adv:
                # obs: {team: {agent_id: obs_tensor}}
                for team, agents_obs in obs.items():
                    for agent_id, agent_obs in agents_obs.items():
                        agent_num = runner.env.env._agent_map[agent_id]
                        action, _, rnn_state = runner.actors[team][agent_id].get_actions(
                            agent_obs,
                            rnn_states[:, agent_num, :],
                            masks[:, agent_num, :],
                            None,
                            None,
                        )
                        a_dim = action.shape[1]
                        actions[:, agent_num, :a_dim] = action
                        rnn_states[:, agent_num, :] = rnn_state
            else:
                # obs: {agent_name: obs_tensor}  (or similar mapping)
                for agent_num, agent_obs in enumerate(obs.values()):
                    action, _, rnn_state = runner.actor[agent_num].get_actions(
                        agent_obs,
                        rnn_states[:, agent_num, :],
                        masks[:, agent_num, :],
                        None,
                        None,
                    )
                    a_dim = action.shape[1]
                    actions[:, agent_num, :a_dim] = action
                    rnn_states[:, agent_num, :] = rnn_state

        # step env
        obs, _, _, dones, _, _ = runner.env.step(actions)

        # episode completion aggregation
        dones_env = torch.all(dones, dim=1)
        curr_envs_completed = int(dones_env.sum().item())
        envs_completed += curr_envs_completed

        if curr_envs_completed > 0 and hasattr(runner.env, "log_info"):
            for k, v in runner.env.log_info.items():
                if k not in log_infos:
                    log_infos[k] = 0.0
                log_infos[k] += float(v) * curr_envs_completed

        # reset masks/rnn where done
        masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float32, device=device)
        masks[dones_env] = 0.0

        if curr_envs_completed > 0:
            rnn_states[dones_env] = torch.zeros(
                (curr_envs_completed, runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
                dtype=torch.float32,
                device=device,
            )

        # update pbar using total env-steps (matches adv script semantics)
        sim_steps = int(runner.env.unwrapped.sim._number_of_steps)
        num_steps = int(args["num_envs"]) * sim_steps
        pbar.update(max(0, num_steps - pbar.n))

        if num_steps >= int(args["num_env_steps"]):
            break

    runner.env.close()
    pbar.close()

    # print averaged log infos (from adv script)
    if envs_completed <= 0:
        envs_completed = 1
    for k in list(log_infos.keys()):
        log_infos[k] /= envs_completed
    if log_infos:
        pprint.pprint(log_infos)


if __name__ == "__main__":
    main()
    simulation_app.close()
