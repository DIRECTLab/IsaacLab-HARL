from __future__ import annotations

import argparse
from harl.utils.hf_policies import HF_POLICY_MAP, policies_summary

def add_harl_args(parser: argparse.ArgumentParser):
    """Add HARL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """

    # add the policy map as the epilogue to the parser help message
    parser.epilog = policies_summary(HF_POLICY_MAP)

    arg_group = parser.add_argument_group("HARL", description="Arguments for the HARL agent(s)")


    arg_group.add_argument("--exp_name", type=str, default="test", help="Name of the Experiment")
    parser.add_argument("--save_interval", type=int, default=None, help="How often to save the model")
    parser.add_argument("--save_checkpoints", action="store_true", default=False, help="Whether or not to save checkpoints")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=200,
        help="How often to save a model checkpoint (episodes, episodes = num_envs*episode_length steps)",
    )
    parser.add_argument("--log_interval", type=int, default=None, help="How often to log outputs")
    parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument("--dir", type=str, default=None, help="folder with trained models")
    parser.add_argument("--debug", action="store_true", help="whether to run in debug mode for visualization")


    arg_group.add_argument(
        "--adversarial_training_mode",
        default="parallel",
        choices=["parallel", "ladder", "leapfrog"],
        help=(
            "the mode type for adversarial training,                     note on ladder training with teams that are"
            " composed of heterogeneous agents, the two teams must place the robots in the same order in their environment "
            "                    for ladder to work"
        ),
    )
    arg_group.add_argument(
        "--adversarial_training_iterations",
        default=50_000_000,
        type=int,
        help="the number of iterations to swap training for adversarial modes like ladder and leapfrog",
    )
    
    arg_group.add_argument(
        "--algorithm",
        type=str,
        default="happo",
        choices=["happo", "hatrpo", "haa2c", "mappo", "mappo_unshare", "happo_adv"],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, mappo, and mappo_unshare.",
    )
    arg_group.add_argument(
        "--load_starting_policy",
        action="store_true",
        help="If set, load the starting policy for this env from HuggingFace (if one exists).",
    )
    arg_group.add_argument(
        "--load_trained_policy",
        action="store_true",
        help="If set, load the trained policy for this env from HuggingFace (if one exists).",
    )


