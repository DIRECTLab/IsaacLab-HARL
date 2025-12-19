"""This file maps the environment name to the starting and trained policy names on huggingface
"""

HF_REPO_ID = "isaacwilliam4/isaaclab-harl-dataset"

HF_POLICY_MAP = {
    "Leatherback-Stage1-Soccer-v0": {
        "starting": None,
        "trained": "adversarial_policies/policy_for_sa_score_goal_leatherback"
    },
    "Leatherback-Stage2-Soccer-v0": {
        "starting": "adversarial_policies/stage_2_leatherback_start_policy_soccer",
        "trained": "adversarial_policies/trained_vs_trained_soccer_leatherback"
    },
    "AnymalC_Soccer_Hetero_By_Team-v0": {
        "starting": "adversarial_policies/anymals_vs_leatherback_start_policy_soccer",
        "trained": None
    },
    "Sumo-Stage2-Hetero-By-Team-v0": {
        "starting": "adversarial_policies/hetero_by_team_start_policies_sumo",
        "trained": None
    },
    "Sumo-Stage2-Hetero-v0": {
        "starting": "adversarial_policies/hetero_within_team_start_policies_sumo",
        "trained": None
    },
    "Minitank-Adversarial-Direct-v0": {
        "starting": "adversarial_policies/3dg_model",
        "trained": None
    },
    "Isaac-Multi-Agent-Flat-Sumo-Stage1-Blocks-Push-v0": {
        "starting": "adversarial_policies/anymal_c_walk_to_point_policy",
        "trained": None
    },
    "AnymalC_Soccer_Stage1-v0": {
        "starting": None,
        "trained": "adversarial_policies/anymal_c_go_to_ball"
    },
    "AnymalC_Soccer_Stage2-v0": {
        "starting": "adversarial_policies/anymal_c_go_to_ball",
        "trained": None
    },
    "Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0": {
        "starting": "adversarial_policies/anymal_c_velocity_model",
        "trained": "adversarial_policies/bar_carrying_trained"
    },
}