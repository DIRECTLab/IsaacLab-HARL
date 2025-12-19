"""This file maps the environment name to the starting and trained policy names on huggingface
"""

HF_REPO_ID = "isaacwilliam4/isaaclab-harl-dataset"

HF_POLICY_MAP = {
    "Leatherback-Stage1-Soccer-v0": {
        "starting": None,
        "trained": "adversarial_policies/policy_for_sa_score_goal_leatherback"
    }
}