# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing utilities for common operations and helper functions."""

from .array import *
from .buffers import *
from .configclass import configclass
from .dict import *
from .interpolation import *
from .modifiers import *
from .string import *
from .timer import Timer
from .types import *

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
    "Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0": {
        "starting": "adversarial_policies/anymal_c_velocity_model",
        "trained": "adversarial_policies/bar_carrying_trained",
    },
    "Anymal-C-Sumo-Stage1-Blocks-Push-v0": {
        "starting": "adversarial_policies/anymal_c_go_to_point_sumo",
        "trained": "adversarial_policies/anymal_push_blocks",
    },
    "AnymalC_Soccer_Go_To_Point_Stage_0": {"starting": None, "trained": "adversarial_policies/anymal_c_go_to_point_soccer"},
    "AnymalC_Soccer_Go_To_Ball_Stage_1": {
        "starting": "adversarial_policies/anymal_c_go_to_point_soccer",
        "trained": "adversarial_policies/anymal_c_go_to_ball",
    },
    "AnymalC_Soccer_Score_Goals_Stage_2": {
        "starting": "adversarial_policies/anymal_c_go_to_ball",
        "trained": "adversarial_policies/score_goals_policy",
    },
}

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