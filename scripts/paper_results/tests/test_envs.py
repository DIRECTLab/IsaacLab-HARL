# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import subprocess

from isaaclab.utils import HF_POLICY_MAP

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

LOG_FILE = "test_output.log"


def build_args_from_policy_map(env_name: str, policy_info: dict) -> list:
    """Build training arguments from HF_POLICY_MAP entry."""
    args = [
        "--algorithm",
        policy_info["algorithm"],
        "--task",
        env_name,
        "--num_envs",
        "10",
        "--num_env_steps",
        "100_000",  # Short test run
        "--save_interval",
        "100",
        "--log_interval",
        "10",
        "--headless",
    ]

    # Add starting policy if available
    if policy_info.get("starting") is not None:
        args.append("--load_starting_policy")

    return args


def run_config(name, script_path, args, successes, failures):
    command = ["python3", script_path] + args
    print(f"\n{YELLOW}Running: {BOLD}{name}{RESET}", flush=True)
    print(f"{BOLD}Command:{RESET} {' '.join(command)}", flush=True)

    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"\n===== Running: {name} =====\n")
        log_file.write(f"Command: {' '.join(command)}\n")

        try:
            subprocess.run(command, stdout=log_file, stderr=log_file, text=True, check=True)
            success_str = f"SUCCESS, ENV: {name}"
            len_str = len(success_str)
            print(GREEN + BOLD, flush=True)
            print("=" * len_str, flush=True)
            print(success_str, flush=True)
            print("=" * len_str + RESET, flush=True)
            successes.append(name)
        except subprocess.CalledProcessError as e:
            # Also log the error for reference
            log_file.write("\n[ERROR OCCURRED]\n")
            clean_err = ANSI_ESCAPE.sub("", e.stderr if e.stderr else "")
            log_file.write(clean_err + "\n")
            failure_str = f"FAILURE, ENV: {name}"
            len_str = len(failure_str)
            print(RED + BOLD + "=" * len_str, flush=True)
            print(clean_err, flush=True)
            print(failure_str, flush=True)
            print("=" * len_str + RESET, flush=True)
            failures.append((name, clean_err))


def print_summary(successes, failures):
    print(f"\n{BOLD}===== SUMMARY =====", flush=True)
    print(f"{GREEN}Successes: {len(successes)}", flush=True)
    for name in successes:
        print(f"  - {name}", flush=True)

    print(f"\n{RED}Failures: {len(failures)}", flush=True)
    for name, _ in failures:
        print(f"  - {name}", flush=True)
    print(RESET, flush=True)


def main():
    # Clear old log
    with open(LOG_FILE, "w"):
        pass

    script_path = "../../reinforcement_learning/harl/train.py"
    successes = []
    failures = []

    for env_name, policy_info in HF_POLICY_MAP.items():
        args = build_args_from_policy_map(env_name, policy_info)
        run_config(env_name, script_path, args, successes, failures)

    print_summary(successes, failures)


if __name__ == "__main__":
    main()
