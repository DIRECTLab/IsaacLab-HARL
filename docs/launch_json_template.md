# Launch Json Template
This is a template to build your `launch.json` file to use the VS code debugger when training and playing with different policies.

### Example `__init__.py` containing the registered gym environment for H1
* [`__init__.py`](../source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/__init__.py)
```python
gym.register(
    id="Isaac-H1-Velocity-Direct-v0",
    entry_point=f"{__name__}.h1_env_velocity:H1VelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_velocity:H1VelocityEnvCfg",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```


### Example `launch.json`
* The example below has two configurations, one for `play.py` and one for `train,py` with the harl policy.
* Note:
    * In `args`, you need to pass in the `id` you registered with the gym environment to the `--task` option.
    * For the play config, you need to update the argument for the `--dir` option to be the directory that contains the `.pt` files of the trained model.


```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "h1_train",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/reinforcement_learning/harl/train.py",
            "python": "${command:python.interpreterPath}",
            "justMyCode": false,
            "console": "integratedTerminal",
            "args": [
                "--num_envs", "3_500",
                "--task", "Isaac-H1-Velocity-Direct-v0",
                "--num_env_steps", "1_000_000_000",
                "--save_interval", "100",
                "--log_interval", "1",
                "--headless",
            ],
        },
        {
            "name": "h1_play",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/reinforcement_learning/harl/play.py",
            "python": "${command:python.interpreterPath}",
            "justMyCode": false,
            "console": "integratedTerminal",
            "args": [
                "--num_envs", "7",
                "--task", "Isaac-H1-Velocity-Direct-v0",
                "--num_env_steps", "10000",
                "--algo", "happo",
                "--enable_cameras",
                "--debug",
                "--dir", "${workspaceFolder}/results/isaaclab/Isaac-H1-Velocity-Direct-v0/happo/test/seed-00001-2025-08-04-10-44-47/best_model",
            ],
        }
    ]
}
```
