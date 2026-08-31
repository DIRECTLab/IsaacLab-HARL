
# Install
Make sure you have conda installed, for reference we used the following script to install conda

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

Once the script is installed, follow the prompts until conda is install and then follow the next steps.

![image](https://github.com/user-attachments/assets/dc883271-761c-4342-9b0b-15c584b33127)

# Install

Install the conda environment

```
./isaaclab.sh -c
```

Activate the conda environment and install other dependencies.
```
conda activate env_isaaclab
./isaaclab.sh -i
```

This will automatically install the modified HARL package that works with isaaclab that we developed located at [https://github.com/DIRECTLab/HARL](https://github.com/DIRECTLab/HARL).

Install isaacsim

```
pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
```


# Adversarial Multi-Agent Training with HARL

Adversarial multi-agent training enables two teams of agents to compete against each other, leading to emergent behaviors and more robust policies through strategic gameplay. This framework supports multiple training modes for different competitive scenarios.

## Command

```bash
cd scripts/reinforcement_learning/harl
python train.py --algorithm happo_adv --num_envs 1000 --num_env_steps 10000000000 --task "AnymalC_Soccer_Hetero_By_Team-v0" --save_interval 5 --log_interval 1 --adversarial_training_mode leapfrog --headless --load_starting_policy
```

## Adversarial Training Modes

The framework supports three distinct adversarial training modes:

* `parallel`: Both teams train simultaneously against each other in the same environment instance. This mode encourages dynamic strategy development as policies improve in tandem.
* `ladder`: Teams alternate training iterations, with one team training while the other uses a frozen policy. This allows each team to optimize against a stable opponent before roles reverse (default: 50,000,000 iterations per swap).
* `leapfrog`: Similar to ladder mode but with more frequent policy swaps. Teams alternate training more frequently to enable progressive skill development.

## Adversarial-Specific Parameters

* `--algorithm`: Use `happo_adv` for adversarial training with heterogeneous agents.
* `--adversarial_training_mode`: Choose the training mode (`parallel`, `ladder`, or `leapfrog`). Default: `parallel`.
* `--adversarial_training_iterations`: Number of training steps before swapping policies in `ladder` and `leapfrog` modes. Default: 50,000,000.

**Note**: When using ladder training with teams composed of heterogeneous agents, both teams must place robots in the same order in their environment configuration for the mode to work correctly.

## Playing Trained Adversarial Models

```bash
cd scripts/reinforcement_learning/harl
python play.py --algorithm happo_adv --num_envs 64 --num_env_steps 10000000000 --task "Sumo-Stage2-Hetero-By-Team-v0" --dir <path_to_trained_model>
```

The play script supports rendering trained adversarial policies and can optionally load pre-trained models from local paths or HuggingFace Hub using `--load_starting_policy` or `--load_trained_policy` flags instead of the `--dir` flag.

# Multi-Agent Training with HARL

This command runs training on the multi-agent ANYmal environment using the HAPPO (Heterogeneous Agent Proximal Policy Optimization) algorithm in IsaacLab-HARL.

## Command

```bash
cd scripts/reinforcement_learning/harl
python train.py --video --video_length 500 --video_interval 20000 --num_envs 64 --task "Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0" --seed 1 --save_interval 10000 --log_interval 1 --exp_name "multi_agent_anymal_harl" --num_env_steps 1000000 --algorithm happo --headless
```

Outputs will be located at `scripts/reinforcement_learning/harl/results`, to view the progress in tensorboard run

```bash
cd scripts/reinforcement_learning/harl/results/
tensorboard --logdir=./
```

## Parameter Descriptions

* `--video`: Enables recording of videos during training episodes.
* `--video_length`: Number of environment steps per recorded video (default: 500).
* `--video_interval`: Number of environment steps between video recordings (default: 20000).
* `--num_envs`: Number of parallel simulation environments to run (here, 64).
* `--task`: Specifies the training task/environment.
* `--seed`: Random seed for reproducibility (here, 1).
* `--save_interval`: Frequency (in episode steps) at which the model is saved.
* `--log_interval`: Frequency (in environment steps) at which logs are recorded (here, every 1000 steps).
* `--exp_name`: Name identifier for the experiment, used for organizing output files and logs.
* `--num_env_steps`: Total number of environment steps for training (here, 1,000,000).
* `--algorithm`: Specifies the RL algorithm to use.
* `--headless`: Runs the simulation without rendering.
* `--load_starting_policy`: Load a pre-trained starting policy from HuggingFace Hub for transfer learning. Cannot be used with `--load_trained_policy` or `--dir`.
* `--load_trained_policy`: Load a fully trained policy from HuggingFace Hub to resume training. Cannot be used with `--load_starting_policy` or `--dir`.
* `--dir`: Path to local directory containing model checkpoints to continue training from. Cannot be used with `--load_starting_policy` or `--load_trained_policy`.

## Available Algorithms

* `happo`: Heterogeneous Agent Proximal Policy Optimization
* `happo_adv`: Heterogeneous Agent Proximal Policy Optimization (adversarial variant for competitive multi-agent training)
* `hatrpo`: Heterogeneous Agent Trust Region Policy Optimization
* `haa2c`: Heterogeneous Agent Advantage Actor-Critic
* `mappo`: Multi-Agent Proximal Policy Optimization (shared policy)
* `mappo_unshare`: Multi-Agent Proximal Policy Optimization (unshared policy)

## Available Tasks

These environments are located in:

```
source/isaaclab_tasks/isaaclab_tasks/direct
```

Policy availability by environment:

| Environment                                   | Starting | Trained |
|-----------------------------------------------|----------|---------|
| Leatherback-Stage1-Soccer-v0                  | NO       | YES     |
| Leatherback-Stage2-Soccer-v0                  | YES      | YES     |
| AnymalC_Soccer_Hetero_By_Team-v0              | YES      | NO      |
| Sumo-Stage2-Hetero-By-Team-v0                 | YES      | YES     |
| Sumo-Stage2-Hetero-Same-Critic-v0             | YES      | NO      |
| Sumo-Stage2-Hetero-Same-Critic-No-Negative-v0 | YES      | NO      |
| Sumo-Stage2-Hetero-v0                         | YES      | YES     |
| Minitank-Adversarial-Direct-v0                | YES      | YES     |
| Anymal-C-Go-To-Point-Sumo                     | NO       | YES     |
| Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0    | YES      | YES     |
| Anymal-C-Sumo-Stage1-Blocks-Push-v0           | YES      | YES     |
| AnymalC_Soccer_Go_To_Point_Stage_0            | NO       | YES     |
| AnymalC_Soccer_Go_To_Ball_Stage_1             | YES      | YES     |
| AnymalC_Soccer_Score_Goals_Stage_2            | YES      | YES     |
| leatherback-Sumo-Direct-MA-Stage1-v0          | NO       | YES     |
| AnymalC-VS-Leatherback-Soccer-v0              | YES      | NO      |


## Playing an Environment

```bash
cd scripts/reinforcement_learning/harl
python play.py --algorithm happo --num_envs 32 --task "Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0" --dir <path_to_trained_model> --headless
```

### Playing Parameters

* `--algorithm`: Specifies which algorithm was used during training (e.g., `happo`, `happo_adv`, `mappo`).
* `--num_envs`: Number of parallel environments to run during playback. Default: 1.
* `--task`: Name of the task to play.
* `--dir`: Path to the directory containing trained model checkpoints. Cannot be used together with `--load_starting_policy` or `--load_trained_policy`.
* `--load_starting_policy`: Load a pre-trained starting policy from HuggingFace Hub. Mutually exclusive with `--dir` and `--load_trained_policy`. Policies are automatically downloaded if available for the specified task.
* `--load_trained_policy`: Load a fully trained policy from HuggingFace Hub for evaluation. Mutually exclusive with `--dir` and `--load_starting_policy`. Use this to play with published benchmark policies.
* `--num_env_steps`: Total environment steps to run before stopping. Default: unlimited until interrupted.
* `--debug`: Enable visualization debug mode for interactive rendering.

### Loading Policies from HuggingFace

Both training and playing scripts support loading pre-trained policies from the HuggingFace Hub when available. This enables:

* **Transfer learning**: Start training with a pre-trained policy using `--load_starting_policy`
* **Benchmark evaluation**: Load published trained policies using `--load_trained_policy` in play.py

**Important**: Policy loading is task-specific and depends on the availability of entries in the HuggingFace repository linked in the code (`HF_POLICY_MAP`). If a policy is not available for your chosen task, the scripts will display a message and continue with default initialization. You can run `play.py -h` or `train.py -h` for more details on these options.
## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{haight2025harl,
  author    = {Haight, Jacob and Peterson, Isaac and Allred, Christopher and Harper, Mario},
  booktitle = {2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title     = {Heterogeneous Multi-Agent Learning in Isaac Lab: Scalable Simulation for Robotic Collaboration},
  year      = {2025},
  pages     = {13446-13451},
  keywords  = {Training;Autonomous systems;Robot kinematics;Scalability;Collaboration;Reinforcement learning;Robots;Physics;Optimization;Videos},
  doi       = {10.1109/IROS60139.2025.11247098},
  url       = {https://directlab.github.io/IsaacLab-HARL/}
}
```

# Isaac Lab 3.0.0 Beta 2

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.1-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


This is the stable release branch for Isaac Lab 3.0.0 Beta 2 and supports
Isaac Sim 6.0.0 and 6.0.1. Use the
[`v3.0.0-beta2.patch1`](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-beta2.patch1)
tag for a reproducible release checkout. Active feature development continues
on the [`develop`](https://github.com/isaac-sim/IsaacLab/tree/develop) branch.


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows,
such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html),
it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real
transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based
cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and
computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks.
Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

A detailed description of Isaac Lab can be found in our [arXiv paper](https://arxiv.org/abs/2511.04831).

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:

- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with more than 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

### Documentation

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including
detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/environments.html)

## Performance Dashboard

We continuously benchmark Isaac Lab across different physics backends, renderers, and data types.
The **[Isaac Lab Performance Dashboard](https://nvidia.github.io/omniperf/)** provides interactive
charts showing preset comparison results, performance history, and environment scaling data from
our internal CI/CD benchmarks.

## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each
release of Isaac Lab. Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding
dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version         |
| ----------------------------- | ------------------------- |
| `release/3.0.0-beta2` branch  | Isaac Sim 6.0.0 / 6.0.1   |
| `develop` branch              | Isaac Sim 6.0.0 / 6.0.1   |
| `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v3.0.0*`                     | Isaac Sim 6.0.0 / 6.0.1   |
| `v2.3.X`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.X`                      | Isaac Sim 4.5             |
| `v2.0.X`                      | Isaac Sim 4.5             |

## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell)
area in the `Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas,
  asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of
  work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features,
  or general updates.

## Connect with the NVIDIA Omniverse Community

Do you have a project or resource you'd like to share more widely? We'd love to hear from you!
Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities
to spotlight your work.

You can also join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to
connect with other developers, share your projects, and help grow a vibrant, collaborative ecosystem
where creativity and technology intersect. Your contributions can make a meaningful impact on the Isaac Lab
community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its
corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its
dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Note that full-featured workflows (PhysX, RTX rendering, ROS, URDF/MJCF importers) require
[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), which includes
components under proprietary licensing terms. Kit-less Newton workflows do not require Isaac Sim.
Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for details.

Note that the `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms that can be found in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).


## Citation

If you use Isaac Lab in your research, please cite the technical report:

```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework.
We gratefully acknowledge the authors of Orbit for their foundational contributions.
