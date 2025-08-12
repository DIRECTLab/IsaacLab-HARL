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

This will automatically install the modified HARL package that works with isaaclab that we developed located at [https://github.com/some45bucks/HARL](https://github.com/some45bucks/HARL).  

Install isaacsim 

```
pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

download the 3D assets form here:
[https://usu.box.com/s/af10jukvqp4gun3xx2cjqbd4vsq840ek](https://usu.box.com/s/af10jukvqp4gun3xx2cjqbd4vsq840ek)
Place the downloaded files into a new folder named `assets` located in the root directory of the project:

```
IsaacLab-HARL/
├── assets/
│   └── <downloaded unzipped files>
├── README.md
├── ...
```

# Multi-Agent Training with HARL

This command runs training on the multi-agent ANYmal environment using the HAPPO (Heterogeneous Agent Proximal Policy Optimization) algorithm in IsaacLab-HARL.

## Command

```bash
cd IsaacLab-HARL/scripts/reinforcement_learning/harl
python train.py --video --video_length 500 --video_interval 20000 --num_envs 64 --task "Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0" --seed 1 --save_interval 10000 --log_interval 1 --exp_name "multi_agent_anymal_harl" --num_env_steps 1000000 --algorithm happo --headless
```

Outputs will be located at `IsaacLab-HARL/scripts/reinforcement_learning/harl/results`, to view the progress in tensorboard run

```bash
cd IsaacLab-HARL/scripts/reinforcement_learning/harl/results/
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

## Available Algorithms

* `happo`: Heterogeneous Agent Proximal Policy Optimization
* `hatrpo`: Heterogeneous Agent Trust Region Policy Optimization
* `haa2c`: Heterogeneous Agent Advantage Actor-Critic
* `mappo`: Multi-Agent Proximal Policy Optimization (shared policy)
* `mappo_unshare`: Multi-Agent Proximal Policy Optimization (unshared policy)

## Available Tasks

These environments are located in:

```
IsaacLab-HARL/source/isaaclab_tasks/isaaclab_tasks/direct
```

* `Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0`
* `Isaac-Anymal-H1-Ball-Direct-v0`
* `Isaac-Anymal-H1-Piano-Direct-v0`
* `Isaac-Anymal-H1-Push-Direct-v0`
* `Isaac-Anymal-H1-Surf-Flat-Direct`


## Playing an Environment

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{haight2025heterogeneous,
  author    = {Haight, Jacob and Peterson, Isaac and Allred, Christopher and Harper, Mario},
  title     = {Heterogeneous Multi-Agent Learning in Isaac Lab: Scalable Simulation for Robotic Collaboration},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2025},
  pages     = {xx--xx},
  address   = {Hangzhou, China},
  publisher = {IEEE},
  doi       = {DOI_TBD_HERE},
  url       = {https://directlab.github.io/IsaacLab-HARL/}
}
```



# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows, such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks. Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:
- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area in the
`Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

Have a project or resource you'd like to share more widely? We'd love to hear from you! Reach out to the
NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to discuss potential opportunities
for broader dissemination of your work.

Join us in building a vibrant, collaborative ecosystem where creativity and technology intersect. Your
contributions can make a significant impact on the Isaac Lab community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
