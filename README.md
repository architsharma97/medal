# Autonomous RL via MEDAL
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

## Setup

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries for a linux machine (if not already installed):
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Setup EARL Benchmark:
```sh
# navigate to where you want to setup EARL
git clone https://github.com/architsharma97/earl_benchmark.git
export PYTHONPATH=$PYTHONPATH:/path/to/earl_benchmark
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate arl
```

## Training
Train an episodic RL agent using SAC:
```sh
python3 oracle.py
```

Train an autonomous RL agent using MEDAL:
```sh
python3 medal.py
```

The training scripts use the config in `cfgs/<script_name>.yaml` by default. For example, `medal.py` uses `cfgs/medal.yaml`. To override the default config, you can either change the values in the config or do it directly in the command line as follows:
```sh
python3 medal.py env_name=sawyer_door # to run on sawyer door environment
python3 medal.py env_name=sawyer_peg # sawyer peg environment
python3 medal.py env_name=tabletop_manipulation # tabletop manipulation
```

You can monitor the results via tensorboard:
```sh
tensorboard --logdir exp_local
```

## Acknowledgements

The codebase is built on top of the PyTorch implementation of [DrQ-v2](https://arxiv.org/abs/2107.09645), original codebase linked [here](https://github.com/facebookresearch/drqv2). We thank the authors for an easy codebase to work with!
