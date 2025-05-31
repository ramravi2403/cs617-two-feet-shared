# BipedalWalker RL Implementations

This repository contains implementations of Soft Actor-Critic (SAC) and Advantage Actor-Critic (A2C) for training the BipedalWalker-v3 environment from OpenAI Gymnasium.

## Installation

If you use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
```bash
uv sync
```

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### SAC Implementation

To train the SAC agent with default parameters:
```bash
python sac.py
```

#### SAC Command Line Arguments

- `--env`: Environment name (default: "BipedalWalker-v3")
- `--seed`: Random seed (default: 0)
- `--max_timesteps`: Maximum number of training timesteps (default: 1,000,000)
- `--batch_size`: Batch size for training (default: 256)
- `--save_freq`: How often to save model checkpoints (default: 50000 steps)
- `--eval_freq`: How often to evaluate the agent (default: 5000 steps)
- `--save_video`: Flag to enable video recording of episodes (records every 50th episode)

Example with custom parameters:
```bash
python sac.py --max_timesteps 2000000 --save_video
```

Evaluation after model is trained
```bash
python sac.py --evaluate \
    --model_path results/sac_BipedalWalker-v3_20250522_144908/sac_critic_step_300000.pth  \
    --eval_episodes 100
```

### A2C Implementation

To train the A2C agent with default parameters:
```bash
python a2c.py
```

#### A2C Command Line Arguments

- `--env`: Environment name (default: "BipedalWalker-v3")
- `--seed`: Random seed (default: 0)
- `--max_timesteps`: Maximum number of training timesteps (default: 1,000,000)
- `--update_freq`: Steps between policy updates (default: 2048)
- `--save_freq`: How often to save model checkpoints (default: 50000 steps)
- `--eval_freq`: How often to evaluate the agent (default: 5000 steps)
- `--save_video`: Flag to enable video recording of episodes (records every 50th episode)

Example with custom parameters:
```bash
python a2c.py --max_timesteps 2000000 --save_video
```

Evaluation after model is trained
```bash
python a2c.py --evaluate \
    --model_path results/a2c_BipedalWalker-v3_20250523_014106/a2c_critic_step_300000.pth \
    --eval_episodes 100
```

## Features

### SAC Features
- Automatic temperature tuning
- Model checkpointing
- Video recording of training episodes
- Efficient replay buffer implementation
- Support for both training and evaluation modes

### A2C Features
- On-policy learning with advantage estimation
- Model checkpointing
- Video recording of training episodes
- Entropy regularization for exploration
- Support for both training and evaluation modes

## Output

The training results will be saved in the `results` directory with the following structure:

For SAC:
```
results/
    sac_BipedalWalker-v3_YYYYMMDD_HHMMSS/
        sac_actor_step_XXXXX.pth
        sac_critic_step_XXXXX.pth
        videos/  (if --save_video is enabled)
```

For A2C:
```
results/
    a2c_BipedalWalker-v3_YYYYMMDD_HHMMSS/
        a2c_actor_critic_step_XXXXX.pth
        videos/  (if --save_video is enabled)
```