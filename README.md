# BipedalWalker RL Implementations

This repository contains implementations of Soft Actor-Critic (SAC), Advantage Actor-Critic (A2C), A2C with Generalized Advantage Estimation (A2C-GAE), and a Random Agent for training the BipedalWalker-v3 environment from OpenAI Gymnasium.

## Installation

If you use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management:
```bash
uv sync
```

Else, you could create the environment as below:

1. Create a virtual environment:
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
- `--hardcore`: Use hard core mode (flag)
- `--seed`: Random seed for reproducibility (default: 0)
- `--max_timesteps`: Maximum number of timesteps to train for (default: 500,000)
- `--batch_size`: Batch size for training (default: 256)
- `--save_freq`: Frequency to save model weights (default: 50,000 steps)
- `--save_video`: Flag to enable video recording of episodes
- `--evaluate`: Run evaluation mode (flag)
- `--model_path`: Path to the model to evaluate
- `--eval_episodes`: Number of episodes for evaluation (default: 100)
- `--lr`: Learning rate for all networks (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Target network update rate (default: 0.005)
- `--train_freq`: How often to train the agent in environment steps (default: 64)
- `--gradient_steps`: Number of gradient steps to perform per training iteration (default: 32)
- `--learning_start`: Number of steps to wait before training (default: 10,000)
- `--replay_buffer_size`: Size of the replay buffer (default: 300,000)

Example with custom parameters:
```bash
python sac.py \
    --max_timesteps 300000 \
    --batch_size 512 \
    --lr 0.00036000911929116066 \
    --gamma 0.99 \
    --tau 0.0062973606231269685 \
    --train_freq 96 \
    --gradient_steps 44 \
    --learning_start 10000 \
    --replay_buffer_size 300000 \
    --save_video
```

Evaluation after model is trained:
```bash
python sac.py --evaluate \
    --model_path results/sac_BipedalWalker-v3_20250522_144908sac_critic_step_300000.pth  \
    --eval_episodes 50
    --save_video
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
- `--update_freq`: Steps between policy updates (default: 2,048)
- `--save_freq`: How often to save model checkpoints (default: 50,000 steps)
- `--save_video`: Flag to enable video recording of episodes
- `--evaluate`: Run evaluation mode (flag)
- `--model_path`: Path to the model to evaluate
- `--eval_episodes`: Number of episodes for evaluation (default: 100)

Example with custom parameters:
```bash
python a2c.py --max_timesteps 2000000 --save_video
```

Evaluation after model is trained:
```bash
python a2c.py --evaluate \
    --model_path results/a2c_BipedalWalker-v3_20250523_014106/a2c_critic_step_300000.pth \
    --eval_episodes 50
    --save_video
```

### A2C with GAE Implementation

To train the A2C-GAE agent with default parameters:
```bash
python a2c_gae.py
```

#### A2C-GAE Command Line Arguments

- `--env`: Environment name (default: "BipedalWalker-v3")
- `--seed`: Random seed (default: 0)
- `--max_timesteps`: Maximum number of training timesteps (default: 1,000,000)
- `--n_steps`: Number of steps for rollout (default: 8)
- `--save_freq`: How often to save model checkpoints (default: 50,000 steps)
- `--eval_freq`: How often to evaluate the agent (default: 5,000 steps)
- `--save_video`: Flag to enable video recording of episodes
- `--entropy_coef`: Entropy coefficient for exploration (default: 0.01)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--gae_lambda`: GAE lambda parameter (default: 0.9)
- `--hidden_dim`: Hidden layer dimension (default: 128)
- `--log_std_init`: Initial log standard deviation (default: 0.0)
- `--evaluate`: Run evaluation mode (flag)
- `--model_path`: Path to the model to evaluate
- `--eval_episodes`: Number of episodes for evaluation (default: 100)

Example with custom parameters:
```bash
python a2c_gae.py --max_timesteps 2000000 --save_video --gae_lambda 0.95
```

### Random Agent

To run the random agent:
```bash
python random_agent.py
```

#### Random Agent Command Line Arguments

- `--env`: Environment name (default: "BipedalWalker-v3")
- `--seed`: Random seed (default: 0)
- `--evaluate`: Run evaluation mode (flag)
- `--eval_episodes`: Number of episodes for evaluation (default: 50)
- `--save_video`: Flag to enable video recording of episodes

Example:
```bash
python random_agent.py --eval_episodes 100 --save_video
```


## Output

The training results will be saved in the `results` directory with the following structure:

```
results/
    {algorithm_name}_BipedalWalker-v3_YYYYMMDD_HHMMSS/
        sac_actor_step_XXXXX.pth
        sac_critic_step_XXXXX.pth
        videos/  (if --save_video is enabled)
```