from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch

from ppo import PPO
from tuning.config.optimization_config import OptimizationConfig
from tuning.trainers.base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def create_agent(self, params: Dict[str, Any]) -> PPO:

        state_dim, action_dim, max_action = self.env_manager.get_env_info()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"create_agent {state_dim=}, {action_dim=}, {max_action=}, {device=}")

        agent = PPO(
            state_dim,
            action_dim,
            max_action,
            device,
            **params
        )

        return agent

    def train_step(self, agent: PPO, env: gym.Env, state: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        assert state.shape == next_state.shape
        done = terminated or truncated
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(float(done))

        if len(self.states) >= params['n_steps'] or done:
            print("Updating agent") 
            agent.train(
                self.states, self.actions, self.rewards,
                self.dones,
                self.next_states[-1],
            )
            self.states, self.actions, self.rewards = [], [], []
            self.next_states, self.dones = [], []

        if done:
            next_state, _ = env.reset()
        return next_state
