import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns

from common.utils import (
    EvalWrapper,
    plot_metrics,
    get_device,
    run_evaluation,
    save_expt_metadata,
    setup_environment,
    get_env_info,
    setup_save_directory,
    setup_video_recording,
    print_episode_info,
)
from common.base_agent import BaseAgent
from common.actor import Actor
from common.critic import ValueCritic
from torch.distributions import Normal


class PPO(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        n_epochs=10,
        batch_size=64,
        hidden_dim=64,
        target_kl=0.1,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, max_action, device)

        self.actor = Actor(state_dim, action_dim, max_action=max_action, hidden_dim=hidden_dim).to(device)
        self.critic = ValueCritic(state_dim, hidden_dim=hidden_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = max(target_kl, 0.02)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.detach().cpu().numpy().flatten()
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def compute_gae(self, rewards, values, dones, next_value):
        values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        gae = 0
        returns, advantages = [], []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        return returns, advantages

    def _get_log_probs(self, states, actions):
        mean, log_std = self.actor(states)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        normal = Normal(mean, std)
        pre_tanh_action = torch.atanh(torch.clamp(actions / self.actor.max_action, -0.999, 0.999))
        log_prob = normal.log_prob(pre_tanh_action).sum(dim=1)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=1)
        return log_prob.unsqueeze(-1)

    def _get_entropy(self, states):
        mean, log_std = self.actor(states)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        return normal.entropy().sum(dim=1).mean()

    def train(self, states, actions, rewards, dones, next_state):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            old_log_probs = self._get_log_probs(states, actions)

            values = self.critic(states).squeeze().detach()
            next_state_tensor = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
            next_value = self.critic(next_state_tensor).detach().squeeze()

            returns, advantages = self.compute_gae(rewards, values, dones, next_value)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            kl_divs = []
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
                new_log_probs = self._get_log_probs(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_preds = self.critic(batch_states).squeeze()
                value_loss = F.mse_loss(value_preds, batch_returns)

                entropy = self._get_entropy(batch_states)
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                kl = (batch_old_log_probs - new_log_probs).mean().item()
                kl_divs.append(kl)

            if np.mean(kl_divs) > self.target_kl * 1.5:
                break

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
            "kl_divergence": np.mean(kl_divs),
        }

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f"{directory}/ppo_actor_{name}.pth")
        torch.save(self.critic.state_dict(), f"{directory}/ppo_critic_{name}.pth")

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f"{directory}/ppo_actor_{name}.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/ppo_critic_{name}.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=3_000_000)
    parser.add_argument("--update-freq", type=int, default=2048)
    parser.add_argument("--save-freq", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--eval-episodes", type=int, default=100)
    return parser.parse_args()


def train_ppo(agent, env, args, save_dir):
    state, _ = env.reset()
    episode_reward, episode_steps, episode_num = 0, 0, 0
    states, actions, rewards, dones = [], [], [], []
    episode_rewards, episode_lengths = [], []
    total_training_start = time.time()
    episode_start = time.time()

    for t in range(args.max_timesteps):
        episode_steps += 1
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))

        state = next_state
        episode_reward += reward

        if len(states) >= args.update_freq or done:
            update_info = agent.train(states, actions, rewards, dones, next_state)
            states.clear(), actions.clear(), rewards.clear(), dones.clear()

        if done:
            episode_time = time.time() - episode_start
            print_episode_info(t + 1, episode_num, episode_steps, episode_reward, episode_time)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            state, _ = env.reset()
            episode_reward, episode_steps = 0, 0
            episode_num += 1
            episode_start = time.time()

        if (t + 1) % args.save_freq == 0:
            agent.save(save_dir, f"step_{t+1}")

    total_training_time = time.time() - total_training_start
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    plot_metrics(
        data=[episode_rewards, episode_lengths],
        data_labels=["Episode Reward", "Episode Length"],
        img_name="training_curves.png",
        x_label="Episodes",
        sma_window_size=10,
        save_dir=save_dir,
    )
    save_expt_metadata(
        save_dir=save_dir,
        hyperparameters=args,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        total_training_time=total_training_time,
        convergence_metrics={},
    )


def main():
    args = parse_args()
    env = setup_environment(args.env, args.seed, render_mode="human" if args.render else 'rgb_array')

    state_dim, action_dim, max_action = get_env_info(env)
    device = get_device()
    print(f"Using device: {device}")

    agent = PPO(
        state_dim,
        action_dim,
        max_action,
        device,
        **vars(args)
    )

    if args.evaluate:
        run_evaluation(agent, env, args)
        return

    save_dir = setup_save_directory("ppo", args.env)
    if args.save_video:
        env = setup_video_recording(env, save_dir)

    train_ppo(agent, env, args, save_dir)
    env.close()


if __name__ == "__main__":
    main()
