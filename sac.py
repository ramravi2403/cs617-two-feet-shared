from typing import Dict
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns

from common.utils import (
    EvalWrapper,
    get_device,
    plot_metrics,
    setup_environment,
    get_env_info,
    setup_save_directory,
    setup_video_recording,
    print_episode_info,
    run_evaluation,
)
from common.base_agent import BaseAgent
from common.replay_buffer import ReplayBuffer
from common.actor import Actor
from common.critic import Critic
import gymnasium as gym

from common.utils import save_expt_metadata

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


class SAC(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
        lr: float = 3e-4,
    ):
        super().__init__(state_dim, action_dim, max_action, device)

        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Temperature
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.cpu().data.numpy().flatten()
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
    ) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_pi
            target_q = rewards + (1 - dones) * gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_new, log_pi = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, action_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * log_pi - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target networks
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
        }

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f"{directory}/sac_actor_{name}.pth")
        torch.save(self.critic.state_dict(), f"{directory}/sac_critic_{name}.pth")

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f"{directory}/sac_actor_{name}.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/sac_critic_{name}.pth"))


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--hardcore", action="store_true", help="Use hard core mode")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for reproducibility")
    parser.add_argument("--max_timesteps", default=500_000, type=int, help="Maximum number of timesteps to train for")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training")
    parser.add_argument("--save_freq", default=50000, type=int, help="Frequency to save model weights")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for all networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--train_freq", type=int, default=64, help="How often to train the agent (in environment steps)")
    parser.add_argument("--gradient_steps", type=int, default=32, help="Number of gradient steps to perform per training iteration")
    parser.add_argument("--learning_start", type=int, default=10_000, help="Number of steps to wait before training")
    parser.add_argument("--replay_buffer_size", type=int, default=300_000, help="Size of the replay buffer")
    return parser.parse_args()
    # fmt: on


def train_sac(
    agent: SAC,
    env: gym.Env,
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    """Main training loop for SAC.

    Args:
        agent: The SAC agent to train
        env: The environment to train in
        args: Command line arguments
        save_dir: Directory to save model checkpoints and metrics
    """
    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    episode_rewards, episode_lengths = [], []
    actor_losses, critic_losses, alpha_losses = [], [], []

    total_training_start = time.time()
    episode_start = time.time()

    for t in range(args.max_timesteps):
        episode_steps += 1
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(replay_buffer) > args.learning_start and t % args.train_freq == 0:
            # Perform multiple gradient steps
            for _ in range(args.gradient_steps):
                train_info = agent.train(
                    replay_buffer=replay_buffer,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    tau=args.tau,
                )
                actor_losses.append(train_info["actor_loss"])
                critic_losses.append(train_info["critic_loss"])
                alpha_losses.append(train_info["alpha_loss"])

        if done:
            episode_time = time.time() - episode_start
            print_episode_info(
                t + 1, episode_num, episode_steps, episode_reward, episode_time
            )

            episode_rewards.append(float(episode_reward))
            episode_lengths.append(int(episode_steps))

            # Reset env
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
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
    plot_metrics(
        data=[actor_losses, critic_losses, alpha_losses],
        data_labels=["Actor Loss", "Critic Loss", "Temperature Loss"],
        img_name="convergence_metrics.png",
        x_label="Steps",
        sma_window_size=10_000,
        save_dir=save_dir,
    )
    save_expt_metadata(
        save_dir=save_dir,
        hyperparameters=vars(args),
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        total_training_time=total_training_time,
        convergence_metrics={
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "alpha_losses": alpha_losses,
        },
    )


def main():
    args = parse_args()
    env = setup_environment(env_name=args.env, seed=args.seed, hardcore=args.hardcore)
    state_dim, action_dim, max_action = get_env_info(env)

    device = get_device()
    print(f"Using device: {device}")
    agent = SAC(state_dim, action_dim, max_action, device, lr=args.lr)

    if args.evaluate:
        run_evaluation(agent, env, args)
        return

    save_dir = setup_save_directory("sac", args.env)
    if args.save_video:
        env = setup_video_recording(env, save_dir)

    train_sac(agent=agent, env=env, args=args, save_dir=save_dir)
    env.close()


if __name__ == "__main__":
    main()
