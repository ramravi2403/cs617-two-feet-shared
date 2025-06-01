import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from common.utils import get_env_info
from torch.distributions import Normal
import matplotlib.pyplot as plt
import argparse
import os

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), log_std_init=-0.5):
        super().__init__()
        self.policy_net = self._mlp(obs_dim, hidden_sizes, act_dim)
        self.value_net = self._mlp(obs_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def _mlp(self, input_dim, hidden_sizes, output_dim):
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    def step(self, obs):
        with torch.no_grad():
            obs = obs.to(next(self.parameters()).device)
            mean = self.policy_net(obs)
            std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            value = self.value_net(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        obs = obs.to(next(self.parameters()).device)
        actions = actions.to(next(self.parameters()).device)
        mean = self.policy_net(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        values = self.value_net(obs).squeeze(-1)
        return log_probs, entropy, values


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        device,
        hidden_sizes=(64, 64),
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        target_kl=0.01,
        gamma=0.99,
        lam=0.95,
        steps_per_epoch=4000,
        batch_size=64,
            **kwargs
    ):
        self.actor_critic = MLPActorCritic(obs_dim, act_dim, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.device = device
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.act_limit = torch.tensor(act_limit).to(self.device)
        self.reward_avg = []
        self.logs = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "mean_reward": [],
            "episode_lengths": [],
        }
    def to_tensor(self,x):
        return torch.tensor(np.array(x), dtype=torch.float32, device=self.device)

    def compute_advantages(self, rewards, values, dones):
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * (1 - dones[t]) * lastgaelam
        return adv

    def update(self, obs_buf, act_buf, adv_buf, ret_buf, logp_old_buf):
        obs_buf = self.to_tensor(obs_buf)
        act_buf = self.to_tensor(act_buf)
        adv_buf = self.to_tensor(adv_buf)
        ret_buf = self.to_tensor(ret_buf)
        logp_old_buf = self.to_tensor(logp_old_buf)

        for _ in range(self.train_iters):
            idx = np.random.permutation(len(obs_buf))
            for start in range(0, len(obs_buf), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                logp, entropy, values = self.actor_critic.evaluate(obs_buf[batch_idx], act_buf[batch_idx])
                ratio = torch.exp(logp - logp_old_buf[batch_idx])
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                loss_pi = -torch.min(ratio * adv_buf[batch_idx], clipped * adv_buf[batch_idx]).mean()

                loss_v = ((values - ret_buf[batch_idx]) ** 2).mean()
                loss_entropy = entropy.mean()

                loss = loss_pi + 0.5 * loss_v - 0.01 * loss_entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                kl = (logp_old_buf[batch_idx] - logp).mean().item()
                if kl > 1.5 * self.target_kl:
                    return  # Early stopping

    def train(self, env, epochs=50):
        obs, _ = env.reset()
        episode_lengths = []
        current_ep_len = 0

        for epoch in range(epochs):
            obs_buf, act_buf, adv_buf, ret_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], [], []
            for _ in range(self.steps_per_epoch):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action, logp, value = self.actor_critic.step(obs_tensor)
                clipped_action = torch.clamp(action, -self.act_limit, self.act_limit)
                next_obs, reward, terminated, truncated, _ = env.step(clipped_action.numpy().squeeze())
                done = terminated or truncated

                obs_buf.append(obs)
                act_buf.append(action.numpy().squeeze())
                logp_buf.append(logp.item())
                rew_buf.append(reward)
                val_buf.append(value.item())
                done_buf.append(done)

                current_ep_len += 1
                obs = next_obs
                if done:
                    episode_lengths.append(current_ep_len)  # ⬅ Save episode length
                    current_ep_len = 0
                    obs, _ = env.reset()

            val_buf.append(0)
            adv_buf = self.compute_advantages(rew_buf, val_buf, done_buf)
            ret_buf = adv_buf + val_buf[:-1]

            self.update(obs_buf, act_buf, adv_buf, ret_buf, logp_buf)
            self.reward_avg.append(np.mean(rew_buf))
            self.logs["mean_reward"].append(np.mean(rew_buf))
            self.logs["episode_lengths"].append(np.mean(episode_lengths))


    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp, value = self.actor_critic.step(state_tensor)

        if not evaluate:
            clipped_action = torch.clamp(action, -self.act_limit, self.act_limit)
            return clipped_action.cpu().numpy().squeeze(), logp, value
        return action.cpu().numpy().squeeze()

    def plot_training_curves(self):
        steps = np.arange(len(self.logs["actor_loss"]))
        plt.figure(figsize=(16, 10))

        plt.subplot(3, 1, 1)
        plt.plot(steps, self.logs["actor_loss"], label="Actor Loss", alpha=0.6)
        plt.title("Actor Loss Over Time")
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(steps, self.logs["critic_loss"], label="Critic Loss", alpha=0.6, color='blue')
        plt.title("Critic Loss Over Time")
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(steps, self.logs["entropy"], label="Entropy", alpha=0.6, color='green')
        plt.title("Entropy Over Time")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_reward_distribution(self):
        rewards = self.logs["mean_reward"]
        plt.hist(rewards, bins=20, color='skyblue')
        plt.axvline(np.mean(rewards), color='red', linestyle='--',
                    label=f"Mean: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        plt.title("Reward Distribution")
        plt.xlabel("Mean Episode Reward")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()

    def export_summary_report(self):
        rewards = self.logs["mean_reward"]
        lengths = self.logs["episode_lengths"]

        with open("ppo_evaluation_summary.txt", "w") as f:
            f.write("Evaluation Summary:\n")
            f.write(f"Number of epochs: {len(rewards)}\n")
            f.write(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
            f.write(f"Min reward: {np.min(rewards):.2f}\n")
            f.write(f"Max reward: {np.max(rewards):.2f}\n")
            f.write(f"Mean episode length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}\n")
            f.write(f"Min episode length: {np.min(lengths):.2f}\n")
            f.write(f"Max episode length: {np.max(lengths):.2f}\n")

    def save(self, path="checkpoints"):
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Create a state dictionary containing all necessary components
        state_dict = {
            'actor_critic_state': self.actor_critic.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'act_limit': self.act_limit,
            'reward_avg': self.reward_avg if hasattr(self, 'reward_avg') else [],
            'training_step': self.training_step if hasattr(self, 'training_step') else 0
        }

        # Save the state dictionary
        checkpoint_path = os.path.join(path, f'ppo_checkpoint.pt')
        torch.save(state_dict, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def load(self, path="checkpoints"):
        checkpoint_path = os.path.join(path, f'ppo_checkpoint.pt')

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return False

        try:
            # Load the state dictionary
            checkpoint = torch.load(checkpoint_path)

            # Load actor-critic network state
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state'])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # Load other attributes
            self.act_limit = checkpoint['act_limit']
            if 'reward_avg' in checkpoint:
                self.reward_avg = checkpoint['reward_avg']
            if 'training_step' in checkpoint:
                self.training_step = checkpoint['training_step']

            print(f"Model loaded successfully from {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False



    # def record_video(self, video_path="ppo_bipedal_videos", iter = 0, episode_length=1600):
    #     env = RecordVideo(
    #         self.env,
    #         video_folder=video_path,
    #         episode_trigger=lambda ep: True,
    #         name_prefix=f"ppo_bipedal_iter_{iter}"
    #     )
    #     obs, _ = env.reset()
    #     total_reward = 0
    #     for _ in range(episode_length):
    #         obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    #         action, _, _ = self.actor_critic.step(obs_tensor)
    #         action_clipped = torch.clamp(action, -self.act_limit, self.act_limit)
    #         obs, reward, terminated, truncated, _ = env.step(action_clipped.numpy().squeeze())
    #         total_reward += reward
    #         if terminated or truncated:
    #             break
    #     env.close()
    #     print(f"Episode finished. Total reward: {total_reward}")
    #     print(f"Video saved to: {os.path.abspath(video_path)}")
    #
    # def plot_reward_avg(self):
    #     plt.plot(self.reward_avg)
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Average Reward")
    #     plt.title("Average Reward per Epoch")
    #     plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train or evaluate PPO agent')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the PPO agent')
    add_train_arguments(train_parser)

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the PPO agent')
    add_eval_arguments(eval_parser)

    return parser.parse_args()


def add_train_arguments(parser):
    # Environment
    parser.add_argument('--env', type=str, default='BipedalWalker-v3',
                        help='Gymnasium environment ID (default: BipedalWalker-v3)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='Total timesteps for training (default: 1000000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs per update (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda parameter (default: 0.95)')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range (default: 0.2)')

    # Saving and loading
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints (default: checkpoints)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save frequency in episodes (default: 10)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load a saved model (default: None)')

    # Evaluation during training
    parser.add_argument('--eval-freq', type=int, default=10,
                        help='Evaluation frequency in episodes (default: 10)')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes (default: 5)')


def add_eval_arguments(parser):
    parser.add_argument('--env', type=str, default='BipedalWalker-v3',
                        help='Gymnasium environment ID (default: BipedalWalker-v3)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')


def evaluate_policy(env, agent, num_episodes=5, render=False):
    """Evaluate the policy for given number of episodes"""
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            if render:
                env.render()

            action = agent.select_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def train(args):
    # Create environment
    try:
        env = gym.make(args.env, render_mode='human' if args.render else None)
    except Exception as e:
        print(f"Error creating environment {args.env}: {str(e)}")
        print("Available environments:", [env.id for env in gym.envs.registry.values()])
        return

    # Initialize PPO agent
    state_dim, action_dim, max_action = get_env_info(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim, action_dim, max_action, device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range
    )

    # Load pre-trained model if specified
    if args.load_model:
        success = agent.load(args.load_model)
        if not success:
            print("Failed to load model. Starting fresh training.")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    agent.train(env, epochs=args.n_epochs)

    env.close()
    print("Training completed!")
    agent.save('./best_models/ppo_gae/')
    # agent.plot_training_curves()
    agent.plot_reward_distribution()
    agent.export_summary_report()



def evaluate(args):
    # Create environment
    try:
        env = gym.make(args.env, render_mode='human' if args.render else None)
    except Exception as e:
        print(f"Error creating environment {args.env}: {str(e)}")
        return


    # Initialize and load agent
    state_dim, action_dim, max_action = get_env_info(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim, action_dim, max_action, device)
    success = agent.load(args.model_path)

    if not success:
        print(f"Failed to load model from {args.model_path}")
        return

    # Run evaluation
    print(f"Evaluating model from {args.model_path} on {args.env}")
    evaluate_policy(env, agent, args.num_episodes, args.render)

    env.close()


def main():
    args = parse_arguments()

    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        print("Please specify a command: train or eval")


if __name__ == "__main__":
    main()
