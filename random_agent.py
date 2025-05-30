import argparse
from common.utils import (
    setup_environment,
    setup_save_directory,
    setup_video_recording,
    evaluate_policy
)


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self,state,evaluate=False):
        return self.action_space.sample()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_episodes", default=50, type=int)
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    save_dir = setup_save_directory("random_agent", args.env)
    env = setup_environment(args.env, args.seed, "rgb_array")
    if args.save_video:
        env = setup_video_recording(env, save_dir)

    agent = RandomAgent(env.action_space)
    evaluate_policy(agent, env, num_episodes=args.eval_episodes, save_dir=save_dir)
    env.close()


if __name__ == "__main__":
    main()
