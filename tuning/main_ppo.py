from tuning.config.ppo_config import PPO_CONFIG
from core.optimization_manager import OptimizationManager
from trainers.ppo_trainer import PPOTrainer



def main():
    # Create study
    trainer = PPOTrainer(PPO_CONFIG)
    optimizer = OptimizationManager(PPO_CONFIG, trainer)
    study = optimizer.optimize()


if __name__ == "__main__":
    main()
