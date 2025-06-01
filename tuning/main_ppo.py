from tuning.config.ppo_config import PPO_CONFIG
from tuning.core.optimization_manager import OptimizationManager
from tuning.trainers.ppo_trainer import PPOTrainer



def main():
    # Create study
    trainer = PPOTrainer(PPO_CONFIG)
    optimizer = OptimizationManager(PPO_CONFIG, trainer)
    study = optimizer.optimize()

    # Print results
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
