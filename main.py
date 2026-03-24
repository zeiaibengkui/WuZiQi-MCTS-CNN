"""
Main entry point for Gomoku AlphaZero.
"""

import random
import os
import torch
from time import sleep

import config
import train
import play

if __name__ == "__main__":
    choice = input("1. Train AI (self-play)\n2. Human vs AI\nPlease choose (1/2): ")

    if choice == "1":
        trainer = train.Trainer()
        # load existing model if any
        trainer.load_model()

        for i in range(config.TRAIN_ITERATIONS):
            # print(f"\n--- Training iteration {i+1}/{config.TRAIN_ITERATIONS} ---")

            # config.MCTS_SIMULATIONS = random.randrange(150, 200)
            trainer.mcts.n_simulations = config.MCTS_SIMULATIONS
            # print(f"starting next | MCTS Sims: {config.MCTS_SIMULATIONS}")
            trainer.train_step()
            sleep(0.7)
    elif choice == "2":
        config.MCTS_SIMULATIONS = 200
        play.human_vs_ai()
    else:
        print("Invalid choice")
