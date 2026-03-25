"""
Configuration constants for Gomoku AlphaZero.
"""

import torch
import json
import os

BOARD_SIZE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "gomoku_model.pth"
BEST_LENGTH_MODEL_DIR = "best_length_models"
TRAIN_ITERATIONS = 700  # training iterations (should be 1000+ in practice)
GAMES_PER_ITER = 5  # number of games per model pair per competition round
MCTS_SIMULATIONS = 150  # MCTS simulations per move (larger is stronger but slower)
C_PUCT = 1.0  # exploration constant for MCTS
PARALLEL_MODELS = 2  # number of parallel models in population
ELIMINATION_THRESHOLD = 0.8  # lose rate threshold for elimination
COMPETITION_FREQUENCY = 1  # run competition every iteration (no self-play)


def load_max_moves():
    """Load maximum moves from config.json, default to 0."""
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
            return config.get("max_moves", 0)
    return 0


def save_max_moves(max_moves):
    """Save maximum moves to config.json."""
    with open("config.json", "w") as f:
        json.dump({"max_moves": max_moves}, f)
