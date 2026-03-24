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
GAMES_PER_ITER = 8  # number of self-play games per iteration
MCTS_SIMULATIONS = 30  # MCTS simulations per move (larger is stronger but slower)
C_PUCT = 1.0  # exploration constant for MCTS

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