# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Gomoku (五子棋) AI trainer using Monte Carlo Tree Search (MCTS) with a convolutional neural network (CNN), inspired by AlphaZero. The project has been extended with a parallel models competition mechanism.

## Key Files

- `main.py`: Entry point (train or human vs AI)
- `train.py`: Trainer class with self‑play, training loop, and parallel models competition
- `model.py`: Neural network architecture (U‑Net style with policy and value heads)
- `mcts.py`: Monte Carlo Tree Search implementation
- `game.py`: Gomoku game logic
- `config.py`: Configuration constants
- `play.py`: Human vs AI play interface

## Parallel Models Competition

The trainer maintains a population of `PARALLEL_MODELS` (default 3) neural networks. Every `COMPETITION_FREQUENCY` training iterations (default 1), a competition round is run:

1. **Pairwise matches**: Each model plays every other model twice (once as black, once as white).
2. **Statistics**: Wins, losses, and draws are recorded per model.
3. **Loss data collection**: Game data (states, policies, rewards) from games where a model lost are stored for training.
4. **Elimination**: Models whose loss rate reaches `ELIMINATION_THRESHOLD` (default 0.7) are eliminated.
5. **Replacement**: Eliminated models are replaced with random initialization (not copies of the best model).
6. **Best model propagation**: The best model’s weights are copied to the main network (index 0). The best model’s weights are kept unchanged (no training).

## Configuration (config.py)

- `BOARD_SIZE`: 15 (standard Gomoku board)
- `DEVICE`: cuda if available, else cpu
- `MODEL_PATH`: "gomoku_model.pth"
- `BEST_LENGTH_MODEL_DIR`: directory for models that achieved longest games
- `TRAIN_ITERATIONS`: total training iterations (default 700)
- `GAMES_PER_ITER`: (unused in current version, kept for compatibility) originally number of self‑play games per iteration (default 5)
- `MCTS_SIMULATIONS`: MCTS simulations per move (default 30)
- `C_PUCT`: exploration constant (default 1.0)
- `PARALLEL_MODELS`: number of parallel models in the population (default 3)
- `COMPETITION_FREQUENCY`: run competition every N training iterations (default 1, i.e., every iteration)
- `ELIMINATION_THRESHOLD`: loss rate threshold for elimination (default 0.7)

## Training Workflow

1. **Competition round**: Every `COMPETITION_FREQUENCY` iterations (default 1), all models compete in pairwise matches. Game data from losses is collected.
2. **Training losers**: Models that lost games (excluding the best model) are trained using their own loss data. The best model’s weights are kept unchanged.
3. **Elimination**: Models with loss rate ≥ `ELIMINATION_THRESHOLD` are eliminated and replaced with random initialization.
4. **Best model propagation**: The best model’s weights are copied to the main network (index 0) for saving and human play.
5. **Longest game recording**: The longest game from competition is saved as a benchmark; if it beats the previous record, the model is saved in `BEST_LENGTH_MODEL_DIR`.

Note: No self‑play is used; training relies solely on competition loss data.

## Commands

### Setup
```bash
# Install dependencies (uses pyproject.toml)
pip install -e .
```

### Running the AI
```bash
# Start the main program (choose training or human play)
python main.py
```

### Training
```bash
# Directly invoke the trainer (bypasses menu)
python -c "import train; t = train.Trainer(); t.load_model(); t.train_step()"
```

### Human vs AI
```bash
# Start a human‑vs‑AI game (requires trained model)
python play.py
```

### Model management
```bash
# List saved models in best_length_models directory
ls best_length_models/
```

## Development Notes

- The first move is always fixed at the center (7,7).
- Training is competition‑based: no self‑play, losers are trained using their loss data, winner weights are frozen.
- Critical moves (win, block, open four, open three) are prioritized during MCTS.
- The training loop includes a short sleep per game to reduce CPU load.
- All models share the same architecture; only their weights differ.
- The neural network uses a U‑Net style encoder‑decoder with skip connections, producing policy and value heads.
- MCTS search focuses on a neighborhood of the last two moves after the second move, improving efficiency.
- Reward function includes a length factor that encourages longer games.

## Future Improvements

- Add symmetry augmentation for training data.
- Adjust competition scheduling (e.g., more frequent early, less later).
- Introduce noise (mutations) when copying the best model to eliminated slots.
- Save and load the whole population (not just the main model).