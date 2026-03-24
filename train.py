"""
Training loop for Gomoku AlphaZero.
"""

import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from time import sleep

import config
import game
import model
import mcts

class Trainer:
    def __init__(self):
        self.net = model.GomokuNet().to(config.DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.mcts = mcts.MCTS(self.net, n_simulations=config.MCTS_SIMULATIONS)
        self.replay_buffer = []
        # load or initialize longest game length
        self.longest_game_length = config.load_max_moves()

    def self_play(self):
        g = game.GomokuGame()
        states = []
        mcts_policies = []

        while not g.is_terminal():
            if not g.move_history:
                # first move fixed in center
                move = (7, 7)
                action_probs = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE, dtype="float")
                action_probs[7 * config.BOARD_SIZE + 7] = 1.0
            else:
                action_probs, move = self.mcts.get_action_distribution(g)

            if move is None:
                break

            state = g.get_state_tensor()
            states.append(state)
            mcts_policies.append(action_probs)
            g.make_move(move)

        # compute final reward (z)
        winner = g.winner
        total_steps = len(g.move_history)
        # long game reward factor: more steps -> higher win reward (max 2x), lower loss penalty (min 0.5x)
        length_factor = np.log(max(total_steps - 10, 1)) * 0.3

        rewards = []
        for i in range(len(states)):
            # for each step, if final winner is the player of that step, reward = 1 * length_factor, else -1 * loss_factor
            step_player = 1 if i % 2 == 0 else -1
            if winner == step_player:
                rewards.append(1.0 * length_factor + 1)
            elif winner == 0:
                rewards.append(0.9 * length_factor + 1)
            else:
                loss_factor = np.max((0.2, 1 - length_factor))
                rewards.append(-1.0 * loss_factor)

        return states, mcts_policies, rewards, g

    def train_step(self):
        self.net.train()
        states, policies, rewards = [], [], []

        print(f"Generating {config.GAMES_PER_ITER} self-play games...")
        games = []
        for i in range(config.GAMES_PER_ITER):
            sleep(0.7)
            print(f"Game {i+1}  ", end="\r")
            s, p, r, g = self.self_play()
            states.extend(s)
            policies.extend(p)
            rewards.extend(r)
            games.append(g)

        # save current iteration model
        torch.save(self.net.state_dict(), config.MODEL_PATH)

        # record and save model with longest game length
        if games:
            longest_game = max(games, key=lambda g: len(g.move_history))
            longest_moves = len(longest_game.move_history)
            if longest_moves > self.longest_game_length:
                self.longest_game_length = longest_moves
                os.makedirs(config.BEST_LENGTH_MODEL_DIR, exist_ok=True)
                best_path = os.path.join(
                    config.BEST_LENGTH_MODEL_DIR,
                    f"gomoku_model_max_moves_{longest_moves}.pth",
                )
                torch.save(self.net.state_dict(), best_path)
                # save to config.json
                config.save_max_moves(self.longest_game_length)
                print(f"New longest game ({longest_moves} moves)，model saved: {best_path}")

        if len(states) == 0:
            return

        # output an example game
        if games:
            print("Example game final state:")
            game.print_board(games[-1].board)

            print("Move order:")
            for step, (r, c, p) in enumerate(games[-1].move_history):
                player = "black" if p == 1 else "white"
                print(f"Move {step+1}: {player} ({r}{chr(ord('a') + c)})")
            if games[-1].winner == 0:
                print("Result: draw")
            elif games[-1].winner == 1:
                print("Result: black wins")
            else:
                print("Result: white wins")

        # apply symmetry augmentation (currently disabled)
        # states, policies, rewards = apply_symmetry_augmentation(states, policies, rewards)

        # build tensors
        state_tensor = torch.cat(states, dim=0).to(config.DEVICE)
        policy_tensor = torch.FloatTensor(np.array(policies)).to(config.DEVICE)
        value_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(config.DEVICE)

        # train
        self.optimizer.zero_grad()
        pred_policy, pred_value = self.net(state_tensor)

        # Loss = Value MSE + Policy CrossEntropy
        loss_value = F.mse_loss(pred_value, value_tensor)
        loss_policy = torch.mean(
            torch.sum(-policy_tensor * torch.log(pred_policy + 1e-6), dim=1)
        )
        loss = loss_value + loss_policy

        loss.backward()
        self.optimizer.step()

        print(
            f"Loss: {loss.item():.4f} (Val: {loss_value.item():.4f}, Pol: {loss_policy.item():.4f})"
        )

        print(f"Model saved to {config.MODEL_PATH}")

    def load_model(self):
        if os.path.exists(config.MODEL_PATH):
            self.net.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
            print("Model loaded.")
        else:
            print("No model found, starting with random weights.")
        self.net.eval()