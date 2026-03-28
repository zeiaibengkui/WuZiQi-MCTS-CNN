"""
Training loop for Gomoku AlphaZero.
"""

import os
import json
import random, copy
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
        # create population of models
        self.population = []
        num_models = max(1, config.PARALLEL_MODELS)
        for i in range(num_models):
            net = model.GomokuNet().to(config.DEVICE)
            self.population.append(net)
        # main net is the first one (will be trained)
        self.net = self.population[0]
        # create optimizer for each model
        self.optimizers = [
            optim.Adam(net.parameters(), lr=1e-3) for net in self.population
        ]
        self.optimizer = self.optimizers[0]  # keep for compatibility
        self.mcts = mcts.MCTS(self.net, n_simulations=config.MCTS_SIMULATIONS)
        self.replay_buffer = []
        # load or initialize longest game length
        self.longest_game_length = config.load_max_moves()
        # competition stats
        self.wins = [0.0] * len(self.population)
        self.losses = [0.0] * len(self.population)
        self.draws = [0.0] * len(self.population)
        self.best_index = 0  # index of best model (initially 0)
        self.iteration = 0  # training iteration counter
        self.game_data = [
            [] for _ in range(len(self.population))
        ]  # store game data for training losers
        self.longest_game = (
            None  # store the longest game object from last competition round
        )
        self.steps = 1

    def self_play(self):
        g = game.GomokuGame()
        states = []
        mcts_policies = []

        while not g.is_terminal():
            if not g.move_history:
                # first move fixed in center
                move = (7, 7)
                action_probs = np.zeros(
                    config.BOARD_SIZE * config.BOARD_SIZE, dtype="float"
                )
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
        # length_factor = np.log(max(total_steps - 10, 1)) * 0.3

        rewards = []
        for i in range(len(states)):
            # for each step, if final winner is the player of that step, reward = 1 * length_factor, else -1 * loss_factor
            step_player = 1 if i % 2 == 0 else -1
            if winner == step_player:
                rewards.append(1.0)
            elif winner == 0:
                rewards.append(0)
            else:
                rewards.append(-1.0)

        return states, mcts_policies, rewards, g

    def train_step(self):
        """Run competition round and train losing models using their loss data."""
        # print(f"Iteration {self.iteration + 1}")

        # Run competition round every COMPETITION_FREQUENCY iterations
        longest_game = None
        if (
            len(self.population) > 1
            and self.iteration % config.COMPETITION_FREQUENCY == 0
        ):
            print(f"\n--- Competition round (iteration {self.iteration + 1}) ---")
            longest_game = self.competition_round()
            self._update_best_index()
            win_rate = lambda idx: self.wins[idx] / (
                self.wins[idx] + self.losses[idx] + self.draws[idx]
            )

            # record longest game if available
            if longest_game is not None:
                # output example game (the longest game)
                # print("Example game final state:")
                longest_game.print_board()

                # Save
                longest_moves = len(longest_game.move_history)
                if longest_moves > self.longest_game_length:
                    self.longest_game_length = longest_moves
                    os.makedirs(config.BEST_LENGTH_MODEL_DIR, exist_ok=True)
                    best_path = os.path.join(
                        config.BEST_LENGTH_MODEL_DIR,
                        f"gomoku_model_max_moves_{longest_moves}.pth",
                    )
                    torch.save(self.population[0].state_dict(), best_path)
                    config.save_max_moves(self.longest_game_length)
                    print(
                        f"New longest game ({longest_moves} moves)，model saved: {best_path}"
                    )

                print(
                    f"Best model: {self.best_index} "
                    f"(win rate: {win_rate(self.best_index):.2f})"
                )

        # Train losing models using their loss data
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        trained_any = False

        for idx in range(len(self.population)):
            if idx == self.best_index:
                continue  # skip winner (keep weights unchanged)
            if not self.game_data[idx]:
                continue  # no loss data for this model

            trained_any = True
            net = self.population[idx]
            optimizer = self.optimizers[idx]
            net.train()

            # aggregate all loss data for this model
            all_states = []
            all_policies = []
            all_rewards = []
            for data in self.game_data[idx]:
                all_states.extend(data["states"])
                all_policies.extend(data["policies"])
                all_rewards.extend(data["rewards"])

            if len(all_states) == 0:
                continue

            # build tensors
            state_tensor = torch.cat(all_states, dim=0).to(config.DEVICE)
            policy_tensor = torch.FloatTensor(np.array(all_policies)).to(config.DEVICE)
            value_tensor = (
                torch.FloatTensor(np.array(all_rewards)).unsqueeze(1).to(config.DEVICE)
            )

            # train step
            optimizer.zero_grad()
            pred_policy, pred_value = net(state_tensor)

            loss_value = F.mse_loss(pred_value, value_tensor)
            loss_policy = torch.mean(
                torch.sum(-policy_tensor * torch.log(pred_policy + 1e-6), dim=1)
            )
            loss = loss_value + loss_policy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_value_loss += loss_value.item()
            total_policy_loss += loss_policy.item()

        # Eliminate weak models after competition round (if competition happened)
        if (
            len(self.population) > 1
            and self.iteration % config.COMPETITION_FREQUENCY == 0
        ):
            self.eliminate_weak_models()

        self.iteration += 1

        if trained_any:
            avg_loss = total_loss / len(
                [
                    idx
                    for idx in range(len(self.population))
                    if idx != self.best_index and self.game_data[idx]
                ]
            )
            avg_val = total_value_loss / len(
                [
                    idx
                    for idx in range(len(self.population))
                    if idx != self.best_index and self.game_data[idx]
                ]
            )
            avg_pol = total_policy_loss / len(
                [
                    idx
                    for idx in range(len(self.population))
                    if idx != self.best_index and self.game_data[idx]
                ]
            )
            print(
                f"Average loss: {avg_loss:.4f}"
                f" (Val: {avg_val:.4f}, Pol: {avg_pol:.4f})"
            )
        else:
            print("No training data for losing models.")

        self.steps += 1
        if self.steps % 30 == 1:
            torch.save(self.population[self.best_index].state_dict(), config.MODEL_PATH)
            print(f"Model saved to {config.MODEL_PATH}")

    def load_model(self):
        if os.path.exists(config.MODEL_PATH):
            state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
            self.net.load_state_dict(state_dict)
            # copy to all other models in population
            for i in range(1, len(self.population)):
                self.population[i].load_state_dict(state_dict)
            print("Model loaded and copied to all parallel models.")
        else:
            print("No model found, starting with random weights.")
        for net in self.population:
            net.eval()

    def _play_match(self, idx1, idx2):
        """Play a single game between model idx1 (black) and idx2 (white).
        Returns (win, loss, draw) from perspective of idx1."""
        net1 = self.population[idx1]
        net2 = self.population[idx2]
        game_state = game.GomokuGame()
        mcts1 = mcts.MCTS(net1, n_simulations=config.MCTS_SIMULATIONS)
        mcts2 = mcts.MCTS(net2, n_simulations=config.MCTS_SIMULATIONS)

        while not game_state.is_terminal():
            if not game_state.move_history:
                # first move fixed in center
                move = (7, 7)
                game_state.make_move(move)
                continue

            if game_state.current_player == 1:
                # black's turn (net1)
                action_probs, move = mcts1.get_action_distribution(game_state)
            else:
                # white's turn (net2)
                action_probs, move = mcts2.get_action_distribution(game_state)

            if move is None:
                break
            game_state.make_move(move)

        winner = game_state.winner
        if winner == 1:
            return (1, 0, 0)  # black wins (idx1)
        elif winner == -1:
            return (0, 1, 0)  # white wins (idx2)
        else:
            return (0, 0, 1)  # draw

    def _play_match_with_data(self, idx1, idx2):
        """Play a single game between model idx1 (black) and idx2 (white).
        Returns (win, loss, draw) from perspective of idx1,
        and tuple (data1, data2) where each data is dict with keys:
        'states', 'policies', 'rewards' for that player."""
        net1 = self.population[idx1]
        net2 = self.population[idx2]
        game_state = game.GomokuGame()
        mcts1 = mcts.MCTS(net1, n_simulations=config.MCTS_SIMULATIONS)
        mcts2 = mcts.MCTS(net2, n_simulations=config.MCTS_SIMULATIONS)

        # data collection
        states1, policies1 = [], []  # for idx1 (black)
        states2, policies2 = [], []  # for idx2 (white)

        while not game_state.is_terminal():
            if not game_state.move_history:
                # first move fixed in center
                move = (7, 7)
                # No policy for first move (deterministic)
                game_state.make_move(move)
                continue

            state_tensor = game_state.get_state_tensor()
            if game_state.current_player == 1:
                # black's turn (idx1)
                action_probs, move = mcts1.get_action_distribution(game_state)
                if move is None:
                    break
                states1.append(state_tensor)
                policies1.append(action_probs)
            else:
                # white's turn (idx2)
                action_probs, move = mcts2.get_action_distribution(game_state)
                if move is None:
                    break
                states2.append(state_tensor)
                policies2.append(action_probs)

            game_state.make_move(move)

        winner = game_state.winner
        total_steps = len(game_state.move_history)
        # reward length factor (same as self_play)
        length_factor = np.log(max(total_steps - 10, 1)) * 0.3

        # compute rewards for each step
        rewards1 = []
        for i in range(len(states1)):
            # player of step is black (idx1)
            step_player = 1
            if winner == step_player:
                rewards1.append(1.0 * length_factor + 1)
            elif winner == 0:
                rewards1.append(0.9 * length_factor + 1)
            else:
                loss_factor = np.max((0.2, 1 - length_factor))
                rewards1.append(-1.0 * loss_factor)

        rewards2 = []
        for i in range(len(states2)):
            # player of step is white (idx2)
            step_player = -1
            if winner == step_player:
                rewards2.append(1.0 * length_factor + 1)
            elif winner == 0:
                rewards2.append(0.9 * length_factor + 1)
            else:
                loss_factor = np.max((0.2, 1 - length_factor))
                rewards2.append(-1.0 * loss_factor)

        # result from perspective of idx1
        if winner == 1:
            result = (1, 0, 0)  # black wins (idx1)
        elif winner == -1:
            result = (0, 1, 0)  # white wins (idx2)
        else:
            result = (0, 0, 1)  # draw

        data1 = {"states": states1, "policies": policies1, "rewards": rewards1}
        data2 = {"states": states2, "policies": policies2, "rewards": rewards2}
        return result, (data1, data2), game_state

    def competition_round(self):
        """Run matches between all pairs of models, update stats, and collect loss data for training.
        Returns the longest game played in this round (or None)."""
        n = len(self.population)
        if n <= 1:
            self.longest_game = None
            return None
        # clear previous game data
        for i in range(n):
            self.game_data[i].clear()

        longest_game = None
        longest_moves = 0

        # Play each pair GAMES_PER_ITER times (alternating colors).
        for i in range(n):
            for j in range(i + 1, n):
                sleep(0.7)
                # reduce history once per pair
                hist = 0.1
                self.wins[i] *= hist
                self.losses[i] *= hist
                self.draws[i] *= hist
                self.wins[j] *= hist
                self.losses[j] *= hist
                self.draws[j] *= hist

                for game_idx in range(config.GAMES_PER_ITER * 2):
                    print(f"Game {game_idx+1}", end="\r")

                    # Determine colors: even game_idx -> i black, j white; odd -> j black, i white
                    if game_idx % 2 == 0:
                        black_idx, white_idx = i, j
                    else:
                        black_idx, white_idx = j, i
                    result, (data_black, data_white), game_state = (
                        self._play_match_with_data(black_idx, white_idx)
                    )
                    win, loss, draw = result

                    # Update stats from perspective of i (since result is from black's perspective)
                    if black_idx == i:
                        # i is black
                        self.wins[i] += win
                        self.losses[i] += loss
                        self.draws[i] += draw
                        self.wins[j] += loss  # loss for i is win for j
                        self.losses[j] += win
                        self.draws[j] += draw
                        # store loss data
                        if loss == 1:  # i lost
                            self.game_data[i].append(data_black)
                        elif win == 1:  # j lost
                            self.game_data[j].append(data_white)
                    else:
                        # j is black
                        self.wins[j] += win
                        self.losses[j] += loss
                        self.draws[j] += draw
                        self.wins[i] += loss  # loss for j is win for i
                        self.losses[i] += win
                        self.draws[i] += draw
                        if loss == 1:  # j lost
                            self.game_data[j].append(data_black)
                        elif win == 1:  # i lost
                            self.game_data[i].append(data_white)
                    # draws: no loss data stored
                    # track longest game
                    moves = len(game_state.move_history)
                    if moves > longest_moves:
                        longest_moves = moves
                        longest_game = game_state

        self.longest_game = longest_game
        return longest_game

    def _update_best_index(self):
        """Update self.best_index to model with highest win rate."""
        n = len(self.population)
        if n <= 1:
            self.best_index = 0
            return
        total_games = [self.wins[i] + self.losses[i] + self.draws[i] for i in range(n)]
        best_idx = 0
        best_win_rate = -1
        for i in range(n):
            if total_games[i] == 0:
                continue
            win_rate = self.wins[i] / total_games[i]
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_idx = i
        self.best_index = best_idx

    def eliminate_weak_models(self):
        """Remove models that lose >= ELIMINATION_THRESHOLD of games and replace with random initialization or copy best."""
        n = len(self.population)
        if n <= 1:
            return
        total_games = [self.wins[i] + self.losses[i] + self.draws[i] for i in range(n)]
        to_eliminate = []
        for i in range(n):
            if total_games[i] == 0:
                continue
            loss_rate = self.losses[i] / total_games[i]
            if loss_rate >= config.ELIMINATION_THRESHOLD:
                to_eliminate.append(i)

        if not to_eliminate:
            return

        # update best model index
        self._update_best_index()
        best_idx = self.best_index

        # ensure main net (index 0) has the best weights
        if best_idx != 0:
            self.population[0].load_state_dict(self.population[best_idx].state_dict())
            # optionally reset stats for index 0? We keep as is.

        # replace eliminated models with random initialization
        for idx in to_eliminate:
            if idx == best_idx:
                continue
            print(f"Model {idx} eliminated", end=" and ")
            if random.random() > 0.7:
                # create a new randomly initialized model
                new_net = model.GomokuNet().to(config.DEVICE)
                self.population[idx].load_state_dict(new_net.state_dict())
                print("inited as random")
            else:
                self.population[idx].load_state_dict(
                    self.population[best_idx].state_dict()
                )
                print(f"copied the best (index {best_idx})")
            # reset stats for this model
            self.wins[idx] = 2
            self.losses[idx] = 1
            self.draws[idx] = 1
