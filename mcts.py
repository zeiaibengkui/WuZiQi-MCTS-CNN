"""
Monte Carlo Tree Search for Gomoku.
"""

import copy
import random
import math
import numpy as np
import torch

import config
import game
import model


class MCTSNode:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children: dict[tuple, MCTSNode] = {}
        self.n = 0  # visit count
        self.w = 0  # cumulative value
        self.p = 0  # prior probability (from NN)
        self.q = 0  # average value (w / n)

    def is_expanded(self):
        return len(self.children) > 0

    def best_child(self, c_puct=1.0):
        best_score = -float("inf")
        best_node = None
        for move, node in self.children.items():
            # UCT formula: Q + P * sqrt(N_parent) / (1 + N_child)
            uct = node.q + c_puct * node.p * np.sqrt(self.n) / (1 + node.n)
            if uct > best_score:
                best_score = uct
                best_node = node
        return best_node


class MCTS:
    def __init__(self, net, c_puct=config.C_PUCT, n_simulations=50):
        self.net = net
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.root: MCTSNode = MCTSNode()

    def search(self, game_state: game.GomokuGame):
        self.root = MCTSNode()
        for _ in range(self.n_simulations):
            game_copy = copy.deepcopy(game_state)
            node: MCTSNode = self.root
            path = []

            # 1. Selection
            while node.is_expanded() and not game_copy.is_terminal():
                node = node.best_child(self.c_puct)
                if node.move:
                    game_copy.make_move(node.move)
                path.append(node)

            # 2. Expansion & Evaluation
            if not game_copy.is_terminal():
                state = game_copy.get_state_tensor()
                with torch.no_grad():
                    policy, value = self.net(state)

                policy = policy.cpu().numpy()[0]
                value = value.cpu().numpy()[0][0]

                # only keep first move fixed in center; second move forced in center 5x5;
                # from move 3 onward, search neighbourhood of opponent's last move and own last move
                if len(game_copy.move_history) == 1:
                    # second move: center 5x5 area (5-9, 5-9)
                    valid_moves = [
                        move
                        for move in game_copy.get_valid_moves()
                        if 5 <= move[0] <= 9 and 5 <= move[1] <= 9
                    ]
                elif len(game_copy.move_history) >= 2:
                    last_r, last_c, _ = game_copy.move_history[-1]
                    prev_r, prev_c, _ = game_copy.move_history[-2]
                    focused = set()
                    radius = 2
                    for center_r, center_c in [(last_r, last_c), (prev_r, prev_c)]:
                        for dr in range(-radius, radius + 1):
                            for dc in range(-radius, radius + 1):
                                nr, nc = center_r + dr, center_c + dc
                                if (
                                    0 <= nr < config.BOARD_SIZE
                                    and 0 <= nc < config.BOARD_SIZE
                                    and game_copy.board[nr, nc] == 0
                                ):
                                    focused.add((nr, nc))
                    valid_moves = (
                        list(focused) if focused else game_copy.get_valid_moves()
                    )
                else:
                    valid_moves = game_copy.get_valid_moves()

                # filter illegal move probabilities
                policy_vec = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
                for r, c in valid_moves:
                    idx = r * config.BOARD_SIZE + c
                    policy_vec[idx] = policy[idx]

                # normalize
                s = np.sum(policy_vec)
                if s > 0:
                    policy_vec /= s

                for r, c in valid_moves:
                    idx = r * config.BOARD_SIZE + c
                    child = MCTSNode(parent=node, move=(r, c))
                    child.p = policy_vec[idx]
                    node.children[(r, c)] = child

                # if just expanded, directly backpropagate evaluation value
                if not node.children:
                    # no legal moves (draw or end)
                    pass
                else:
                    # simplified: if leaf node, use NN's value
                    # in AlphaZero, leaf node uses true reward if terminal, else V(s)
                    # for simplicity, use V(s) for first backup, corrected later
                    pass
            else:
                # game over
                value = (
                    1.0 if game_copy.winner == game_copy.current_player else -1.0
                )  # note perspective
                # correction: game_copy.current_player is next to move; if game over, winner is previous player
                # so if winner == previous player, value is +1 for that node (previous player's decision)
                if game_copy.winner == 0:
                    value = 0
                elif game_copy.winner == game_copy.move_history[-1][2]:
                    value = 1
                else:
                    value = -1

            # 3. Back propagation
            for node in reversed(path):
                node.n += 1
                node.w += value
                node.q = node.w / node.n
                value = -value  # zero-sum game, flip sign

    def get_action_distribution(self, game_state: game.GomokuGame, temperature=1):
        # priority move from game's critical moves (human vs AI will also go through this)
        priority_move = game_state.get_priority_move()
        if priority_move is not None:
            dist = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
            dist[priority_move[0] * config.BOARD_SIZE + priority_move[1]] = 1.0
            return dist, priority_move

        self.search(game_state)
        # generate policy from root children visit counts
        counts = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
        for move, node in self.root.children.items():
            idx = move[0] * config.BOARD_SIZE + move[1]
            counts[idx] = node.n

        if temperature == 0:
            # choose most visited
            best_move = max(self.root.children, key=lambda k: self.root.children[k].n)
            dist = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
            dist[best_move[0] * config.BOARD_SIZE + best_move[1]] = 1.0
            return dist, best_move
        else:
            # apply temperature
            counts = counts ** (1 / temperature)
            counts_sum = np.sum(counts)
            if counts_sum == 0:
                # no children (edge case), random choice
                valid = game_state.get_valid_moves()
                if not valid:
                    return None, None
                move = random.choice(valid)
                dist = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
                dist[move[0] * config.BOARD_SIZE + move[1]] = 1.0
                return dist, move

            probs = counts / counts_sum
            # sample
            move_idx = np.random.choice(len(probs), p=probs)
            move = (move_idx // config.BOARD_SIZE, move_idx % config.BOARD_SIZE)
            return probs, move
