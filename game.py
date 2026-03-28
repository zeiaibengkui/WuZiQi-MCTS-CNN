"""
Gomoku game logic.
"""

import numpy as np
import torch

import config


class GomokuGame:
    def __init__(self):
        self.board = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=int)
        self.current_player = 1  # 1: black, -1: white
        self.move_history = []
        self.winner = None

    def get_valid_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def _line_properties(self, board, r, c, dr, dc, player):
        count = 1
        open_ends = 0

        # forward direction
        i = 1
        while True:
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < config.BOARD_SIZE and 0 <= nc < config.BOARD_SIZE):
                break
            if board[nr, nc] == player:
                count += 1
            elif board[nr, nc] == 0:
                open_ends += 1
                break
            else:
                break
            i += 1

        # backward direction
        i = 1
        while True:
            nr, nc = r - dr * i, c - dc * i
            if not (0 <= nr < config.BOARD_SIZE and 0 <= nc < config.BOARD_SIZE):
                break
            if board[nr, nc] == player:
                count += 1
            elif board[nr, nc] == 0:
                open_ends += 1
                break
            else:
                break
            i += 1

        return count, open_ends

    def evaluate_move(self, move, player):
        r, c = move
        if self.board[r, c] != 0:
            return None

        tmp = self.board.copy()
        tmp[r, c] = player

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            length, open_ends = self._line_properties(tmp, r, c, dr, dc, player)
            if length >= 5:
                return "five"
            if length == 4 and open_ends >= 1:
                return "open_four"
            if length == 3 and open_ends == 2:
                return "open_three"
        return None

    def find_critical_moves(self):
        cur = self.current_player
        opp = -cur

        win_moves = []
        block_moves = []
        open_four_moves = []
        open_three_moves = []

        for move in self.get_valid_moves():
            if self.evaluate_move(move, cur) == "five":
                win_moves.append(move)
            elif self.evaluate_move(move, opp) == "five":
                block_moves.append(move)
            elif self.evaluate_move(move, cur) == "open_four":
                open_four_moves.append(move)
            elif self.evaluate_move(move, cur) == "open_three":
                open_three_moves.append(move)

        return {
            "win": win_moves,
            "block": block_moves,
            "open_four": open_four_moves,
            "open_three": open_three_moves,
        }

    def get_priority_move(self):
        critical = self.find_critical_moves()
        if critical["win"]:
            return critical["win"][0]
        if critical["block"]:
            return critical["block"][0]
        # if critical["open_four"]:
        #     return critical["open_four"][0]
        # if critical["open_three"]:
        #     return critical["open_three"][0]
        return None

    def is_terminal(self):
        # check for five in a row
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(config.BOARD_SIZE):
            for c in range(config.BOARD_SIZE):
                if self.board[r, c] == 0:
                    continue
                player = self.board[r, c]
                for dr, dc in directions:
                    count = 1
                    for i in range(1, 5):
                        nr, nc = r + dr * i, c + dc * i
                        if (
                            0 <= nr < config.BOARD_SIZE
                            and 0 <= nc < config.BOARD_SIZE
                            and self.board[nr, nc] == player
                        ):
                            count += 1
                        else:
                            break
                    if count >= 5:
                        self.winner = player
                        return True
        # check for draw
        if len(self.move_history) == config.BOARD_SIZE * config.BOARD_SIZE:
            self.winner = 0
            return True
        return False

    def make_move(self, move):
        r, c = move
        if self.board[r, c] != 0:
            return False
        self.board[r, c] = self.current_player
        self.move_history.append((r, c, self.current_player))
        self.is_terminal()
        self.current_player = -self.current_player
        # self.print_board()
        return True

    def undo_move(self):
        if not self.move_history:
            return
        r, c, p = self.move_history.pop()
        self.board[r, c] = 0
        self.current_player = p
        self.winner = None

    def get_state_tensor(self):
        # neural network input: 2 channels (current player's stones, opponent's stones)
        # always from current player's perspective
        own_board = (self.board == self.current_player).astype(float)
        opp_board = (self.board == -self.current_player).astype(float)
        state = np.stack([own_board, opp_board], axis=0)
        return torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)

    def get_reward(self):
        if self.winner is None:
            return 0
        if self.winner == 0:
            return 0
        return (
            1 if self.winner == self.current_player else -1
        )  # reward relative to last player

    def print_board(self, show_move=False):
        print("\033[2J\033[H", end="", flush=True)
        board = self.board
        print(
            "    " + "".join(f"{chr(ord('a') + i):2}" for i in range(config.BOARD_SIZE))
        )
        for r in range(config.BOARD_SIZE):
            row_str = f"{r:2} "
            for c in range(config.BOARD_SIZE):
                if board[r, c] == 1:
                    row_str += " ●"
                elif board[r, c] == -1:
                    row_str += " ○"
                else:
                    row_str += " ."
            print(row_str)

        if not show_move:
            return
        print("Move order:")
        for step, (r, c, p) in enumerate(self.move_history):
            player = "black" if p == 1 else "white"
            print(f"Move {step+1}: {player} ({r}{chr(ord('a') + c)})")
        if self.winner == 0:
            print("Result: draw")
        elif self.winner == 1:
            print("Result: black wins")
        else:
            print("Result: white wins")
