"""
Human vs AI play interface.
"""

import re

import config
import game
import train
import mcts


def human_vs_ai():
    trainer = train.Trainer()
    trainer.load_model()
    # use trained model for play, no weight updates
    trainer.net.eval()
    mcts_engine = mcts.MCTS(trainer.net, n_simulations=config.MCTS_SIMULATIONS)

    g = game.GomokuGame()
    print("=== Gomoku Human vs AI (black ● moves first) ===")
    print("Input format: row number + column letter (e.g., 7h)")

    # let player choose color
    human_color = 1
    ai_color = -1
    choice = input("Choose first move (1) or second (-1): \nFirst move should be 7h")
    if choice == "-1":
        human_color = -1
        ai_color = 1
        print("AI plays black first.")
        # AI first move
        move = (7, 7)
        g.make_move(move)
        print(f"AI moves: {move[0]}{chr(ord('a') + move[1])}")
    else:
        print("You play black first.")

    g.print_board()

    while not g.is_terminal():
        if g.current_player == human_color:
            try:
                inp = input(
                    "(q to exit) Your turn (row number + column letter): "
                ).strip()
                if inp == "q":
                    print("Pressed Q. Exiting...\n")
                    return
                match = re.match(r"(\d+)([a-o])", inp)
                if not match:
                    print("Invalid format, please try again.")
                    continue
                r = int(match.group(1))
                c = ord(match.group(2)) - ord("a")
                if not (0 <= r < config.BOARD_SIZE and 0 <= c < config.BOARD_SIZE):
                    print("Coordinates out of range, please try again.")
                    continue
                if not g.make_move((r, c)):
                    print("Invalid move, please try again.")
                    continue
            except Exception as e:
                print(f"Input error: {e}")
                continue
        else:
            print("AI thinking...")
            action_probs, move = mcts_engine.get_action_distribution(g, temperature=0)
            if move:
                g.make_move(move)
                print(f"AI moves: {move[0]}{chr(ord('a') + move[1])}")
            else:
                print("AI resigns (no legal moves)")
                break

        g.print_board()

    if g.winner == 0:
        print("Draw!")
    elif g.winner == human_color:
        print("Congratulations, you win!")
    else:
        print("AI wins!")
