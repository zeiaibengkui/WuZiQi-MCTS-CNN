"""
Streamlit web interface for Gomoku AI training visualization and human vs AI play.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch
import pandas as pd
import time

# Import project modules
import config
import game
import train
import mcts

# Set page config
st.set_page_config(
    page_title="Gomoku AI Trainer",
    page_icon="⚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better board display
st.markdown("""
<style>
.board-container {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}
.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if "trainer" not in st.session_state:
        st.session_state.trainer = None
    if "training_active" not in st.session_state:
        st.session_state.training_active = False
    if "training_thread" not in st.session_state:
        st.session_state.training_thread = None
    if "game_state" not in st.session_state:
        st.session_state.game_state = None
    if "ai_color" not in st.session_state:
        st.session_state.ai_color = 1  # AI plays black by default
    if "human_color" not in st.session_state:
        st.session_state.human_color = -1
    if "move_history" not in st.session_state:
        st.session_state.move_history = []
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []  # list of dicts per iteration
    if "iteration" not in st.session_state:
        st.session_state.iteration = 0
    if "longest_game_length" not in st.session_state:
        st.session_state.longest_game_length = 0

def load_trainer():
    """Load or create trainer and model."""
    if st.session_state.trainer is None:
        with st.spinner("Initializing trainer and loading model..."):
            trainer = train.Trainer()
            trainer.load_model()
            st.session_state.trainer = trainer
            st.session_state.longest_game_length = trainer.longest_game_length
            st.success("Trainer initialized and model loaded.")
    return st.session_state.trainer

def plot_board(board, highlight_last_move=None):
    """Create a matplotlib visualization of the Gomoku board."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    for i in range(config.BOARD_SIZE):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)

    # Set limits and aspect
    ax.set_xlim(-0.5, config.BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, config.BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # So row 0 is at top

    # Draw stones
    for r in range(config.BOARD_SIZE):
        for c in range(config.BOARD_SIZE):
            if board[r, c] == 1:  # black
                ax.add_patch(Circle((c, r), 0.4, color='black', zorder=2))
            elif board[r, c] == -1:  # white
                ax.add_patch(Circle((c, r), 0.4, color='white', ec='black', linewidth=1, zorder=2))

    # Highlight last move
    if highlight_last_move:
        r, c = highlight_last_move
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  fill=False, edgecolor='red', linewidth=2, zorder=3))

    # Add coordinates
    ax.set_xticks(range(config.BOARD_SIZE))
    ax.set_yticks(range(config.BOARD_SIZE))
    ax.set_xticklabels([chr(ord('A') + i) for i in range(config.BOARD_SIZE)])
    ax.set_yticklabels([str(i) for i in range(config.BOARD_SIZE)])
    ax.grid(True, alpha=0.3)
    ax.set_title("Gomoku Board")

    return fig

def get_metrics_df(trainer):
    """Convert trainer stats to pandas DataFrame for visualization."""
    metrics = []
    for i in range(len(trainer.population)):
        total = trainer.wins[i] + trainer.losses[i] + trainer.draws[i]
        if total > 0:
            win_rate = trainer.wins[i] / total
            loss_rate = trainer.losses[i] / total
            draw_rate = trainer.draws[i] / total
        else:
            win_rate = loss_rate = draw_rate = 0.0

        metrics.append({
            "Model": f"Model {i}",
            "Win Rate": win_rate,
            "Loss Rate": loss_rate,
            "Draw Rate": draw_rate,
            "Total Games": total,
            "Is Best": i == trainer.best_index
        })
    return pd.DataFrame(metrics)

def render_training_dashboard():
    """Render the training dashboard tab."""
    st.header("Training Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Load/Initialize Trainer", type="primary"):
            load_trainer()
    with col2:
        if st.button("Run One Training Iteration"):
            trainer = load_trainer()
            with st.spinner("Running training iteration..."):
                trainer.train_step()
                st.session_state.iteration += 1
                # Record metrics
                metrics = get_metrics_df(trainer)
                st.session_state.metrics_history.append({
                    "iteration": st.session_state.iteration,
                    "metrics": metrics,
                    "longest_game": trainer.longest_game_length
                })
            st.success(f"Iteration {st.session_state.iteration} completed.")
            st.rerun()
    with col3:
        if st.button("Reset Training"):
            st.session_state.trainer = None
            st.session_state.metrics_history = []
            st.session_state.iteration = 0
            st.success("Trainer reset.")
            st.rerun()

    # Batch training
    with st.expander("Batch Training", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            batch_size = st.slider("Number of iterations", 1, 100, 10, 1,
                                   help="Run this many training iterations sequentially")
        with col2:
            batch_delay = st.slider("Delay (s)", 0.0, 5.0, 0.5, 0.1,
                                    help="Seconds between iterations")
        with col3:
            st.write(" ")  # spacing
            if st.button("Run Batch", type="secondary"):
                trainer = load_trainer()
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(batch_size):
                    status_text.text(f"Iteration {st.session_state.iteration + 1 + i}/{batch_size}")
                    trainer.train_step()
                    st.session_state.iteration += 1
                    # Record metrics
                    metrics = get_metrics_df(trainer)
                    st.session_state.metrics_history.append({
                        "iteration": st.session_state.iteration,
                        "metrics": metrics,
                        "longest_game": trainer.longest_game_length
                    })
                    progress_bar.progress((i + 1) / batch_size)
                    time.sleep(batch_delay)

                progress_bar.empty()
                status_text.text(f"Batch completed. Total iterations: {st.session_state.iteration}")
                st.success(f"Batch of {batch_size} iterations completed.")
                st.rerun()

    # Display trainer status
    trainer = st.session_state.trainer
    if trainer is None:
        st.info("No trainer loaded. Click 'Load/Initialize Trainer' to start.")
        return

    st.subheader("Current Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Iteration", st.session_state.iteration)
    with col2:
        st.metric("Longest Game", trainer.longest_game_length)
    with col3:
        best_idx = trainer.best_index
        total = trainer.wins[best_idx] + trainer.losses[best_idx] + trainer.draws[best_idx]
        win_rate = trainer.wins[best_idx] / total if total > 0 else 0
        st.metric("Best Model Win Rate", f"{win_rate:.2%}", f"Model {best_idx}")

    # Model metrics table
    st.subheader("Model Performance")
    metrics_df = get_metrics_df(trainer)
    st.dataframe(metrics_df.style.highlight_max(subset=["Win Rate"]), use_container_width=True)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win Rate Evolution")
        if st.session_state.metrics_history:
            # Prepare data for line chart
            win_rates_history = []
            iterations = []
            for hist in st.session_state.metrics_history:
                iterations.append(hist["iteration"])
                win_rates = []
                for _, row in hist["metrics"].iterrows():
                    win_rates.append(row["Win Rate"])
                win_rates_history.append(win_rates)

            # Plot each model's win rate over time
            fig, ax = plt.subplots(figsize=(8, 4))
            for model_idx in range(len(trainer.population)):
                model_rates = [rates[model_idx] for rates in win_rates_history]
                ax.plot(iterations, model_rates, label=f"Model {model_idx}", marker='o', markersize=3)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Win Rate")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Run at least one training iteration to see evolution.")

    with col2:
        st.subheader("Longest Game Progress")
        if st.session_state.metrics_history:
            longest_games = [h["longest_game"] for h in st.session_state.metrics_history]
            iterations = [h["iteration"] for h in st.session_state.metrics_history]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(iterations, longest_games, marker='o', markersize=3, color='green')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Longest Game Length")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Run at least one training iteration to see progress.")

    st.subheader("Longest Game Board")
    if trainer.longest_game is not None:
        fig = plot_board(trainer.longest_game.board)
        st.pyplot(fig)
        st.caption(f"Game length: {len(trainer.longest_game.move_history)} moves. Winner: {trainer.longest_game.winner}")
    else:
        st.info("No longest game recorded from competition.")

def render_play_interface():
    """Render the human vs AI play interface."""
    st.header("Human vs AI Play")

    # Initialize game state
    if st.session_state.game_state is None:
        st.session_state.game_state = game.GomokuGame()
        st.session_state.move_history = []

    # Sidebar controls
    with st.sidebar:
        st.subheader("Game Settings")

        # Load trainer for AI
        trainer = load_trainer()

        # Color selection
        color_option = st.radio("You play as:", ["Black (●) - First", "White (○) - Second"])
        if color_option == "Black (●) - First":
            st.session_state.human_color = 1
            st.session_state.ai_color = -1
        else:
            st.session_state.human_color = -1
            st.session_state.ai_color = 1

        # AI strength
        simulations = st.slider("MCTS Simulations", 10, 300, config.MCTS_SIMULATIONS, 10)

        # Game controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Game"):
                st.session_state.game_state = game.GomokuGame()
                st.session_state.move_history = []
                if st.session_state.human_color == -1:
                    # AI moves first (center)
                    st.session_state.game_state.make_move((7, 7))
                    st.session_state.move_history.append((7, 7, st.session_state.ai_color))
                st.success("New game started!")
                st.rerun()
        with col2:
            if st.button("Undo Last Move"):
                if st.session_state.move_history:
                    st.session_state.game_state.undo_move()
                    st.session_state.move_history.pop()
                    # If AI made the last move, undo human move as well for fairness
                    if len(st.session_state.move_history) > 0 and \
                       st.session_state.move_history[-1][2] == st.session_state.ai_color:
                        st.session_state.game_state.undo_move()
                        st.session_state.move_history.pop()
                    st.rerun()

        # Display move history
        st.subheader("Move History")
        if st.session_state.move_history:
            for i, (r, c, player) in enumerate(st.session_state.move_history):
                player_str = "Black (●)" if player == 1 else "White (○)"
                move_str = f"{r}{chr(ord('a') + c)}"
                st.write(f"{i+1}. {player_str}: {move_str}")
        else:
            st.write("No moves yet.")

    # Main game board
    col1, col2 = st.columns([3, 1])
    with col1:
        # Draw board
        board = st.session_state.game_state.board
        last_move = None
        if st.session_state.move_history:
            last_move = st.session_state.move_history[-1][:2]

        fig = plot_board(board, last_move)
        st.pyplot(fig)

        # Game status
        if st.session_state.game_state.is_terminal():
            winner = st.session_state.game_state.winner
            if winner == 0:
                st.success("Game over: Draw!")
            elif winner == st.session_state.human_color:
                st.success("🎉 Congratulations, you win!")
            else:
                st.info("AI wins!")
        else:
            current_player = st.session_state.game_state.current_player
            if current_player == st.session_state.human_color:
                st.info("Your turn (●)" if current_player == 1 else "Your turn (○)")
            else:
                st.info("AI thinking...")

    with col2:
        # Interactive board for moves
        st.subheader("Make a Move")

        if not st.session_state.game_state.is_terminal():
            if st.session_state.game_state.current_player == st.session_state.human_color:
                # Create a grid of buttons for moves
                st.write("Click on a coordinate:")

                # Use a form to capture click
                with st.form("move_form"):
                    cols = st.columns(config.BOARD_SIZE)
                    clicked_move = None

                    # Create button grid
                    for c in range(config.BOARD_SIZE):
                        with cols[c]:
                            for r in range(config.BOARD_SIZE):
                                if board[r, c] == 0:
                                    if st.button(f"{r}{chr(ord('a') + c)}", key=f"{r}_{c}"):
                                        clicked_move = (r, c)

                    submitted = st.form_submit_button("Make Move")
                    if submitted and clicked_move is not None:
                        if st.session_state.game_state.make_move(clicked_move):
                            st.session_state.move_history.append(
                                (clicked_move[0], clicked_move[1], st.session_state.human_color)
                            )
                            st.rerun()
                        else:
                            st.error("Invalid move!")
            else:
                # AI's turn
                if st.button("AI Move"):
                    with st.spinner("AI thinking..."):
                        # Create MCTS engine with current best model
                        trainer.net.eval()
                        mcts_engine = mcts.MCTS(trainer.net, n_simulations=simulations)
                        _, move = mcts_engine.get_action_distribution(
                            st.session_state.game_state, temperature=0
                        )
                        if move:
                            st.session_state.game_state.make_move(move)
                            st.session_state.move_history.append(
                                (move[0], move[1], st.session_state.ai_color)
                            )
                            st.rerun()
                        else:
                            st.error("AI resigns (no legal moves)")
        else:
            st.write("Game over. Start a new game to play again.")

def main():
    """Main app function."""
    st.title("⚫ Gomoku AI Trainer")
    st.markdown("""
    This interactive dashboard lets you:
    - **Visualize** the training process of a Gomoku AI using Monte Carlo Tree Search (MCTS) with parallel model competition
    - **Play** against the trained AI in real-time
    """)

    # Initialize session state
    init_session_state()

    # Create tabs
    tab1, tab2 = st.tabs(["Training Dashboard", "Human vs AI Play"])

    with tab1:
        render_training_dashboard()

    with tab2:
        render_play_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
    **About**: This project implements Gomoku AI training inspired by AlphaZero,
    using Monte Carlo Tree Search with a convolutional neural network.
    Training is competition-based: parallel models compete, losers are trained, winner weights are frozen.
    """)

if __name__ == "__main__":
    main()