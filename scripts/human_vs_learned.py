"""scripts/human_vs_learned.py — play against the learned AI on the console.

Usage:
    python scripts/human_vs_learned.py [--checkpoint path] [--side W|B]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BOARD_REFERENCE, BoardState
from game.rules import does_form_mill, get_all_legal_moves, get_game_phase, is_terminal
from learned_ai.agents.learned_agent import LearnedAgent


def prompt_move(board: BoardState) -> dict:
    legal = get_all_legal_moves(board)
    phase = get_game_phase(board, board.turn)
    print(f"\n{board.to_display_grid()}")
    print(f"Phase: {phase}. Legal sample: {legal[:5]} ... ({len(legal)} total)")
    while True:
        raw = input("Your move (e.g. d2 / c5-c4 [/xCAPTURE]): ").strip().lower()
        capture = None
        if "x" in raw and "-" in raw:
            base, capture = raw.split("x", 1)
        elif raw.endswith("x") or "x" in raw and "-" not in raw:
            base, capture = raw.split("x", 1)
        else:
            base = raw
        capture = capture or None
        if "-" in base:
            src, dst = base.split("-", 1)
            move = {"from": src, "to": dst, "capture": capture}
        else:
            move = {"from": None, "to": base, "capture": capture}
        if move in legal:
            return move
        print("  Not a legal move. Try again.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--side", default="W", choices=["W", "B"])
    args = p.parse_args()

    human = args.side
    ai_color = "B" if human == "W" else "W"
    agent = LearnedAgent(color=ai_color, checkpoint_path=args.checkpoint, mode="argmax")

    print(BOARD_REFERENCE)
    print(f"You play {human}, the learned AI plays {ai_color}.\n")
    board = BoardState.new_game()
    while True:
        terminal, winner = is_terminal(board)
        if terminal:
            print(f"\nGame over. Winner: {winner}")
            return 0
        if board.turn == human:
            move = prompt_move(board)
        else:
            move = agent.choose_move(board)
            if not move:
                print("AI has no move.")
                return 0
            print(f"AI plays: {move}")
        if move.get("capture") is None and does_form_mill(board, {**move, "capture": None}):
            print("Mill formed but no capture supplied — choosing first legal.")
            move["capture"] = board.legal_captures(board.turn)[0]
        board = board.apply_move(move)


if __name__ == "__main__":
    raise SystemExit(main())
