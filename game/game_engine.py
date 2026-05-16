"""
game/game_engine.py — Game loop, turn management, win conditions, and game record.

GameEngine manages a single game session.  It is UI-agnostic: it holds state
and validates moves, but does not perform I/O.  The __main__ block at the
bottom provides a minimal console harness for Stage 1 testing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from .board import BoardState, BOARD_REFERENCE
from .notation import encode_move, export_pgn_style
from .rules import (
    does_form_mill,
    get_all_legal_moves,
    get_game_phase,
    is_terminal,
)


class GameEngine:
    """
    Manages a full Nine Men's Morris game session.

    Typical usage:
        engine = GameEngine()
        while not engine.finished:
            move = ...get move from UI or AI...
            engine.apply_move(move)

    A 'move' dict always has the form:
        {"from": str|None, "to": str, "capture": str|None}
    """

    # Half-moves after placement before the draw-offer button unlocks.
    DRAW_OFFER_THRESHOLD = 40
    # Half-moves after placement before automatic draw (50 per player).
    AUTO_DRAW_THRESHOLD  = 100

    def __init__(self, human_color: str = "W") -> None:
        self.board: BoardState = BoardState.new_game()
        self.human_color: str = human_color
        self.finished: bool = False
        self.winner: Optional[str] = None
        self.draw_reason: Optional[str] = None
        self._turn_num: int = 1  # pair number (White + Black = 1 turn)
        self._post_placement_moves: int = 0   # half-moves after placement ends
        self._move_log: List[Tuple] = []       # (color, from, to) per half-move

        self.game_record: Dict = {
            "session_id": str(uuid.uuid4()),
            "date": datetime.now().isoformat(),
            "human_color": human_color,
            "winner": None,
            "draw_reason": None,
            "recognised_opening_id": None,
            "recognised_opening_name": None,
            "opening_recognition_status": None,
            "opening_deviation_ply": None,
            "moves": [],
            "bad_moves_taught": [],
            "llm_summary": None,
        }

    # ── Move application ──────────────────────────────────────────────────────

    def apply_move(self, move: dict) -> None:
        """
        Apply a validated, complete move dict and update all internal state.
        Raises ValueError if the game is already finished.
        """
        if self.finished:
            raise ValueError("Game is already over.")

        color = self.board.turn
        phase = get_game_phase(self.board, color)
        notation = encode_move(move, phase)

        record_entry: Dict = {
            "turn": self._turn_num,
            "color": color,
            "type": phase,
            "from": move.get("from"),
            "to": move["to"],
            "capture": move.get("capture"),
            "notation": notation,
            "board_fen_before": self.board.to_fen_string(),
            "game_ai_score": None,
            "llm_opinion": None,
            "human_feedback": None,
            "llm_poor_move_comment": None,
            "score_delta": None,
            "opening_recognition": None,
        }

        self.game_record["moves"].append(record_entry)
        self.board = self.board.apply_move(move)

        # Advance pair counter after Black moves
        if color == "B":
            self._turn_num += 1

        # Check for game end
        terminal, winner = is_terminal(self.board)
        if terminal:
            self.finished = True
            self.winner = winner
            self.game_record["winner"] = winner

        # Draw checks (only when game not already finished by terminal condition)
        if not self.finished:
            placement_done = (
                self.board.pieces_placed.get("W", 0) >= 9
                and self.board.pieces_placed.get("B", 0) >= 9
            )
            if placement_done:
                self._post_placement_moves += 1

            # Store (color, from, to, capture) so the repetition check can exclude
            # moves that changed material — a capture can never be part of repetition.
            self._move_log.append(
                (color, move.get("from"), move.get("to"), move.get("capture"))
            )

            # Threefold repetition: the SAME BOARD POSITION occurred 3 times.
            # This requires BOTH players to oscillate over 6 half-moves (3 per player):
            #   Player A: A→B  Player B: C→D
            #   Player A: B→A  Player B: D→C   (position repeats)
            #   Player A: A→B  Player B: C→D   (position repeats a 3rd time)
            # A draw where only one player oscillates is NOT threefold repetition —
            # the other player's different moves change the board state each cycle.
            log = self._move_log
            if len(log) >= 6:
                no_captures = all(log[i][3] is None for i in (-1, -2, -3, -4, -5, -6))
                if no_captures:
                    c1, f1, t1, _ = log[-1]
                    c3, f3, t3, _ = log[-3]
                    c5, f5, t5, _ = log[-5]
                    c2, f2, t2, _ = log[-2]
                    c4, f4, t4, _ = log[-4]
                    c6, f6, t6, _ = log[-6]
                    osc_a = (
                        c1 == c3 == c5 and f1 is not None
                        and f1 == t3 and t1 == f3
                        and f1 == f5 and t1 == t5
                    )
                    osc_b = (
                        c2 == c4 == c6 and f2 is not None
                        and f2 == t4 and t2 == f4
                        and f2 == f6 and t2 == t6
                    )
                    if osc_a and osc_b:
                        self.finished = True
                        self.winner = None
                        self.draw_reason = "repetition"
                        self.game_record["draw_reason"] = "repetition"

            # 50-move rule (100 half-moves post-placement without capture)
            if not self.finished and self._post_placement_moves >= self.AUTO_DRAW_THRESHOLD:
                self.finished = True
                self.winner = None
                self.draw_reason = "50-move rule"
                self.game_record["draw_reason"] = "50-move rule"

    # ── Helpers used by the UI / AI ───────────────────────────────────────────

    def get_all_legal_moves(self) -> List[dict]:
        """Return every complete legal move for the current player."""
        return get_all_legal_moves(self.board)

    def move_forms_mill(self, partial_move: dict) -> bool:
        """
        Check whether placing/moving to partial_move['to'] would form a mill,
        requiring the current player to choose a capture.
        partial_move need not include a 'capture' key.
        """
        return does_form_mill(self.board, {**partial_move, "capture": None})

    def status_line(self) -> str:
        b = self.board
        phase = get_game_phase(b, b.turn)
        placed = b.pieces_placed[b.turn]
        return (
            f"Turn: {'White' if b.turn == 'W' else 'Black'} | "
            f"Phase: {phase} | "
            f"Placed — W:{b.pieces_placed['W']}/9  B:{b.pieces_placed['B']}/9 | "
            f"On board — W:{b.pieces_on_board['W']}  B:{b.pieces_on_board['B']}"
        )

    def export(self) -> str:
        """Return the full game in PGN-style notation."""
        return export_pgn_style(self.game_record)


# ── Console harness (Stage 1 / Stage 2 testing) ───────────────────────────────

def _prompt_placement(engine: GameEngine) -> dict:
    board = engine.board
    color = board.turn
    legal = board.legal_placements(color)
    while True:
        raw = input("  Place piece at (e.g. d2): ").strip().lower()
        if raw not in legal:
            print(f"  ! '{raw}' is not a legal placement. Legal: {sorted(legal)}")
            continue
        return {"from": None, "to": raw, "capture": None}


def _prompt_movement(engine: GameEngine) -> dict:
    board = engine.board
    color = board.turn
    phase = get_game_phase(board, color)
    legal_pairs = board.legal_moves(color)
    legal_srcs = sorted({s for s, _ in legal_pairs})
    while True:
        raw = input(
            f"  {'Fly' if phase == 'fly' else 'Move'} piece (e.g. c5-c4): "
        ).strip().lower()
        if "-" not in raw:
            print("  ! Format must be src-dst, e.g. c5-c4")
            continue
        src, dst = raw.split("-", 1)
        if (src, dst) not in legal_pairs:
            print(f"  ! '{raw}' is not a legal move.")
            print(f"  Legal sources: {legal_srcs}")
            continue
        return {"from": src, "to": dst, "capture": None}


def _prompt_capture(engine: GameEngine) -> str:
    board = engine.board
    color = board.turn
    legal = board.legal_captures(color)
    print(f"  Mill formed! Choose an opponent piece to capture.")
    print(f"  Legal captures: {sorted(legal)}")
    while True:
        raw = input("  Capture: ").strip().lower()
        if raw not in legal:
            print(f"  ! '{raw}' is not a legal capture.")
            continue
        return raw


def run_console_game(human_color: str = "W", vs_human: bool = True) -> None:
    """
    Play a Human vs Human game on the console.
    Pass vs_human=False to later plug in an AI for one side.
    """
    engine = GameEngine(human_color=human_color)
    board = engine.board

    print("\n═══ Nine Men's Morris — Console ═══\n")
    print("Board position reference:")
    print(BOARD_REFERENCE)
    print("\nEach player places 9 pieces. First to form a mill of 3 may capture.\n")
    print("Notation: placement → 'd2'   move → 'c5-c4'   (captures are prompted)\n")

    while not engine.finished:
        board = engine.board
        color = board.turn
        name = "White (W)" if color == "W" else "Black (B)"
        phase = get_game_phase(board, color)

        print("\n" + engine.status_line())
        print(board.to_display_grid())
        print(f"\n{name}'s turn  [{phase}]")

        # --- Get move from player ---
        if phase == "place":
            move = _prompt_placement(engine)
        else:
            move = _prompt_movement(engine)

        # --- Check for mill and prompt capture ---
        if engine.move_forms_mill(move):
            cap = _prompt_capture(engine)
            move["capture"] = cap

        engine.apply_move(move)

    # --- Game over ---
    board = engine.board
    print("\n" + board.to_display_grid())
    winner_name = "White" if engine.winner == "W" else "Black"
    print(f"\n{'═'*40}")
    print(f"  Game over — {winner_name} wins!")
    print(f"{'═'*40}\n")
    print(engine.export())


if __name__ == "__main__":
    run_console_game()
