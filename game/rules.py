"""
game/rules.py — Legal move generation, phase logic, and terminal detection.

All functions are pure (no mutation).  The core helper get_all_legal_moves()
is the single source of truth for what moves are legal in any position; it is
used by both the game engine and the minimax AI.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from .board import MILLS, SQUARE_MILLS, BoardState


# ── Phase helpers ─────────────────────────────────────────────────────────────

def get_game_phase(board: BoardState, color: str) -> str:
    """
    Return the phase for a specific color:
      'place' — player still has pieces to place
      'fly'   — player has placed all 9 and has exactly 3 pieces remaining
      'move'  — player has placed all 9 and has 4+ pieces
    """
    if board.pieces_placed[color] < 9:
        return "place"
    if board.pieces_on_board[color] <= 3:
        return "fly"
    return "move"


def can_fly(board: BoardState, color: str) -> bool:
    return board.pieces_placed[color] == 9 and board.pieces_on_board[color] <= 3


def is_blocked(board: BoardState, color: str) -> bool:
    """
    True when color is in move phase (not fly, not place) and has no legal moves.
    A player who can fly is never blocked.
    """
    phase = get_game_phase(board, color)
    if phase != "move":
        return False
    return len(board.legal_moves(color)) == 0


# ── Terminal detection ────────────────────────────────────────────────────────

def is_terminal(board: BoardState) -> Tuple[bool, Optional[str]]:
    """
    Return (terminal, winner).
    winner is 'W' or 'B'; None is returned only when terminal is False.

    A player loses when:
    1. They have placed all 9 pieces and fewer than 3 remain on the board.
    2. It is their turn, they are in move phase, and they have no legal moves.
    """
    for color in ("W", "B"):
        if board.pieces_placed[color] == 9 and board.pieces_on_board[color] < 3:
            winner = "B" if color == "W" else "W"
            return True, winner

    current = board.turn
    if get_game_phase(board, current) == "move" and is_blocked(board, current):
        winner = "B" if current == "W" else "W"
        return True, winner

    return False, None


# ── Mill-formation check ──────────────────────────────────────────────────────

def does_form_mill(board: BoardState, move: dict) -> bool:
    """
    Return True if applying the placement/movement part of 'move' (ignoring
    any capture) would place board.turn's piece in a mill.

    Uses precomputed SQUARE_MILLS to check only the 2 mills containing the
    destination square, without constructing a temporary BoardState.
    """
    color = board.turn
    to  = move["to"]
    src = move["from"]
    pos = board.positions
    for mill in SQUARE_MILLS[to]:
        if all(p == to or (p != src and pos[p] == color) for p in mill):
            return True
    return False


# ── Complete legal move enumeration ──────────────────────────────────────────

def get_all_legal_moves(board: BoardState) -> List[dict]:
    """
    Return every complete legal move dict for board.turn.
    Each dict has the form: {"from": str|None, "to": str, "capture": str|None}

    Mill formation is expanded into all legal capture combinations so the
    minimax AI receives atomic, fully-specified moves.
    `legal_captures` is called at most once per call (result reused for all
    mill-forming moves, since capture options are the same regardless of which
    move formed the mill).
    """
    color = board.turn
    phase = get_game_phase(board, color)

    partial: List[dict] = []
    if phase == "place":
        for dest in board.legal_placements(color):
            partial.append({"from": None, "to": dest})
    else:
        for src, dest in board.legal_moves(color):
            partial.append({"from": src, "to": dest})

    complete: List[dict] = []
    captures: List[str] | None = None
    for pm in partial:
        if does_form_mill(board, pm):
            if captures is None:
                captures = board.legal_captures(color)
            for cap in captures:
                complete.append({**pm, "capture": cap})
        else:
            complete.append({**pm, "capture": None})

    return complete
