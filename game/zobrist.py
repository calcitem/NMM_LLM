"""
game/zobrist.py — Zobrist hashing for NMM board positions.

Keys are generated once at import time with a fixed seed so hashes are
reproducible across processes (needed if we ever serialise TT entries).

The canonical square ordering here MUST match game.board.POSITIONS exactly.
"""
from __future__ import annotations

import random as _rng

_rng.seed(0x9E3779B97F4A7C15)  # fixed golden-ratio seed

# Same ordering as game.board.POSITIONS (duplicated to avoid circular import).
_SQUARES = [
    "a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4",
    "b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4",
    "c5", "d5", "e5", "e4", "e3", "d3", "c3", "c4",
]

# Map square name → index (0–23).
SQ_INDEX: dict[str, int] = {sq: i for i, sq in enumerate(_SQUARES)}

# PIECE_KEYS[color_idx][sq_idx]: XOR in when piece of that color sits on sq.
# color_idx: 0 = W, 1 = B.
PIECE_KEYS: list[list[int]] = [
    [_rng.getrandbits(64) for _ in range(24)],
    [_rng.getrandbits(64) for _ in range(24)],
]

# PLACED_DONE_KEYS[color_idx]: XOR in when pieces_placed[color] transitions to ≥9.
PLACED_DONE_KEYS: list[int] = [_rng.getrandbits(64), _rng.getrandbits(64)]

# SIDE_KEY: XOR in when it is Black's turn to move (White-to-move == base state).
SIDE_KEY: int = _rng.getrandbits(64)


def hash_board(board) -> int:
    """Compute a full Zobrist hash from scratch for any BoardState.

    Called once by BoardState.new_game() and from_setup(); thereafter the hash
    is updated incrementally inside apply_move().
    """
    h = 0
    pos = board.positions
    for i, sq in enumerate(_SQUARES):
        piece = pos[sq]
        if piece == "W":
            h ^= PIECE_KEYS[0][i]
        elif piece == "B":
            h ^= PIECE_KEYS[1][i]
    if board.pieces_placed.get("W", 0) >= 9:
        h ^= PLACED_DONE_KEYS[0]
    if board.pieces_placed.get("B", 0) >= 9:
        h ^= PLACED_DONE_KEYS[1]
    if board.turn == "B":
        h ^= SIDE_KEY
    return h
