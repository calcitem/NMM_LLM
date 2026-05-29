"""State encoder: BoardState -> 84-float tensor + phase id.

The encoder imports the canonical board representation from the
existing game engine — no board logic is duplicated here.

State vector layout (84 floats):
    [0:72)  24 positions x 3-way one-hot (empty / white / black), in POSITIONS order
    [72]    side-to-move (0.0 = white, 1.0 = black)
    [73:78) 5-way one-hot phase (see PHASE_NAMES)
    [78]    white_placed / 9
    [79]    black_placed / 9
    [80]    white_on_board / 9
    [81]    black_on_board / 9
    [82]    white_mills_formed / 3   (capped at 1.0)
    [83]    black_mills_formed / 3   (capped at 1.0)
"""

from __future__ import annotations

from typing import Tuple

import torch

from game.board import BoardState, MILLS, POSITIONS

STATE_DIM = 84
NUM_PHASES = 5

PHASE_OPENING_PLACEMENT = 0
PHASE_FULL_PLACEMENT = 1
PHASE_MIDGAME = 2
PHASE_ENDGAME = 3
PHASE_FLYING = 4

PHASE_NAMES = [
    "opening_placement",
    "full_placement",
    "midgame",
    "endgame",
    "flying",
]


def detect_phase(board: BoardState) -> int:
    """Return the 5-way phase id for the side to move."""
    stm = board.turn
    opp = "B" if stm == "W" else "W"
    placed_stm = board.pieces_placed[stm]
    on_stm = board.pieces_on_board[stm]
    on_opp = board.pieces_on_board[opp]
    placed_opp = board.pieces_placed[opp]

    if placed_stm < 9:
        if placed_stm < 4:
            return PHASE_OPENING_PLACEMENT
        return PHASE_FULL_PLACEMENT

    if placed_stm == 9 and on_stm <= 3:
        return PHASE_FLYING

    if placed_opp == 9 and on_opp <= 3:
        return PHASE_ENDGAME
    if on_stm <= 4:
        return PHASE_ENDGAME
    return PHASE_MIDGAME


def _count_mills(board: BoardState, color: str) -> int:
    return sum(
        1 for mill in MILLS if all(board.positions[p] == color for p in mill)
    )


def encode_state(board: BoardState) -> torch.Tensor:
    """Encode a BoardState into a flat 84-float tensor.

    Raises ValueError if the board is malformed.
    """
    if len(board.positions) != 24:
        raise ValueError(
            f"BoardState must have 24 positions, got {len(board.positions)}"
        )

    vec = torch.zeros(STATE_DIM, dtype=torch.float32)

    for idx, pos in enumerate(POSITIONS):
        val = board.positions[pos]
        base = idx * 3
        if val == "":
            vec[base + 0] = 1.0
        elif val == "W":
            vec[base + 1] = 1.0
        elif val == "B":
            vec[base + 2] = 1.0
        else:
            raise ValueError(f"Unexpected piece value {val!r} at {pos}")

    vec[72] = 0.0 if board.turn == "W" else 1.0

    phase = detect_phase(board)
    vec[73 + phase] = 1.0

    vec[78] = board.pieces_placed["W"] / 9.0
    vec[79] = board.pieces_placed["B"] / 9.0
    vec[80] = board.pieces_on_board["W"] / 9.0
    vec[81] = board.pieces_on_board["B"] / 9.0
    vec[82] = min(_count_mills(board, "W") / 3.0, 1.0)
    vec[83] = min(_count_mills(board, "B") / 3.0, 1.0)

    return vec


def encode_state_with_phase(board: BoardState) -> Tuple[torch.Tensor, int]:
    """Return (state_tensor, phase_id) together — convenient for batched code."""
    return encode_state(board), detect_phase(board)
