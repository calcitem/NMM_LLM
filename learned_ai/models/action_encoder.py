"""Action encoder: move dict <-> integer index, plus legal-action masks.

Unified action space (size 624) so the network has a single output head per
phase but every move encodes uniquely regardless of phase:

    [0   :  24)   placement on POSITIONS[i]
    [24  : 600)   movement / fly from POSITIONS[src] to POSITIONS[dst]  (src*24+dst)
                   — diagonal entries (src == dst) are unused; they will simply
                     never appear in the legal mask
    [600 : 624)   capture POSITIONS[i]

A *complete* move includes its capture, so a placement / move that closes a
mill is represented by the placement/move index — the capture choice is then
encoded on the *next* network call after the partial move is applied? No:
that would couple inference to two passes per turn. Instead, since the
existing engine emits *atomic* moves (placement+capture or move+capture),
we encode the atomic move with a single index by picking the placement /
movement index. To recover the capture target the agent re-samples from the
capture slice using the same forward pass logits — keeping a single forward
pass per turn while still letting the network choose which piece to remove.

Concretely:
    encode_action(move, phase) returns a tuple of indices:
        (primary_index, capture_index_or_None)
    decode_action_indices(primary_index, capture_index, board) returns a
    complete move dict (with capture set when the partial move forms a mill).

The legal mask is built so that:
    * Placement / move primary indices are masked to only those that appear in
      the legal-move list (capture or not).
    * Capture indices are masked to those returned by board.legal_captures().
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from game.board import POSITIONS, BoardState
from game.rules import does_form_mill, get_all_legal_moves, get_game_phase

POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
PLACE_OFFSET = 0
MOVE_OFFSET = 24
CAPTURE_OFFSET = 24 + 24 * 24  # 600
ACTION_DIM = CAPTURE_OFFSET + 24  # 624

PLACE_SLICE = slice(PLACE_OFFSET, MOVE_OFFSET)
MOVE_SLICE = slice(MOVE_OFFSET, CAPTURE_OFFSET)
CAPTURE_SLICE = slice(CAPTURE_OFFSET, ACTION_DIM)


def _primary_index(move: dict) -> int:
    """Index of the placement/movement portion of a move (ignores capture)."""
    if move["from"] is None:
        return PLACE_OFFSET + POS_INDEX[move["to"]]
    src = POS_INDEX[move["from"]]
    dst = POS_INDEX[move["to"]]
    return MOVE_OFFSET + src * 24 + dst


def encode_action(move: dict) -> Tuple[int, Optional[int]]:
    """Return (primary_index, capture_index_or_None) for a complete move dict."""
    primary = _primary_index(move)
    cap = move.get("capture")
    cap_idx = CAPTURE_OFFSET + POS_INDEX[cap] if cap else None
    return primary, cap_idx


def decode_action(
    primary_index: int,
    board: BoardState,
    capture_index: Optional[int] = None,
) -> dict:
    """Convert (primary_index, capture_index) back into a complete move dict.

    The board is needed both to fill in any required capture (when the partial
    move closes a mill but no explicit capture was provided) and to validate
    the index range.
    """
    if not (0 <= primary_index < CAPTURE_OFFSET):
        raise ValueError(
            f"primary_index {primary_index} out of range for placement/movement"
        )

    if primary_index < MOVE_OFFSET:
        partial = {"from": None, "to": POSITIONS[primary_index]}
    else:
        rel = primary_index - MOVE_OFFSET
        src = rel // 24
        dst = rel % 24
        partial = {"from": POSITIONS[src], "to": POSITIONS[dst]}

    forms_mill = does_form_mill(board, {**partial, "capture": None})
    capture: Optional[str] = None
    if forms_mill:
        if capture_index is None:
            raise ValueError(
                "Move forms a mill but no capture_index was provided"
            )
        if not (CAPTURE_OFFSET <= capture_index < ACTION_DIM):
            raise ValueError(
                f"capture_index {capture_index} not in capture slice"
            )
        capture = POSITIONS[capture_index - CAPTURE_OFFSET]
    return {**partial, "capture": capture}


def get_legal_moves(board: BoardState) -> List[dict]:
    """Single source of truth for legal moves — defers to the existing engine."""
    return get_all_legal_moves(board)


def get_legal_mask(board: BoardState) -> torch.BoolTensor:
    """Build a (ACTION_DIM,) boolean mask of legal primary-and-capture indices.

    Primary slice (placement/movement) is set for every primary index that
    appears in *any* complete legal move. The capture slice is set for every
    opponent piece returned by ``board.legal_captures``. When no mill can be
    formed by the side to move on this turn, the capture slice is left all
    False — the agent uses ``move_requires_capture`` to know whether to read
    the capture logits.
    """
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    legal_moves = get_all_legal_moves(board)
    can_form_mill = False
    for mv in legal_moves:
        mask[_primary_index(mv)] = True
        if mv.get("capture") is not None:
            can_form_mill = True
    if can_form_mill:
        for sq in board.legal_captures(board.turn):
            mask[CAPTURE_OFFSET + POS_INDEX[sq]] = True
    return mask


def move_requires_capture(board: BoardState, primary_index: int) -> bool:
    """Return True if the partial move at primary_index would form a mill."""
    if primary_index < MOVE_OFFSET:
        partial = {"from": None, "to": POSITIONS[primary_index]}
    else:
        rel = primary_index - MOVE_OFFSET
        partial = {
            "from": POSITIONS[rel // 24],
            "to": POSITIONS[rel % 24],
        }
    return does_form_mill(board, {**partial, "capture": None})


# Phase-aware helpers ---------------------------------------------------------

def phase_action_slice(phase_id: int) -> slice:
    """Return the *primary* action slice that is meaningful for a given phase.

    Useful for evaluators / loggers that want to count primary picks by
    region. The mask is still the authoritative legality check.
    """
    from learned_ai.models.state_encoder import (
        PHASE_FLYING,
        PHASE_FULL_PLACEMENT,
        PHASE_MIDGAME,
        PHASE_OPENING_PLACEMENT,
    )

    if phase_id in (PHASE_OPENING_PLACEMENT, PHASE_FULL_PLACEMENT):
        return PLACE_SLICE
    return MOVE_SLICE


def board_phase_id(board: BoardState) -> int:
    """Convenience: map the board to the encoder's 5-way phase id."""
    from learned_ai.models.state_encoder import detect_phase

    return detect_phase(board)


def coarse_phase_name(board: BoardState) -> str:
    """Return the engine's coarse phase ('place'/'move'/'fly')."""
    return get_game_phase(board, board.turn)
