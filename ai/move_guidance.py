"""Shared helpers for opening book + trajectory DB move guidance (B-65)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ai.board_symmetry import transform_notation as _transform_book_notation
from ai.opening_book import Opening, OpeningBook
from ai.opening_recognizer import INACTIVE_RESULT, RecognitionResult
from game.rules import get_all_legal_moves, get_game_phase

if TYPE_CHECKING:
    from ai.endgame_db import EndgameDB
    from ai.endgame_recognizer import EndgameState
    from ai.game_ai import GameAI
    from ai.opening_recognizer import OpeningRecognizer
    from ai.trajectory_db import TrajectoryDB
    from game.board import BoardState


def pick_target_opening(
    book: OpeningBook,
    ai_color: str,
) -> tuple[Opening | None, int]:
    """UCB opening selection plus per-game D4 symmetry for White AI."""
    candidate = book.select_opening(ai_color=ai_color)
    if not candidate or candidate.side not in (ai_color, "both"):
        return None, 0
    sym_idx = 0
    if ai_color == "W" and candidate.line_moves:
        sym_idx = random.randint(0, 7)
    return candidate, sym_idx


def synthesize_opening_recognition(
    recognition: RecognitionResult,
    target_opening: Opening | None,
    board: BoardState,
    game_moves: list[dict],
    game_sym_idx: int = 0,
) -> RecognitionResult:
    """When still inactive/novel in placement, steer from the targeted opening."""
    phase = get_game_phase(board, board.turn)
    if (
        phase != "place"
        or target_opening is None
        or recognition.status not in ("inactive", "novel")
    ):
        return recognition

    ply = len(game_moves)
    line = target_opening.line_moves
    raw_mv = line[ply] if ply < len(line) else None
    book_mv = (
        (_transform_book_notation(raw_mv, game_sym_idx) or raw_mv)
        if raw_mv else None
    )
    if book_mv is None:
        return recognition

    legal_dests = {m["to"] for m in get_all_legal_moves(board)}
    if book_mv not in legal_dests:
        return recognition

    return RecognitionResult(
        opening_id=target_opening.opening_id,
        name=target_opening.name,
        family=target_opening.family,
        confidence=target_opening.confidence,
        status="probable",
        matched_ply=ply,
        deviation_ply=None,
        deviation_move=None,
        book_move=book_mv,
        branch_name=None,
        strategic_notes=target_opening.strategic_notes,
        common_blunders=list(target_opening.common_blunders),
        tags=list(target_opening.tags),
    )


def build_trajectory_hints(
    trajectory_db: TrajectoryDB | None,
    board: BoardState,
    game_moves: list[dict],
    game_ai: GameAI,
    endgame_db: EndgameDB | None = None,
    endgame_state: EndgameState | None = None,
) -> dict[str, float] | None:
    """Merge trajectory, opponent-loss, and endgame DB hints for choose_move()."""
    trajectory_hints: dict[str, float] | None = None

    if trajectory_db is not None:
        trajectory_hints = trajectory_db.query(board, board.turn) or None

        opp_color = "B" if board.turn == "W" else "W"
        loss_weight = (
            game_ai._weights.loss_exploit / 100.0
            if hasattr(game_ai, "_weights")
            else 1.5
        )
        if loss_weight > 0:
            exploit_hints = trajectory_db.query_opponent_loss(board, opp_color)
            if exploit_hints:
                if trajectory_hints:
                    for notation, delta in exploit_hints.items():
                        if notation in trajectory_hints:
                            trajectory_hints[notation] = (
                                trajectory_hints[notation] + loss_weight * delta
                            ) / (1 + loss_weight)
                        else:
                            trajectory_hints[notation] = (
                                delta * loss_weight / (1 + loss_weight)
                            )
                else:
                    trajectory_hints = {
                        n: d * loss_weight / (1 + loss_weight)
                        for n, d in exploit_hints.items()
                    }

    if endgame_db is not None and endgame_state is not None and endgame_state.active:
        eg_hints = endgame_db.query(board, board.turn)
        if eg_hints:
            if trajectory_hints:
                for notation, delta in eg_hints.items():
                    trajectory_hints[notation] = (
                        trajectory_hints.get(notation, 0.0) + delta
                    ) / 2.0
            else:
                trajectory_hints = eg_hints

    return trajectory_hints


def compute_force_book_early(
    board: BoardState,
    game_moves: list[dict],
    ai_color: str,
) -> bool:
    """Force book move for the AI's first two placements."""
    if get_game_phase(board, board.turn) != "place":
        return False
    ai_placements = sum(
        1 for m in game_moves
        if m.get("color") == ai_color and m.get("type") == "place"
    )
    return ai_placements < 2


def format_trajectory_context(trajectory_hints: dict[str, float] | None) -> str:
    """Human-readable trajectory summary for LLM prompts and server logs."""
    if not trajectory_hints:
        return ""
    items = sorted(trajectory_hints.items(), key=lambda kv: -abs(kv[1]))
    return "Trajectory DB: " + ", ".join(f"{n} {d:+.2f}" for n, d in items)


def build_choose_move_kwargs(
    board: BoardState,
    game_ai: GameAI,
    game_moves: list[dict],
    *,
    opening_recognizer: OpeningRecognizer | None = None,
    target_opening: Opening | None = None,
    game_sym_idx: int = 0,
    trajectory_db: TrajectoryDB | None = None,
    endgame_db: EndgameDB | None = None,
    endgame_state: EndgameState | None = None,
) -> dict:
    """Bundle keyword arguments for GameAI.choose_move()."""
    recognition = (
        opening_recognizer.get_current_result()
        if opening_recognizer else INACTIVE_RESULT
    )
    recognition = synthesize_opening_recognition(
        recognition, target_opening, board, game_moves, game_sym_idx,
    )
    notations = [m.get("notation", "") for m in game_moves if m.get("notation")]
    trajectory_hints = build_trajectory_hints(
        trajectory_db, board, game_moves, game_ai,
        endgame_db=endgame_db, endgame_state=endgame_state,
    )
    force_book_early = compute_force_book_early(board, game_moves, game_ai.color)
    return {
        "recognition": recognition,
        "endgame_state": endgame_state,
        "trajectory_hints": trajectory_hints,
        "force_book_early": force_book_early,
        "trajectory_context": format_trajectory_context(trajectory_hints),
        "trajectory_db": trajectory_db,      # SE-11: opponent frequency extension
        "game_notations": notations,         # SE-11: game history prefix for DB lookup
    }
