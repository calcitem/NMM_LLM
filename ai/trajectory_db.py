"""ai/trajectory_db.py — Board-state-first game trajectory memory.

ARCHITECTURE (v2):
  Primary key: canonical board-state key (position + turn + phase + placed
  counts, D4-normalised). Two boards reached by different move sequences share
  one DB bucket — transpositions are merged automatically.

  Index structure:
    _index[state_key][canon_notation] = {
        "wins_ai": int,    "losses_ai": int,    "draws_ai": int,
        "wins_human": int, "losses_human": int, "draws_human": int,
        "total": int,
        "reward_sum": float,  # accumulated per-move reward (Phase 3; 0.0 until then)
        "blame_sum":  float,  # accumulated per-move blame  (Phase 3; 0.0 until then)
    }

  Notations stored in canonical space (same D4 transform as the board key).
  query() maps them back to actual-game notation via the inverse transform.

  Replaces the v1 move-sequence-prefix index.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from ai.board_symmetry import (
    canonical_board_str as _canonical_board_str,
    transform_notation as _transform_notation,
    SYM_INVERSE as _SYM_INVERSE,
)
from game.board import POSITIONS

if TYPE_CHECKING:
    from game.board import BoardState

logger = logging.getLogger(__name__)

_UNICODE_X = "×"


def _norm(notation: str) -> str:
    return notation.replace(_UNICODE_X, "x")


def make_board_state_key(board: "BoardState") -> tuple[str, int]:
    """Return (canonical_state_key, sym_idx) for this board under D4 symmetry.

    sym_idx must be retained by callers: stored notations are in canonical
    space; query results are mapped back to actual-game notation via
    _SYM_INVERSE[sym_idx].

    Key components — all are necessary to distinguish game states:
      canon      — D4-canonical 24-char board layout
      turn       — side to move
      phase      — STM's individual phase (place/move/fly)
      placed_w   — cumulative W placements (encodes pieces still to place)
      placed_b   — cumulative B placements
      on_w       — W pieces currently on board (explicit: determines fly eligibility)
      on_b       — B pieces currently on board (explicit: determines fly eligibility)

    on_w / on_b are derivable from the canon string (count W's and B's), but
    making them explicit avoids relying on that derivation for correctness.
    Together with placed counts they fully encode pieces captured and flying
    eligibility for both sides without ambiguity.
    """
    from game.rules import get_game_phase
    board24 = "".join(board.positions.get(p, "") or "." for p in POSITIONS)
    canon, sym_idx = _canonical_board_str(board24)
    phase = get_game_phase(board, board.turn)
    placed_w = board.pieces_placed.get("W", 0)
    placed_b = board.pieces_placed.get("B", 0)
    on_w = board.pieces_on_board.get("W", 0)
    on_b = board.pieces_on_board.get("B", 0)
    return f"{canon}|{board.turn}|{phase}|{placed_w}|{placed_b}|{on_w}|{on_b}", sym_idx


def _smooth(values: list[float], window: int = 2) -> list[float]:
    """Causal moving average over a list of floats."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def _compute_blame_reward(
    boards: list[tuple],   # [(BoardState, color_who_moved), ...]
    winner: str | None,
) -> tuple[list[float], list[float]]:
    """Compute per-ply blame and reward weights for one game.

    Returns (blame_weights, reward_weights) both indexed by move position.
    blame  — how much this move contributed to the loser's defeat [0, 1]
    reward — how much this move contributed to the winner's victory [0, 1]
    Draws and games with < 10 moves get zeros throughout.
    """
    n = len(boards)
    blame  = [0.0] * n
    reward = [0.0] * n

    if winner not in ("W", "B") or n < 10:
        return blame, reward

    from ai.heuristics import evaluate

    loser = "B" if winner == "W" else "W"

    # Build raw strength-from-loser's-perspective for every ply.
    raw_loser = [evaluate(b, loser, strength_mode=True) for b, _ in boards]
    smooth_loser = _smooth(raw_loser, window=2)

    # Build raw strength-from-winner's-perspective for every ply.
    raw_winner = [evaluate(b, winner, strength_mode=True) for b, _ in boards]
    smooth_winner = _smooth(raw_winner, window=2)

    # ── Find turning point: loser's first ply where strength was ≥ -0.15
    #    and dropped ≥ 0.12 within the next 2 loser-plies, without recovery. ──
    loser_plies = [i for i, (_, c) in enumerate(boards) if c == loser]
    turning_idx: int | None = None

    for pos, ply in enumerate(loser_plies):
        s = smooth_loser[ply]
        if s < -0.20:
            continue   # already losing — not the turning point
        future = loser_plies[pos + 1 : pos + 4]
        if not future:
            continue
        if min(smooth_loser[p] for p in future) < s - 0.12:
            # Verify it doesn't recover above -0.10 afterwards
            rest = loser_plies[pos + 1:]
            if rest and max(smooth_loser[p] for p in rest) < -0.10:
                turning_idx = ply
                break
            elif not rest:
                turning_idx = ply
                break

    # ── Assign blame around the turning point ────────────────────────────────
    if turning_idx is not None:
        tp_pos = loser_plies.index(turning_idx)
        for delta, weight in ((0, 1.0), (-1, 0.6), (-2, 0.3), (1, 0.2)):
            target_pos = tp_pos + delta
            if 0 <= target_pos < len(loser_plies):
                idx = loser_plies[target_pos]
                blame[idx] = max(blame[idx], weight)

    # ── Assign reward to winner moves with meaningful strength gain ───────────
    winner_plies = [i for i, (_, c) in enumerate(boards) if c == winner]
    for i, ply in enumerate(winner_plies):
        if ply == 0:
            continue
        delta = smooth_winner[ply] - smooth_winner[ply - 1]
        if delta > 0.05:
            reward[ply] = min(1.0, delta * 4.0)
        # Late-game converting moves always earn a minimum reward
        if ply >= n - 8:
            reward[ply] = max(reward[ply], 0.4)

    return blame, reward


class TrajectoryDB:
    """
    In-memory index of historical game trajectories, keyed by canonical
    board state rather than move-sequence prefix.

    query() returns a per-move score delta (positive = historically good for
    the colour about to move) so the engine can boost moves that have won
    before and down-weight those that have lost.

    Confidence-weighted: low-sample positions return smaller deltas.
    """

    def __init__(self, games_dir: Path | str) -> None:
        self._games_dir = Path(games_dir)
        self._index: dict[str, dict[str, dict]] = {}
        self._game_count = 0

    # ── Build / update ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Index every *.jsonl file in the games directory from scratch."""
        self._index.clear()
        self._game_count = 0
        if not self._games_dir.exists():
            logger.warning("TrajectoryDB: games directory not found: %s", self._games_dir)
            return
        for path in sorted(self._games_dir.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._index_game(json.loads(line))
                except Exception as exc:
                    logger.debug("TrajectoryDB: skipping line in %s — %s", path.name, exc)
        logger.info(
            "TrajectoryDB: indexed %d games → %d state entries.",
            self._game_count, len(self._index),
        )

    def _index_game(self, record: dict) -> None:
        """Index a single game record by board-state key.

        Two-pass approach:
          Pass 1 — parse FENs, compute per-ply strength, derive blame/reward.
          Pass 2 — update the index with win/loss counts and blame/reward sums.
        """
        if record.get("adaptive_softened"):
            return
        winner = record.get("winner")
        moves = record.get("moves", [])
        if not moves:
            return

        source_type = record.get("source_type")
        if source_type is None:
            if record.get("self_play") or (
                record.get("white_difficulty") and record.get("black_difficulty")
                and not record.get("human_color")
            ):
                source_type = "ai_vs_ai"
            else:
                source_type = "human_involved"
        is_ai = (source_type == "ai_vs_ai")

        self._game_count += 1

        # ── Pass 1: parse boards, resolve keys ───────────────────────────────
        parsed: list[tuple] = []   # (board, state_key, sym_idx, canon_notation, color)
        boards: list          = []  # BoardState per parsed move (for strength computation)

        for move in moves:
            notation = _norm(move.get("notation", ""))
            fen = move.get("board_fen_before", "")
            if not notation or not fen:
                continue
            try:
                from game.board import BoardState
                board = BoardState.from_fen_string(fen)
            except Exception:
                continue

            state_key, sym_idx = make_board_state_key(board)
            canon_notation = _transform_notation(notation, sym_idx)
            if canon_notation is None:
                continue

            color = move.get("color", "W")
            parsed.append((board, state_key, sym_idx, canon_notation, color))
            boards.append((board, color))

        if not parsed:
            return

        # ── Compute blame/reward via inline position-strength evaluation ──────
        blame_wts, reward_wts = _compute_blame_reward(boards, winner)

        # ── Pass 2: update index ──────────────────────────────────────────────
        for i, (board, state_key, sym_idx, canon_notation, color) in enumerate(parsed):
            bucket = self._index.setdefault(state_key, {})
            entry = bucket.setdefault(canon_notation, {
                "wins_ai": 0, "losses_ai": 0, "draws_ai": 0,
                "wins_human": 0, "losses_human": 0, "draws_human": 0,
                "total": 0,
                "reward_sum": 0.0, "blame_sum": 0.0,
            })
            entry["total"] += 1
            entry["blame_sum"]  += blame_wts[i]
            entry["reward_sum"] += reward_wts[i]

            if winner == color:
                entry["wins_ai" if is_ai else "wins_human"] += 1
            elif winner is not None and winner != color:
                entry["losses_ai" if is_ai else "losses_human"] += 1
            else:
                entry["draws_ai" if is_ai else "draws_human"] += 1

    def add_game(self, record: dict) -> None:
        """Incrementally add one completed game without a full reload."""
        self._index_game(record)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        board: "BoardState",
        current_color: str,
        min_samples: int = 3,
        prefer_ai: bool = False,
    ) -> dict[str, float]:
        """Return a score-delta dict for candidate next-move notations.

        Positive delta  → move historically correlates with current_color winning
                          (max +0.5 when 100% win rate, 20+ samples).
        Negative delta  → correlates with a loss (min -0.5).
        Returns {}      when no data or fewer than min_samples total.

        Confidence-weighted: low-sample buckets return smaller deltas.
        Notations in canonical space are mapped back to actual-game notation
        via the inverse D4 transform before returning.
        """
        state_key, sym_idx = make_board_state_key(board)
        candidates = self._index.get(state_key)
        if not candidates:
            return {}

        inv = _SYM_INVERSE[sym_idx]
        result: dict[str, float] = {}

        for canon_notation, stats in candidates.items():
            total = stats["total"]
            if total < min_samples:
                continue

            actual_notation = _transform_notation(canon_notation, inv)
            if actual_notation is None:
                continue

            if prefer_ai:
                wins  = stats["wins_ai"]   + 0.5 * stats["wins_human"]
                draws = stats["draws_ai"]  + 0.5 * stats["draws_human"]
                eff   = max(1,
                    stats["wins_ai"] + stats["losses_ai"] + stats["draws_ai"]
                    + 0.5 * (stats["wins_human"] + stats["losses_human"] + stats["draws_human"])
                )
            else:
                wins  = stats["wins_ai"]  + stats["wins_human"]
                draws = stats["draws_ai"] + stats["draws_human"]
                eff   = max(1, total)

            win_rate = (wins + 0.4 * draws) / eff
            raw = win_rate - 0.5

            # Blend blame/reward signal; harmless (both 0) until Phase 3.
            avg_blame  = stats["blame_sum"]  / total
            avg_reward = stats["reward_sum"] / total
            adjusted = raw - avg_blame * 0.4 + avg_reward * 0.3

            # Confidence: reaches 1.0 at ~20 samples; shrinks delta for low-sample buckets.
            confidence = min(1.0, math.log(total + 1) / math.log(20))

            result[actual_notation] = max(-0.5, min(0.5, adjusted * confidence))

        return result

    def query_opponent_loss(
        self,
        board: "BoardState",
        opponent_color: str,
        min_samples: int = 3,
    ) -> dict[str, float]:
        """Score candidate moves by how often opponent_color loses from this position.

        In v2, state_key encodes board.turn so all stored moves at a key are by
        the same mover. "Opponent loses" = "current mover wins" — identical signal
        to query(). This method is kept for caller compatibility; it delegates to
        query() with the board mover as current_color.
        """
        return self.query(board, board.turn, min_samples=min_samples)

    def query_line(
        self,
        board: "BoardState",
        k: int = 4,
        min_samples: int = 3,
    ) -> list[tuple[str, float]]:
        """Top-k historically strong next moves sorted by score descending.

        Used by game_ai to promote high-trajectory moves to the front of the
        root move list before alpha-beta search, improving cut efficiency.
        Returns [(notation, score_delta), ...] — empty list when no data.
        """
        scores = self.query(board, board.turn, min_samples=min_samples)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def query_all_frequencies(
        self,
        board: "BoardState",
        min_samples: int = 5,
    ) -> dict[str, float]:
        """Return per-move relative frequency (0.0–1.0) for next moves at this board state.

        SE-11: used to identify commonly-played opponent replies so the search
        can extend by 1 ply for high-frequency (≥ 0.5) opponent moves.
        Returns {} when no data or fewer than min_samples total.
        """
        state_key, sym_idx = make_board_state_key(board)
        candidates = self._index.get(state_key)
        if not candidates:
            return {}

        total_all = sum(c["total"] for c in candidates.values())
        if total_all < min_samples:
            return {}

        inv = _SYM_INVERSE[sym_idx]
        result: dict[str, float] = {}
        for canon_n, c in candidates.items():
            if c["total"] == 0:
                continue
            actual_n = _transform_notation(canon_n, inv)
            if actual_n:
                result[actual_n] = c["total"] / total_all
        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def game_count(self) -> int:
        return self._game_count

    @property
    def entry_count(self) -> int:
        return len(self._index)
