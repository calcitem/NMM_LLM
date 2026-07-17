"""learned_ai/models/lookahead_advisor.py — 20-ply sentinel + human-DB lookahead.

For each legal move at the current position, simulates 20 half-plies using the
static heuristic for BOTH sides (no model calls, no recursion, no feedback loop).
At each depth, records 3 signals from the learner's perspective:

  h_norm    : (evaluate(board, learner_color) + 1) / 2  → [0, 1]
  sent_mean : mean sentinel score for current-player moves  (0.5 if disabled/unavailable)
              Flipped to 1 - mean when it is the opponent's turn, so the signal
              always expresses learner-perspective favourability.
  human_norm: confidence of the dominant human move from this position, derived from
              HumanDB (max move frequency across legal moves, 0.5 if no DB data).

The 20-depth × 3-signal = 60-float block is appended to the 62-float base features
by encode_position_with_lookahead(), producing the 122-float specialist input.
Total with top-K extras: 126 floats (unchanged from the previous 4-signal 15-ply design).

Value-net and gap-net signals have been removed: empirical ablation showed they
detract from play quality as direct heuristic modifiers.  VN_BLEND is set to 0 in
the base encoding so slots [59] and [61] carry pure heuristic signals.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal


def _static_best_move(board: BoardState, color: str, evaluate_fn) -> Optional[dict]:
    """Pick the move that maximises static heuristic eval for `color`."""
    moves = get_all_legal_moves(board)
    if not moves:
        return None
    best_score = -math.inf
    best_move  = moves[0]
    for mv in moves:
        try:
            after = board.apply_move(mv)
            score = float(evaluate_fn(after, color))
            if score > best_score:
                best_score = score
                best_move  = mv
        except Exception:
            pass
    return best_move


class LookaheadAdvisor:
    """N-ply heuristic lookahead scoring for each legal move.

    Returns a (k, ply_depth*3) ndarray — one row per candidate move —
    for use as the lookahead block in the specialist input.

    Parameters
    ----------
    sentinel      : SentinelAdvisor or None
    evaluate_fn   : callable(board, player) → float in [-1, 1]
    human_db      : HumanDB or None; .query_all_frequencies(board) → {ntn: prob}
    ngram_model   : NGramOpponentModel or None (reserved for future use)
    use_sentinel  : if True (default), sent_mean uses sentinel calls; False fills 0.5
    ply_depth     : number of half-plies to simulate (default 20)
    """

    def __init__(
        self,
        sentinel,
        evaluate_fn,
        human_db=None,
        ngram_model=None,
        use_sentinel: bool = True,
        endgame_db=None,
        ply_depth: int = 20,
        frozen_model=None,
        frozen_device=None,
        sim_ply_depth: Optional[int] = None,
        # Legacy params accepted but ignored — value_net and gap_net removed.
        value_net=None,
        gap_net=None,
    ) -> None:
        self._sentinel      = sentinel
        self._evaluate      = evaluate_fn
        self._human_db      = human_db
        self._ngram_model   = ngram_model
        self._use_sentinel  = use_sentinel
        self._endgame_db    = endgame_db
        self._ply_depth     = ply_depth      # output feature width = ply_depth * 3
        # sim_ply_depth: how many plies to ACTUALLY simulate.  Defaults to ply_depth.
        # Training scripts may set a smaller sim depth (e.g. 10) so simulations run
        # faster; remaining ply slots are filled with the last valid signal, so
        # the feature width matches inference (which uses full ply_depth).
        self._sim_ply_depth = int(sim_ply_depth) if sim_ply_depth is not None else ply_depth
        self._frozen_model  = frozen_model
        self._frozen_device = frozen_device
        self.feat_dim       = ply_depth * 3   # total floats per move row

    def set_frozen_model(self, model, device=None) -> None:
        """Update the frozen-model snapshot used to pick learner-side moves in lookahead."""
        self._frozen_model = model
        if device is not None:
            self._frozen_device = device

    def _frozen_model_best_move(self, board: BoardState, color: str):
        """Pick the frozen model's argmax move for ``color`` at ``board``.

        Uses encode_position_with_lookahead(..., lookahead_advisor=None) which zero-pads
        the 60-float lookahead block — avoids infinite recursion inside a simulation.
        Base 62 floats + zero lookahead is OOD but still policy-informative.
        Returns None on any failure (caller falls back to heuristic)."""
        if self._frozen_model is None:
            return None
        try:
            import torch
            from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead
            enc = encode_position_with_lookahead(
                board, color,
                sentinel_advisor=self._sentinel,
                db=None,
                value_net=None,
                lookahead_advisor=None,
            )
            if enc is None or not enc.legal_moves:
                return None
            device = self._frozen_device if self._frozen_device is not None else next(self._frozen_model.parameters()).device
            feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = self._frozen_model.policy_logits(feat_t)
                idx = int(torch.argmax(logits).item())
            return enc.legal_moves[idx]
        except Exception:
            return None

    def score_moves_matrix(
        self,
        board: BoardState,
        enc,
        learner_color: str,
        moves_subset=None,
    ) -> np.ndarray:
        """Return per-move lookahead feature block.  Shape: (k, ply_depth*4).

        Each row is the trajectory for one legal move.
        Returns a zero-filled matrix on any top-level error (neutral, no distortion).

        ``moves_subset`` — if provided, only simulate lookahead for those moves.
        Returns (len(moves_subset), ply_depth*4).  Used by encode_top_k_candidates
        to avoid running 15-ply simulations on the k-5 rejected candidates.
        """
        opp_color = "B" if learner_color == "W" else "W"
        target_moves = moves_subset if moves_subset is not None else enc.legal_moves
        k = len(target_moves)
        if k == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32)

        rows = []
        for mv in target_moves:
            row = self._simulate_trajectory(board, mv, learner_color, opp_color)
            rows.append(row)

        return np.stack(rows).astype(np.float32)   # (k, ply_depth*4)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _simulate_trajectory(
        self,
        board: BoardState,
        first_move: dict,
        learner_color: str,
        opp_color: str,
    ) -> np.ndarray:
        """Simulate self._ply_depth half-plies and return a (ply_depth*4,) float array.

        Half-plies alternate learner → opponent → learner → …
        Both sides always play the static-heuristic-best move.  first_move is
        the candidate move for half-ply 1 (the learner's choice being evaluated).
        On terminal or no-legal-moves, remaining depths are filled with the last
        valid signal (or the terminal score for all four channels).
        """
        # Layout: [h1, s1, u1,  h2, s2, u2, ...,  hN, sN, uN]  (h=heuristic, s=sentinel, u=human)
        result = np.full(self._ply_depth * 3, 0.5, dtype=np.float32)
        # Simulate only the first sim_ply_depth half-plies; remaining slots are
        # padded with the last valid signal.  Feature width is always ply_depth * 3.
        _sim = min(self._sim_ply_depth, self._ply_depth)
        try:
            b      = board
            actors = [learner_color if i % 2 == 0 else opp_color for i in range(_sim)]
            last_sig = (0.5, 0.5, 0.5)

            for depth_idx in range(_sim):
                actor = actors[depth_idx]

                # Apply the move for this half-ply
                if depth_idx == 0:
                    b = b.apply_move(first_move)
                else:
                    # Learner-side uses frozen policy (if provided), opponent uses heuristic.
                    mv = None
                    if self._frozen_model is not None and actor == learner_color:
                        mv = self._frozen_model_best_move(b, actor)
                    if mv is None:
                        mv = _static_best_move(b, actor, self._evaluate)
                    if mv is None:
                        # No legal moves — propagate last signal to remaining depths
                        for fill in range(depth_idx, self._ply_depth):
                            result[fill * 3 : fill * 3 + 3] = last_sig
                        return result
                    b = b.apply_move(mv)

                # Check for terminal
                terminal, winner = is_terminal(b)
                if terminal:
                    val = 1.0 if winner == learner_color else (0.0 if winner else 0.5)
                    for fill in range(depth_idx, self._ply_depth):
                        result[fill * 3 : fill * 3 + 3] = [val, val, val]
                    return result

                # Endgame DB probe — exact WDL terminates the trajectory early
                if self._endgame_db is not None:
                    try:
                        db_result = self._endgame_db.query(b)
                        if db_result is not None:
                            # db_result is from side-to-move's perspective ("W"/"L"/"D")
                            if b.turn == learner_color:
                                val = 1.0 if db_result == "W" else (0.0 if db_result == "L" else 0.5)
                            else:
                                val = 0.0 if db_result == "W" else (1.0 if db_result == "L" else 0.5)
                            for fill in range(depth_idx, self._ply_depth):
                                result[fill * 3 : fill * 3 + 3] = [val, val, val]
                            return result
                    except Exception:
                        pass

                # Record signals at this position
                sig = self._record_signals(b, learner_color)
                last_sig = sig
                result[depth_idx * 3 : depth_idx * 3 + 3] = sig

            # If we simulated fewer plies than the output width (training speed-up),
            # propagate the last valid signal into the remaining slots so the
            # feature width matches inference.
            if _sim < self._ply_depth:
                for fill in range(_sim, self._ply_depth):
                    result[fill * 3 : fill * 3 + 3] = last_sig

        except Exception:
            pass   # partial result already in `result`; remaining slots stay 0.5

        return result

    def _record_signals(
        self,
        board: BoardState,
        learner_color: str,
    ) -> tuple[float, float, float]:
        """Return (h_norm, sent_mean, human_norm) from the learner's perspective."""
        current_player = board.turn

        # ── heuristic ─────────────────────────────────────────────────────────
        h_norm = 0.5
        try:
            h = float(self._evaluate(board, learner_color))
            h_norm = max(0.0, min(1.0, (h + 1.0) / 2.0))
        except Exception:
            pass

        # ── sentinel ──────────────────────────────────────────────────────────
        sent_mean = 0.5
        if self._use_sentinel and self._sentinel is not None:
            try:
                legal = get_all_legal_moves(board)
                if legal:
                    advice = self._sentinel.advise(board, legal, current_player)
                    m = float(sum(advice.move_scores) / len(advice.move_scores))
                    sent_mean = max(0.0, min(1.0, m if current_player == learner_color else 1.0 - m))
            except Exception:
                pass

        # ── human DB ──────────────────────────────────────────────────────────
        human_norm = 0.5
        if self._human_db is not None:
            try:
                freqs = self._human_db.query_all_frequencies(board)
                if freqs:
                    human_norm = float(max(freqs.values()))
            except Exception:
                pass

        return h_norm, sent_mean, human_norm
