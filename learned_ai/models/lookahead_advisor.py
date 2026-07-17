"""learned_ai/models/lookahead_advisor.py — 12-ply 5-signal lookahead for specialists.

For each legal move, simulates up to 12 half-plies and records 5 signals per ply:

  h_norm      : (evaluate(board, learner_color) + 1) / 2  → [0, 1]
  learner_sent: mean sentinel quality of learner's legal moves (learner-turn plies only,
                else 0.5).  Represents how good the learner's options are at this depth.
  opp_sent    : mean sentinel quality of opponent's legal moves (opp-turn plies only,
                else 0.5).  Represents how constrained the opponent is at this depth.
  vn_norm     : (value_net.predict(board, learner_color) + 1) / 2 → [0, 1]; 0.5 if absent
  gap_norm    : (gap_net.predict(board, board.turn) + 1) / 2 → [0, 1]; 0.5 if absent

The contrast between learner_sent (rising) and opp_sent (falling) across plies is
the signature of a trap being set.  This lets the specialist learn trap patterns
implicitly without explicit trap labelling.

12 plies × 5 signals = 60 floats — same width as the previous 20×3 design.
Total per candidate: 62 base + 60 lookahead + 4 topK = 126 (shape unchanged).
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

    Returns a (k, ply_depth*5) ndarray — one row per candidate move.

    Parameters
    ----------
    sentinel      : SentinelAdvisor or None
    evaluate_fn   : callable(board, player) → float in [-1, 1]
    value_net     : ValueNet or None — provides vn_norm signal (slot 3 per ply)
    gap_net       : GapNet or None — provides gap_norm signal (slot 4 per ply)
    human_db      : HumanDB or None (accepted for API compat, not used as lookahead signal)
    use_sentinel  : if True (default), sent signals use sentinel calls; False → 0.5
    ply_depth     : number of half-plies in output (default 12)
    sim_ply_depth : plies to actually simulate (default = ply_depth); training scripts
                    set this to 5 for speed, with padding to fill the full output width
    """

    def __init__(
        self,
        sentinel,
        evaluate_fn,
        value_net=None,
        gap_net=None,
        human_db=None,
        ngram_model=None,
        use_sentinel: bool = True,
        endgame_db=None,
        ply_depth: int = 12,
        frozen_model=None,
        frozen_device=None,
        sim_ply_depth: Optional[int] = None,
    ) -> None:
        self._sentinel      = sentinel
        self._evaluate      = evaluate_fn
        self._value_net     = value_net
        self._gap_net       = gap_net
        self._human_db      = human_db
        self._use_sentinel  = use_sentinel
        self._endgame_db    = endgame_db
        self._ply_depth     = ply_depth
        self._sim_ply_depth = int(sim_ply_depth) if sim_ply_depth is not None else ply_depth
        self._frozen_model  = frozen_model
        self._frozen_device = frozen_device
        self.feat_dim       = ply_depth * 5   # e.g. 12 × 5 = 60

    def set_frozen_model(self, model, device=None) -> None:
        """Update the frozen-model snapshot used to pick learner-side moves in lookahead."""
        self._frozen_model = model
        if device is not None:
            self._frozen_device = device

    def _frozen_model_best_move(self, board: BoardState, color: str):
        """Pick the frozen model's argmax move for ``color`` at ``board``.

        Uses encode_position_with_lookahead(..., lookahead_advisor=None) which zero-pads
        the lookahead block — avoids infinite recursion inside a simulation.
        Returns None on any failure (caller falls back to heuristic).
        """
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
            device = (self._frozen_device if self._frozen_device is not None
                      else next(self._frozen_model.parameters()).device)
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
        sim_ply_depth: Optional[int] = None,
    ) -> np.ndarray:
        """Return per-move lookahead feature block.  Shape: (k, ply_depth*5).

        ``moves_subset`` — if provided, only simulate for those moves (speedup).
        ``sim_ply_depth`` — overrides self._sim_ply_depth for this call only,
        used by training scripts to run full 12-ply simulations 1-in-20 games.
        """
        opp_color = "B" if learner_color == "W" else "W"
        target_moves = moves_subset if moves_subset is not None else enc.legal_moves
        k = len(target_moves)
        if k == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32)

        _sim = sim_ply_depth if sim_ply_depth is not None else self._sim_ply_depth

        rows = []
        for mv in target_moves:
            row = self._simulate_trajectory(board, mv, learner_color, opp_color,
                                            sim_override=_sim)
            rows.append(row)

        return np.stack(rows).astype(np.float32)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _simulate_trajectory(
        self,
        board: BoardState,
        first_move: dict,
        learner_color: str,
        opp_color: str,
        sim_override: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate half-plies and return (ply_depth*5,) float array.

        Layout per ply: [h_norm, learner_sent, opp_sent, vn_norm, gap_norm]
        Half-plies alternate learner → opponent → learner → …
        Terminal and DB probes fill remaining slots with the terminal/probe value.
        If fewer plies are simulated than ply_depth, last valid signal pads the rest.
        """
        result = np.full(self._ply_depth * 5, 0.5, dtype=np.float32)
        _sim = min(
            sim_override if sim_override is not None else self._sim_ply_depth,
            self._ply_depth
        )
        try:
            b      = board
            actors = [learner_color if i % 2 == 0 else opp_color for i in range(_sim)]
            last_sig = (0.5, 0.5, 0.5, 0.5, 0.5)

            for depth_idx in range(_sim):
                actor = actors[depth_idx]

                # Apply move for this half-ply
                if depth_idx == 0:
                    b = b.apply_move(first_move)
                else:
                    mv = None
                    if self._frozen_model is not None and actor == learner_color:
                        mv = self._frozen_model_best_move(b, actor)
                    if mv is None:
                        mv = _static_best_move(b, actor, self._evaluate)
                    if mv is None:
                        for fill in range(depth_idx, self._ply_depth):
                            result[fill * 5 : fill * 5 + 5] = last_sig
                        return result
                    b = b.apply_move(mv)

                # Terminal check
                terminal, winner = is_terminal(b)
                if terminal:
                    val = 1.0 if winner == learner_color else (0.0 if winner else 0.5)
                    for fill in range(depth_idx, self._ply_depth):
                        result[fill * 5 : fill * 5 + 5] = [val, val, val, val, val]
                    return result

                # Endgame DB probe — exact WDL terminates trajectory early
                if self._endgame_db is not None:
                    try:
                        db_result = self._endgame_db.query(b)
                        if db_result is not None:
                            if b.turn == learner_color:
                                val = 1.0 if db_result == "W" else (0.0 if db_result == "L" else 0.5)
                            else:
                                val = 0.0 if db_result == "W" else (1.0 if db_result == "L" else 0.5)
                            for fill in range(depth_idx, self._ply_depth):
                                result[fill * 5 : fill * 5 + 5] = [val, val, val, val, val]
                            return result
                    except Exception:
                        pass

                # Record 5 signals at this position
                sig = self._record_signals(b, learner_color)
                last_sig = sig
                result[depth_idx * 5 : depth_idx * 5 + 5] = sig

            # Pad remaining slots with last valid signal (training speed-up mode)
            if _sim < self._ply_depth:
                for fill in range(_sim, self._ply_depth):
                    result[fill * 5 : fill * 5 + 5] = last_sig

        except Exception:
            pass   # partial result stays; remaining slots keep 0.5

        return result

    def _record_signals(
        self,
        board: BoardState,
        learner_color: str,
    ) -> tuple:
        """Return (h_norm, learner_sent, opp_sent, vn_norm, gap_norm).

        learner_sent is non-0.5 only on learner-turn plies.
        opp_sent is non-0.5 only on opponent-turn plies.
        This split lets the specialist see the per-ply contrast between its own
        option quality and the opponent's option quality — the trap signature.
        """
        current_player  = board.turn
        is_learner_turn = (current_player == learner_color)

        # h_norm — heuristic eval from learner's perspective
        h_norm = 0.5
        try:
            h = float(self._evaluate(board, learner_color))
            h_norm = max(0.0, min(1.0, (h + 1.0) / 2.0))
        except Exception:
            pass

        # learner_sent / opp_sent — split by whose turn it is
        learner_sent = 0.5
        opp_sent     = 0.5
        if self._use_sentinel and self._sentinel is not None:
            try:
                legal = get_all_legal_moves(board)
                if legal:
                    advice = self._sentinel.advise(board, legal, current_player)
                    if advice and getattr(advice, "move_scores", None):
                        m = float(sum(advice.move_scores) / len(advice.move_scores))
                        m = max(0.0, min(1.0, m))
                        if is_learner_turn:
                            learner_sent = m
                        else:
                            opp_sent = m
            except Exception:
                pass

        # vn_norm — value net from learner's perspective
        vn_norm = 0.5
        if self._value_net is not None:
            try:
                v = float(self._value_net.predict(board, learner_color))
                vn_norm = max(0.0, min(1.0, (v + 1.0) / 2.0))
            except Exception:
                pass

        # gap_norm — opponent blunder probability at current position
        gap_norm = 0.5
        if self._gap_net is not None:
            try:
                g = float(self._gap_net.predict(board, current_player))
                gap_norm = max(0.0, min(1.0, (g + 1.0) / 2.0))
            except Exception:
                pass

        return h_norm, learner_sent, opp_sent, vn_norm, gap_norm
