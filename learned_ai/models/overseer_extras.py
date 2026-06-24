"""learned_ai/models/overseer_extras.py — per-move extra features for the Overseer.

Extends the 77-float specialist base with 8 extra floats per candidate move:
  [77:80)  Specialist policy probs (opening, midgame, endgame)
  [80:82)  GameAI alpha-beta features: score_norm [0,1], is_gameai_best (0/1)
  [82:85)  HumanDB features: win_rate [0,1], freq_norm [0,1], seen_flag (0/1)

Total overseer feature dim: 77 + 8 = 85.

Used by both train_scaffolded_overseer.py (training depth=3) and
ScaffoldedAgent at inference (gameplay depth=5).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

OVERSEER_EXTRA_DIM = 8    # 3 specialist + 2 gameai + 3 humandb


def _move_notation_simple(mv: dict) -> str:
    """Notation for a move dict — mirrors GameAI._move_notation."""
    s = f"{mv['from']}-{mv['to']}" if mv.get("from") else mv.get("to", "")
    if mv.get("capture"):
        s += f"x{mv['capture']}"
    return s


def build_overseer_extras(
    base_feat: np.ndarray,          # (k, 77)
    board,                           # BoardState
    enc,                             # EncodedPosition — .legal_moves list
    learner_color: str,
    spec_open=None,                  # ScaffoldedPolicyNet | None
    spec_mid=None,
    spec_end=None,
    gameai=None,                     # GameAI | None — must have .score_root_moves()
    human_db=None,                   # HumanDB | None
    gameai_depth: int = 3,
    device=None,                     # torch.device | None
) -> np.ndarray:                     # (k, 85)
    """Return full 85-float overseer feature matrix.

    Concatenates 8 extra per-move floats onto the 77-float base.
    All extra slots default to neutral values (0.5 or 0.0) when the
    corresponding source is unavailable or raises an exception.
    """
    k   = base_feat.shape[0]
    ext = np.zeros((k, OVERSEER_EXTRA_DIM), dtype=np.float32)

    # Default neutral values for each block
    ext[:, 0:3] = 1.0 / max(k, 1)   # specialist probs — uniform
    ext[:, 3]   = 0.5                # gameai score — neutral
    ext[:, 4]   = 0.0                # gameai is_best
    ext[:, 5]   = 0.5                # human win_rate — neutral
    ext[:, 6]   = 0.0                # human freq_norm
    ext[:, 7]   = 0.0                # human seen_flag

    feat_t = torch.tensor(base_feat, dtype=torch.float32)
    if device is not None:
        feat_t = feat_t.to(device)

    # ── [77:80] Specialist policy probs ──────────────────────────────────────
    for col, spec in enumerate([spec_open, spec_mid, spec_end]):
        if spec is not None:
            try:
                with torch.no_grad():
                    # Support specialists trained with fewer features (e.g. 62 vs 77)
                    spec_dim = getattr(spec, "move_feat_dim", feat_t.shape[1])
                    spec_input = feat_t[:, :spec_dim]
                    probs = torch.softmax(spec.policy_logits(spec_input), dim=-1)
                ext[:, col] = probs.cpu().numpy()
            except Exception:
                pass   # keep uniform default

    # ── [80:82] GameAI: score_norm, is_best ──────────────────────────────────
    if gameai is not None:
        try:
            scored = gameai.score_root_moves(board, depth=gameai_depth, time_budget=2.0)
            if scored:
                best_notation  = _move_notation_simple(scored[0][0])
                score_by_note  = {_move_notation_simple(mv): s for mv, s in scored}
                for i, mv in enumerate(enc.legal_moves):
                    n = _move_notation_simple(mv)
                    ext[i, 3] = score_by_note.get(n, 0.5)
                    ext[i, 4] = 1.0 if n == best_notation else 0.0
        except Exception:
            pass   # keep neutral defaults

    # ── [82:85] HumanDB: win_rate, freq_norm, seen_flag ──────────────────────
    if human_db is not None:
        try:
            hdb_moves = human_db.query_moves(board)
            if hdb_moves:
                max_total  = max((m.total for m in hdb_moves), default=1) or 1
                note_map   = {m.notation: m for m in hdb_moves}
                for i, mv in enumerate(enc.legal_moves):
                    n = _move_notation_simple(mv)
                    if n in note_map:
                        m = note_map[n]
                        ext[i, 5] = float(m.win_pct)
                        ext[i, 6] = float(m.total) / max_total
                        ext[i, 7] = 1.0
        except Exception:
            pass   # keep neutral defaults

    return np.concatenate([base_feat, ext], axis=1).astype(np.float32)   # (k, 85)
