"""learned_ai/agents/specialist_router.py — three-specialist inference router.

Loads the three v2 phase specialists (opening, midgame, endgame) and routes
each move choice to the appropriate specialist based on game phase.

Drop-in replacement for `OverseerAdvisor`: same interface (`is_loaded`,
`score_moves`, `set_db`, etc.), so the web/app.py wiring for the "Overseer
player" toggle drives it unchanged.

Routing rule (matches ScaffoldedAgent.choose_move_for_phase):
  * placement phase             → opening specialist
  * move/fly + ≤5 own or opp    → endgame specialist
  * else                        → midgame specialist

At inference each specialist sees:
  * feat_matrix (k, 122) — base 62 + 15-ply lookahead (h/vn/sent/gap)
  * The specialist's forward pass is instant; wall time is dominated by the
    LookaheadAdvisor's 15-ply sentinel calls.

Checkpoint search:
  learned_ai/checkpoints/scaffolded/{s_open_v2,s_mid_v2,s_end_v2}/best.pt

Returns None if all three specialists fail to load — caller falls back to
the classical coordinator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from game.board import BoardState
from game.rules import get_game_phase

log = logging.getLogger("nmm.specialist_router")


def _move_key(mv: dict) -> tuple:
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _load_spec_model(path: Path):
    """Load a ScaffoldedPolicyNet checkpoint. Returns (model, cfg) or (None, {})."""
    if not path.exists():
        return None, {}
    try:
        from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg   = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg)
        state = ckpt.get("model") or ckpt
        model.load_state_dict(state)
        model.eval()
        return model, cfg
    except Exception as e:
        log.warning("Specialist load failed at %s: %s", path, e)
        return None, {}


class SpecialistRouter:
    """Phase-router over three v2 specialists.  API-compatible with OverseerAdvisor."""

    def __init__(
        self,
        spec_open,
        spec_mid,
        spec_end,
        ckpt_paths: dict[str, str],
        sentinel_advisor=None,
        db=None,
        value_net=None,
        gap_net=None,
        endgame_db=None,
        human_db=None,
        gameai=None,
        lookahead_advisor_open=None,
        lookahead_advisor_mid=None,
        lookahead_advisor_end=None,
    ) -> None:
        self._spec_open = spec_open
        self._spec_mid  = spec_mid
        self._spec_end  = spec_end
        self._ckpt_paths = ckpt_paths
        self._sentinel  = sentinel_advisor
        self._db        = db          # Malom perfect DB (compat with Overseer set_db)
        self._value_net = value_net
        self._gap_net   = gap_net
        self._endgame_db = endgame_db
        self._human_db  = human_db
        self._gameai    = gameai
        self._la_open   = lookahead_advisor_open
        self._la_mid    = lookahead_advisor_mid
        self._la_end    = lookahead_advisor_end

    # ── OverseerAdvisor-compatible surface ────────────────────────────────────

    def is_loaded(self) -> bool:
        """True when at least one specialist is loaded (router still usable)."""
        return any(m is not None for m in (self._spec_open, self._spec_mid, self._spec_end))

    def set_sentinel(self, sentinel_advisor) -> None:
        self._sentinel = sentinel_advisor
        for la in (self._la_open, self._la_mid, self._la_end):
            if la is not None:
                la._sentinel = sentinel_advisor  # type: ignore[attr-defined]

    def set_db(self, db) -> None:
        """Compat with OverseerAdvisor — Malom perfect DB.  Also wires the
        endgame LookaheadAdvisor's early-terminate DB probe if it isn't set."""
        self._db = db
        if self._endgame_db is None and self._la_end is not None:
            self._la_end._endgame_db = db  # type: ignore[attr-defined]

    def set_value_net(self, value_net) -> None:
        self._value_net = value_net
        for la in (self._la_open, self._la_mid, self._la_end):
            if la is not None:
                la._value_net = value_net  # type: ignore[attr-defined]

    def set_human_db(self, human_db) -> None:
        self._human_db = human_db

    def set_gameai(self, gameai) -> None:
        self._gameai = gameai

    # ── routing ───────────────────────────────────────────────────────────────

    def _pick_specialist(self, board: BoardState, color: str):
        """Return (specialist_model, lookahead_advisor, phase_label)."""
        phase = get_game_phase(board, color)
        if phase == "place":
            return self._spec_open, self._la_open, "opening"
        own = int(board.pieces_on_board.get(color, 0))
        opp_color = "B" if color == "W" else "W"
        opp = int(board.pieces_on_board.get(opp_color, 0))
        if own <= 5 or opp <= 5:
            return self._spec_end, self._la_end, "endgame"
        return self._spec_mid, self._la_mid, "midgame"

    # ── inference ─────────────────────────────────────────────────────────────

    def score_moves(self, board: BoardState, candidates: list[dict], color: str) -> Optional[list[float]]:
        """Return per-candidate pick probabilities (sum to 1.0).

        Routes to the phase-appropriate specialist; falls back to whichever
        specialist is loaded if the preferred one is missing.

        v3 (2026-07-16): uses ``encode_top_k_candidates`` when a GameAI is available,
        so the specialist re-ranks the classical engine's top-K alpha-beta moves plus
        their lookahead + human-prior features.  Falls back to the v2 full-legal-moves
        path when GameAI is None (for backward compatibility with older checkpoints).
        """
        if not candidates:
            return None
        try:
            from learned_ai.models.scaffolded_encoder import (
                encode_position_with_lookahead,
                encode_top_k_candidates,
            )

            spec, la, phase_label = self._pick_specialist(board, color)
            # Fallback ladder: preferred → any other loaded specialist
            if spec is None:
                for alt, alt_la in ((self._spec_mid, self._la_mid),
                                    (self._spec_end, self._la_end),
                                    (self._spec_open, self._la_open)):
                    if alt is not None:
                        spec, la = alt, alt_la
                        phase_label += "→fallback"
                        break
            if spec is None:
                return None

            # v3 top-K path: use GameAI's alpha-beta ordering plus human-prior features.
            # Only enabled when a GameAI is attached AND the loaded specialist's move_feat_dim matches.
            expected_dim = getattr(spec, "move_feat_dim", None)
            use_topk = (
                self._gameai is not None
                and expected_dim == 126        # MOVE_FEAT_DIM_WITH_TOPK
            )

            if use_topk:
                # Shared-search: reuse the coordinator's already-populated
                # transposition table + killer moves + history.  We also cap
                # the depth to whatever the coordinator reached, so the
                # second search returns instantly (all TT hits) instead of
                # burning another 15-60 s of wall time.
                _last_depth = getattr(self._gameai, "last_depth_reached", 0) or 0
                _ab_depth = max(2, int(_last_depth)) if _last_depth else None
                enc = encode_top_k_candidates(
                    board, color,
                    gameai=self._gameai,
                    top_k=5,
                    ab_depth=_ab_depth,
                    ab_time_budget=2.0,             # ceiling — TT hits finish it faster
                    ab_preserve_tt=True,            # ← key: reuse coordinator's search state
                    sentinel_advisor=self._sentinel,
                    db=None,
                    value_net=self._value_net,
                    lookahead_advisor=la,
                    human_db=self._human_db,
                    trajectory_db=None,
                    ngram_model=None,
                )
            else:
                enc = encode_position_with_lookahead(
                    board, color,
                    sentinel_advisor=self._sentinel,
                    db=None,
                    value_net=self._value_net,
                    lookahead_advisor=la,
                )
            if enc is None or not enc.legal_moves:
                return None

            feat = torch.from_numpy(enc.feat_matrix).to(torch.float32)
            with torch.no_grad():
                probs = spec.policy_probs(feat)   # (k,)
            probs_np = probs.cpu().numpy()

            enc_key_to_idx = {_move_key(m): i for i, m in enumerate(enc.legal_moves)}
            result: list[float] = []
            for cand in candidates:
                idx = enc_key_to_idx.get(_move_key(cand))
                result.append(float(probs_np[idx]) if idx is not None and idx < len(probs_np) else 0.0)

            total = sum(result)
            if total > 1e-9:
                result = [v / total for v in result]
            return result

        except Exception as e:
            log.warning("SpecialistRouter.score_moves failed: %s", e, exc_info=True)
            return None


# ── loader ────────────────────────────────────────────────────────────────────

def load_specialist_router(
    ckpt_dir: Optional[Path] = None,
    sentinel_advisor=None,
    db=None,
    value_net=None,
    gap_net=None,
    human_db=None,
    ply_depth: int = 15,
) -> Optional[SpecialistRouter]:
    """Load the three v2 specialists and their LookaheadAdvisors.

    Returns None only if ALL three specialist checkpoints fail to load.
    """
    from learned_ai.models.lookahead_advisor import LookaheadAdvisor
    from learned_ai.agents.heuristic_agent import get_heuristic_evaluate

    root = Path(__file__).parent.parent.parent
    if ckpt_dir is None:
        ckpt_dir = root / "learned_ai" / "checkpoints" / "scaffolded"

    open_path = ckpt_dir / "s_open_v2" / "best.pt"
    mid_path  = ckpt_dir / "s_mid_v2"  / "best.pt"
    end_path  = ckpt_dir / "s_end_v2"  / "best.pt"

    m_open, cfg_open = _load_spec_model(open_path)
    m_mid,  cfg_mid  = _load_spec_model(mid_path)
    m_end,  cfg_end  = _load_spec_model(end_path)

    if not any((m_open, m_mid, m_end)):
        log.info("SpecialistRouter: no v2 checkpoints found (searched %s, %s, %s)",
                 open_path, mid_path, end_path)
        return None

    evaluate_fn = get_heuristic_evaluate()

    def _mk_la(endgame_db_arg=None):
        try:
            return LookaheadAdvisor(
                sentinel=sentinel_advisor,
                value_net=value_net,
                evaluate_fn=evaluate_fn,
                gap_net=gap_net,
                use_sentinel=True,
                endgame_db=endgame_db_arg,
                ply_depth=ply_depth,
            )
        except Exception as e:
            log.warning("LookaheadAdvisor init failed: %s", e)
            return None

    la_open = _mk_la() if m_open is not None else None
    la_mid  = _mk_la() if m_mid  is not None else None
    la_end  = _mk_la(endgame_db_arg=db) if m_end is not None else None

    log.info("SpecialistRouter loaded — open=%s mid=%s end=%s ply_depth=%d",
             "OK" if m_open else "missing",
             "OK" if m_mid  else "missing",
             "OK" if m_end  else "missing",
             ply_depth)

    return SpecialistRouter(
        spec_open=m_open, spec_mid=m_mid, spec_end=m_end,
        ckpt_paths={
            "open": str(open_path) if m_open else "",
            "mid":  str(mid_path)  if m_mid  else "",
            "end":  str(end_path)  if m_end  else "",
        },
        sentinel_advisor=sentinel_advisor,
        db=db, value_net=value_net, gap_net=gap_net,
        endgame_db=db, human_db=human_db,
        lookahead_advisor_open=la_open,
        lookahead_advisor_mid=la_mid,
        lookahead_advisor_end=la_end,
    )
