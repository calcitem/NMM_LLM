"""learned_ai/models/overseer.py — OverseerAdvisor inference wrapper.

Loads a ScaffoldedPolicyNet checkpoint and exposes per-move pick probabilities
for the diagnostic overlay ("O:XX%" labels) and Overseer player mode.

Feature pipeline mirrors training exactly (train_scaffolded_overseer.py):
  1. encode_position_with_lookahead → (k, 77) base features
     - sentinel_advisor: fills base feature slots [58:62)
     - LookaheadAdvisor: fills lookahead block [62:77) using 5-ply heuristic+VN+sentinel
     - db=None (Malom DB was not in the base during Overseer training)
  2. build_overseer_extras → (k, 85) full feature matrix
     - [77:80) specialist policy probs (opening, midgame, endgame)
     - [80:82) GameAI alpha-beta features
     - [82:85) HumanDB features

Checkpoint search order: s_over/best.pt → s_over/latest.pt → s2 → s1b → s1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger("nmm.overseer")


def _move_key(mv: dict) -> tuple:
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _load_spec_model(path: Path):
    """Load a ScaffoldedPolicyNet from a checkpoint. Returns None on failure."""
    if not path.exists():
        return None
    try:
        from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg   = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg)
        state = ckpt.get("model") or ckpt
        model.load_state_dict(state)
        model.eval()
        log.info("  specialist loaded: %s (feat_dim=%s)", path, cfg.get("move_feat_dim", "?"))
        return model
    except Exception as e:
        log.warning("  specialist load failed %s: %s", path, e)
        return None


class OverseerAdvisor:
    """Wraps ScaffoldedPolicyNet for per-move probability scoring.

    Replicates the full 85-float training feature pipeline at inference so the
    model sees the same inputs it was trained on.
    """

    def __init__(
        self,
        model,
        ckpt_path: str,
        sentinel_advisor=None,
        db=None,
        value_net=None,
        stage: str = "unknown",
        spec_open=None,
        spec_mid=None,
        spec_end=None,
        human_db=None,
        gameai=None,
        lookahead_advisor=None,
    ) -> None:
        self._model    = model
        self._ckpt_path = ckpt_path
        self._sentinel  = sentinel_advisor
        self._db        = db          # Malom DB (used by set_db for legacy callers)
        self._value_net = value_net
        self._stage     = stage
        self._spec_open = spec_open
        self._spec_mid  = spec_mid
        self._spec_end  = spec_end
        self._human_db  = human_db
        self._gameai    = gameai
        self._lookahead = lookahead_advisor
        model.eval()

    def is_loaded(self) -> bool:
        return self._model is not None

    def set_sentinel(self, sentinel_advisor) -> None:
        self._sentinel = sentinel_advisor

    def set_db(self, db) -> None:
        self._db = db

    def set_value_net(self, value_net) -> None:
        self._value_net = value_net

    def set_human_db(self, human_db) -> None:
        self._human_db = human_db

    def set_gameai(self, gameai) -> None:
        self._gameai = gameai

    def score_moves(self, board, candidates: list[dict], color: str) -> Optional[list[float]]:
        """Return per-candidate pick probabilities (sum to 1.0).

        Candidates must be a list of move dicts with 'from', 'to', 'capture' keys.
        Returns None on any failure.
        """
        if not self._model or not candidates:
            return None
        try:
            from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead
            from learned_ai.models.overseer_extras import build_overseer_extras

            # Step 1: 77-float base (sentinel in base slots + LookaheadAdvisor for [62:77)).
            # db=None matches training (Malom DB was not in the base during Overseer training).
            enc = encode_position_with_lookahead(
                board, color,
                sentinel_advisor=self._sentinel,
                db=None,
                value_net=self._value_net,
                lookahead_advisor=self._lookahead,
            )
            if enc is None or not enc.legal_moves:
                return None

            # Step 2: extend to 85 floats with specialist probs, GameAI, and HumanDB.
            feat_85 = build_overseer_extras(
                enc.feat_matrix,   # (k, 77)
                board, enc, color,
                spec_open=self._spec_open,
                spec_mid=self._spec_mid,
                spec_end=self._spec_end,
                gameai=self._gameai,
                human_db=self._human_db,
                gameai_depth=5,    # gameplay: deeper than training (3) for better signals
            )

            feat = torch.from_numpy(feat_85)   # (k, 85)
            with torch.no_grad():
                probs = self._model.policy_probs(feat)   # (k,)

            probs_np = probs.cpu().numpy()

            # Align to input candidate order; unmapped candidates get 0.
            enc_key_to_idx = {_move_key(m): i for i, m in enumerate(enc.legal_moves)}
            result = []
            for cand in candidates:
                idx = enc_key_to_idx.get(_move_key(cand))
                result.append(float(probs_np[idx]) if idx is not None and idx < len(probs_np) else 0.0)

            total = sum(result)
            if total > 1e-9:
                result = [v / total for v in result]
            return result

        except Exception as e:
            log.warning("OverseerAdvisor.score_moves failed: %s", e, exc_info=True)
            return None


def load_overseer(
    ckpt_path: Optional[str] = None,
    sentinel_advisor=None,
    db=None,
    value_net=None,
    human_db=None,
    gameai=None,
) -> Optional[OverseerAdvisor]:
    """Load OverseerAdvisor from checkpoint.  Returns None on any failure.

    If ckpt_path is None, searches: s_over/best.pt → s_over/latest.pt → s2 → s1b → s1
    under learned_ai/checkpoints/scaffolded/.

    Automatically loads specialist models (s_open-retired, s_mid, s_end) and
    builds a LookaheadAdvisor matching the training configuration.
    """
    try:
        from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
    except Exception as e:
        log.warning("OverseerAdvisor: cannot import ScaffoldedPolicyNet: %s", e)
        return None

    _root    = Path(__file__).parent.parent.parent
    ckpt_dir = _root / "learned_ai" / "checkpoints" / "scaffolded"

    # ── Find overseer checkpoint ──────────────────────────────────────────────
    if ckpt_path:
        search_paths = [Path(ckpt_path)]
    else:
        search_paths = [
            ckpt_dir / "s_over" / "best.pt",
            ckpt_dir / "s_over" / "latest.pt",
            ckpt_dir / "s2"     / "best.pt",
            ckpt_dir / "s1b"    / "best.pt",
            ckpt_dir / "s1"     / "best.pt",
        ]

    chosen = next((p for p in search_paths if p.exists()), None)
    if chosen is None:
        log.info("OverseerAdvisor: no checkpoint found (searched %s)", search_paths)
        return None

    try:
        ckpt  = torch.load(str(chosen), map_location="cpu", weights_only=False)
        cfg   = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg)
        state = ckpt.get("model") or ckpt
        model.load_state_dict(state)
        model.eval()
        stage = ckpt.get("stage", "unknown")
    except Exception as e:
        log.warning("OverseerAdvisor: failed to load %s: %s", chosen, e)
        return None

    move_feat_dim = cfg.get("move_feat_dim", 0)

    # Derive lookahead ply depth and GameAI depth from the checkpoint's feature dim.
    # move_feat_dim = 62 (base) + ply_depth*3 (lookahead) + 8 (extras)
    _OVERSEER_EXTRA_DIM = 8
    _BASE_DIM = 62
    lookahead_block_dim = move_feat_dim - _BASE_DIM - _OVERSEER_EXTRA_DIM
    ply_depth  = max(1, lookahead_block_dim // 3) if lookahead_block_dim > 0 else 5
    gameai_inf_depth = 7 if move_feat_dim >= 106 else 3

    log.info("OverseerAdvisor: loaded %s (stage=%s, move_feat_dim=%d, ply_depth=%d, gameai_depth=%d)",
             chosen, stage, move_feat_dim, ply_depth, gameai_inf_depth)

    # ── Specialist models ─────────────────────────────────────────────────────
    # Only load specialists when the checkpoint uses the full Overseer feature set.
    spec_open = spec_mid = spec_end = None
    if move_feat_dim >= 85:
        log.info("OverseerAdvisor: loading specialist models…")
        spec_open = _load_spec_model(ckpt_dir / "s_open-retired" / "best.pt")
        spec_mid  = _load_spec_model(ckpt_dir / "s_mid"          / "best.pt")
        spec_end  = _load_spec_model(ckpt_dir / "s_end"          / "best.pt")

    # ── LookaheadAdvisor ─────────────────────────────────────────────────────
    # ply_depth is derived from the checkpoint so inference always matches training.
    lookahead = None
    if move_feat_dim > 62:
        try:
            from learned_ai.models.lookahead_advisor import LookaheadAdvisor
            from learned_ai.agents.heuristic_agent import get_heuristic_evaluate
            evaluate_fn = get_heuristic_evaluate()
            lookahead = LookaheadAdvisor(
                sentinel=sentinel_advisor,
                value_net=value_net,
                evaluate_fn=evaluate_fn,
                use_sentinel=True,
                ply_depth=ply_depth,
            )
            log.info("OverseerAdvisor: LookaheadAdvisor ready (ply_depth=%d, feat_dim=%d)",
                     ply_depth, lookahead.feat_dim)
        except Exception as e:
            log.warning("OverseerAdvisor: LookaheadAdvisor failed — lookahead block will be zero: %s", e)

    # ── GameAI singleton for alpha-beta features ──────────────────────────────
    _gameai = gameai
    if _gameai is None and move_feat_dim >= 85:
        try:
            from ai.game_ai import GameAI
            _gameai = GameAI(color="W", difficulty=gameai_inf_depth)
            log.info("OverseerAdvisor: GameAI singleton created (depth=%d)", gameai_inf_depth)
        except Exception as e:
            log.warning("OverseerAdvisor: GameAI unavailable — alpha-beta features will be neutral: %s", e)

    return OverseerAdvisor(
        model, str(chosen),
        sentinel_advisor=sentinel_advisor,
        db=db,
        value_net=value_net,
        stage=stage,
        spec_open=spec_open,
        spec_mid=spec_mid,
        spec_end=spec_end,
        human_db=human_db,
        gameai=_gameai,
        lookahead_advisor=lookahead,
    )
