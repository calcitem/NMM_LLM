"""learned_ai/models/overseer.py — OverseerAdvisor inference wrapper.

Loads a ScaffoldedPolicyNet checkpoint and exposes per-move pick probabilities
for the diagnostic overlay ("O:XX%" labels).  Advisory only — does not alter AI play.

Checkpoint search order: s1b/best.pt → s1/best.pt (takes the most-trained available).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


def _move_key(mv: dict) -> tuple:
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


class OverseerAdvisor:
    """Wraps ScaffoldedPolicyNet for per-move probability scoring."""

    def __init__(self, model, ckpt_path: str, sentinel_advisor=None, db=None,
                 value_net=None) -> None:
        self._model = model
        self._ckpt_path = ckpt_path
        self._sentinel = sentinel_advisor
        self._db = db
        self._value_net = value_net
        model.eval()

    def is_loaded(self) -> bool:
        return self._model is not None

    def set_sentinel(self, sentinel_advisor) -> None:
        self._sentinel = sentinel_advisor

    def set_db(self, db) -> None:
        self._db = db

    def set_value_net(self, value_net) -> None:
        self._value_net = value_net

    def score_moves(self, board, candidates: list[dict], color: str) -> Optional[list[float]]:
        """Return per-candidate pick probabilities (sum to 1.0).

        Candidates must be a list of move dicts with 'from', 'to', 'capture' keys.
        Returns None on any failure.  If k==1, still returns [1.0] — callers
        may choose to suppress display in that case.
        """
        if not self._model or not candidates:
            return None
        # Capture-phase candidates use {from, to, capture} keys from the engine's pending mill
        # move. encode_position() returns its own legal_move list; if keys don't align,
        # unmapped entries get prob=0 and are suppressed by overseerLabel (pct<1 → None).
        # This is an acceptable silent degradation for v1.
        try:
            from learned_ai.models.scaffolded_encoder import encode_position
            enc = encode_position(board, color,
                                  sentinel_advisor=self._sentinel,
                                  db=self._db,
                                  value_net=self._value_net)
            if enc is None or not enc.legal_moves:
                return None

            # Build key → (index, prob) lookup on encoded legal moves
            enc_key_to_idx = {_move_key(m): i for i, m in enumerate(enc.legal_moves)}

            feat = torch.from_numpy(enc.feat_matrix)   # (k, 62)
            with torch.no_grad():
                probs = self._model.policy_probs(feat)   # (k,) tensor

            probs_np = probs.cpu().numpy()

            # Align to input candidate order; unmapped candidates get 0
            result = []
            for cand in candidates:
                idx = enc_key_to_idx.get(_move_key(cand))
                if idx is not None and idx < len(probs_np):
                    result.append(float(probs_np[idx]))
                else:
                    result.append(0.0)

            # Re-normalise in case any candidates were unmapped
            total = sum(result)
            if total > 1e-9:
                result = [v / total for v in result]
            return result

        except Exception as e:
            log.debug("OverseerAdvisor.score_moves failed: %s", e)
            return None


def load_overseer(
    ckpt_path: Optional[str] = None,
    sentinel_advisor=None,
    db=None,
    value_net=None,
) -> Optional[OverseerAdvisor]:
    """Load OverseerAdvisor from checkpoint.  Returns None on any failure.

    If ckpt_path is None, searches: s1b/best.pt → s1/best.pt under
    learned_ai/checkpoints/scaffolded/.
    """
    try:
        from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
    except Exception as e:
        log.warning("OverseerAdvisor: cannot import ScaffoldedPolicyNet: %s", e)
        return None

    _root = Path(__file__).parent.parent.parent

    search_paths = []
    if ckpt_path:
        search_paths.append(Path(ckpt_path))
    else:
        ckpt_dir = _root / "learned_ai" / "checkpoints" / "scaffolded"
        search_paths = [
            ckpt_dir / "s1b" / "best.pt",
            ckpt_dir / "s1"  / "best.pt",
        ]

    chosen = None
    for p in search_paths:
        if p.exists():
            chosen = p
            break

    if chosen is None:
        log.info("OverseerAdvisor: no checkpoint found (searched %s)", search_paths)
        return None

    try:
        ckpt = torch.load(str(chosen), map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg)
        state = ckpt.get("model") or ckpt
        model.load_state_dict(state)
        model.eval()
        stage = ckpt.get("stage", "unknown")
        log.info("OverseerAdvisor: loaded %s (stage=%s)", chosen, stage)
        return OverseerAdvisor(model, str(chosen),
                               sentinel_advisor=sentinel_advisor, db=db,
                               value_net=value_net)
    except Exception as e:
        log.warning("OverseerAdvisor: failed to load %s: %s", chosen, e)
        return None
