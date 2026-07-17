"""learned_ai/agents/scaffolded_agent.py — ScaffoldedAgent inference wrapper.

Drop-in choose_move() interface for the scaffolded meta-policy.  At inference
the agent needs:
  * a loaded ScaffoldedPolicyNet
  * a loaded SentinelAdvisor (optional but strongly recommended)
  * access to the heuristic evaluate function (via scaffolded_encoder)
  * an optional ExternalSolvedDB for Malom context

Unlike LearnedAgent, there is no fixed action space — the network scores each
legal move directly, so no action masking or phase routing is needed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from learned_ai.agents.heuristic_agent import get_heuristic_evaluate as _get_heuristic_evaluate
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.models.overseer_extras import build_overseer_extras
from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet


SENTINEL_LOOKAHEAD_S_EST = 0.5   # rough estimate of 15-ply lookahead cost per position

def specialist_target_time_s(heuristic_time_s: float,
                             sentinel_lookahead_s: float = SENTINEL_LOOKAHEAD_S_EST) -> float:
    """Target per-move wall time for a specialist AI.

    Rule: specialist_time >= max(2 * heuristic_time, 1 s + sentinel_lookahead).
    The specialist's actual forward pass is instant; its cost is dominated by the
    15-ply sentinel lookahead inside encode_position_with_lookahead().  Callers
    should size the heuristic opponent so this ratio holds."""
    return max(2.0 * heuristic_time_s, 1.0 + sentinel_lookahead_s)


@dataclass
class ScaffoldedDecision:
    """Trace of the most recent move for trainers / loggers."""

    move_features:  np.ndarray     # (k, 62) for all legal moves at this step
    value_input:    np.ndarray     # (23,)
    chosen_idx:     int            # index into legal_moves
    legal_moves:    list           # full list of legal move dicts
    log_prob:       float          # log P(chosen_idx) at decision time
    value:          float          # estimated V(s)
    # For reward computation:
    sentinel_scores: list[float]
    h_scores_abs:    list[float]
    h_before:        float
    h_top1_idx:      int
    db_moves:        list


class ScaffoldedAgent:
    """Inference wrapper around ScaffoldedPolicyNet for use in gameplay."""

    def __init__(
        self,
        color: str = "B",
        model: Optional[ScaffoldedPolicyNet] = None,
        checkpoint_path: Optional[str] = None,
        sentinel_advisor=None,
        db=None,
        value_net=None,
        gap_net=None,
        lookahead_advisor=None,
        # Overseer-only params (ignored for specialist agents)
        is_overseer: bool = False,
        spec_open=None,
        spec_mid=None,
        spec_end=None,
        gameai=None,             # GameAI for overseer alpha-beta features (depth=5 at gameplay)
        human_db=None,           # HumanDB for overseer human-game features
        gameai_depth: int = 5,   # depth for GameAI search at inference
        device: str = "auto",
        mode: str = "sample",
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.color = color
        self.sentinel_advisor = sentinel_advisor
        self.db = db
        self.value_net = value_net
        self.gap_net = gap_net
        self.mode = mode
        self.temperature = max(float(temperature), 1e-6)
        self.last_was_blunder = False
        self.last_thinking = "scaffolded"

        # Overseer mode: applies build_overseer_extras at inference
        self._is_overseer = is_overseer
        self._spec_open   = spec_open
        self._spec_mid    = spec_mid
        self._spec_end    = spec_end
        self._gameai      = gameai
        self._human_db    = human_db
        self._gameai_depth = gameai_depth

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            self.model = ScaffoldedPolicyNet().to(self.device)

        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(seed)
        else:
            self._gen.seed()

        if lookahead_advisor is not None:
            self.lookahead_advisor = lookahead_advisor
        else:
            _evaluate_fn = _get_heuristic_evaluate()
            self.lookahead_advisor = LookaheadAdvisor(
                sentinel=sentinel_advisor,
                value_net=value_net,
                evaluate_fn=_evaluate_fn,
                gap_net=gap_net,
                use_sentinel=True,
                ply_depth=15,
            )

        self.last_decision: Optional[ScaffoldedDecision] = None

    def _load_checkpoint(self, path: str) -> ScaffoldedPolicyNet:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            cfg = ckpt.get("model_config", {})
            model = ScaffoldedPolicyNet.from_config(cfg)
            sd_key = "model" if "model" in ckpt else "state_dict"
            model.load_state_dict(ckpt[sd_key])
        else:
            model = ScaffoldedPolicyNet()
            model.load_state_dict(ckpt)
        return model.to(self.device)

    def set_mode(self, mode: str) -> None:
        if mode not in {"argmax", "sample"}:
            raise ValueError(f"mode must be 'argmax' or 'sample'; got {mode!r}")
        self.mode = mode

    def set_temperature(self, t: float) -> None:
        self.temperature = max(float(t), 1e-6)

    # ── inference ──────────────────────────────────────────────────────────────

    def choose_move(self, board: BoardState, **_) -> dict:
        player = board.turn

        enc = encode_position_with_lookahead(
            board,
            player,
            sentinel_advisor=self.sentinel_advisor,
            db=self.db,
            value_net=self.value_net,
            lookahead_advisor=self.lookahead_advisor,
        )
        if enc is None or len(enc.legal_moves) == 0:
            return {}

        if self._is_overseer:
            feat_matrix = build_overseer_extras(
                enc.feat_matrix, board, enc, player,
                self._spec_open, self._spec_mid, self._spec_end,
                self._gameai, self._human_db, self._gameai_depth,
                self.device,
            )
        else:
            feat_matrix = enc.feat_matrix

        feat_t = torch.tensor(feat_matrix, dtype=torch.float32).to(self.device)
        vi_t   = torch.tensor(enc.value_input,  dtype=torch.float32).to(self.device)

        with torch.no_grad():
            result = self.model.forward(feat_t, vi_t)
            logits = result["logits"]   # (k,)
            value  = float(result["value"].item())

        chosen_idx, log_prob = self._select(logits)

        self.last_decision = ScaffoldedDecision(
            move_features=enc.feat_matrix,
            value_input=enc.value_input,
            chosen_idx=chosen_idx,
            legal_moves=enc.legal_moves,
            log_prob=float(log_prob),
            value=value,
            sentinel_scores=enc.sentinel_scores,
            h_scores_abs=enc.h_scores_abs,
            h_before=enc.h_before,
            h_top1_idx=enc.h_top1_idx,
            db_moves=enc.db_moves,
        )

        return enc.legal_moves[chosen_idx]

    def _select(self, logits: torch.Tensor) -> tuple[int, float]:
        if self.mode == "argmax" or self.temperature <= 1e-6:
            idx = int(torch.argmax(logits).item())
            log_probs = F.log_softmax(logits, dim=-1)
            return idx, float(log_probs[idx].item())

        scaled    = logits / self.temperature
        log_probs = F.log_softmax(scaled, dim=-1)
        probs     = log_probs.exp()
        if not torch.isfinite(probs).all():
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            probs = probs / probs.sum().clamp(min=1e-9)
        idx = int(torch.multinomial(probs.cpu(), 1, generator=self._gen).item())
        return idx, float(log_probs[idx].item())

    # ── phase-routing inference (v2 three-specialist) ──────────────────────────

    def choose_move_for_phase(
        self,
        board: BoardState,
        spec_open: "ScaffoldedAgent",
        spec_mid: "ScaffoldedAgent",
        spec_end: "ScaffoldedAgent",
    ) -> dict:
        """Route to the correct specialist based on game phase and piece counts.

        Routing:
          place phase                         → spec_open
          move/fly + either side ≤ 5 pieces  → spec_end
          otherwise                           → spec_mid
        """
        phase = get_game_phase(board, board.turn)
        if phase == "place":
            return spec_open.choose_move(board)
        own = board.pieces_on_board.get(board.turn, 0)
        opp_color = "B" if board.turn == "W" else "W"
        opp = board.pieces_on_board.get(opp_color, 0)
        if own <= 5 or opp <= 5:
            return spec_end.choose_move(board)
        return spec_mid.choose_move(board)
