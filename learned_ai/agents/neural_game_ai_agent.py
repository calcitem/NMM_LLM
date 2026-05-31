"""Stage-4 agent: GameAI negamax with the neural net as the leaf evaluator.

The move is chosen by GameAI.choose_move() (which internally calls
NeuralEvaluator at search leaves).  In parallel, a root-level forward pass
through the net records policy logits and value in last_decision so that the
REINFORCE trainer can compute gradients — same interface as LearnedAgent.

This gives the net exposure to tree-search corrections during Stage 4 training
while keeping the gradient pipeline intact.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from game.board import BoardState
from learned_ai.agents.learned_agent import LearnedDecision
from learned_ai.models.action_encoder import (
    ACTION_DIM,
    CAPTURE_OFFSET,
    PLACE_OFFSET,
    encode_action,
    get_legal_mask,
)
from learned_ai.models.state_encoder import detect_phase, encode_state
from typing import Optional


class NeuralGameAIAgent:
    """Stage-4 agent: net value inside negamax, REINFORCE gradient via root pass."""

    def __init__(
        self,
        color: str,
        model,                          # NMMNet instance (shared across agents)
        device: str = "cpu",
        difficulty: int = 1,
        time_budget_s: float = 0.05,    # per-move budget; keep short for training
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.color = color
        self.model = model
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.last_was_blunder = False
        self.last_thinking = "neural+search"

        from ai.neural_evaluator import NeuralEvaluator
        from ai.game_ai import GameAI

        self._evaluator = NeuralEvaluator(model, device=str(self.device))
        self._game_ai = GameAI(
            color=color,
            difficulty=difficulty,
            neural_evaluator=self._evaluator,
            override_time_budget=time_budget_s,
        )

        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(seed)
        else:
            self._gen.seed()
        self.last_decision: Optional[LearnedDecision] = None

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-6)

    def choose_move(self, board: BoardState, **_: object) -> dict:
        # ── 1. Root forward pass — gradient tracked; needed for REINFORCE. ──────
        state = encode_state(board).to(self.device)
        phase_id = detect_phase(board)
        mask = get_legal_mask(board).to(self.device)
        out = self.model.forward(state, phase_id=phase_id, legal_mask=mask)
        logits = out["logits"]   # (action_dim,)
        value = out["value"]     # scalar

        # ── 2. GameAI picks the actual move via shallow negamax + net value. ────
        move = self._game_ai.choose_move(board)
        if not move:
            self.last_decision = None
            return {}

        # ── 3. Map chosen move to primary/capture indices. ────────────────────
        primary_full, capture_full = encode_action(move)

        # ── 4. Log-prob of chosen primary action from ROOT logits. ────────────
        primary_logits = logits[PLACE_OFFSET:CAPTURE_OFFSET]   # shape [600]
        primary_mask_sl = mask[PLACE_OFFSET:CAPTURE_OFFSET]
        masked_primary = primary_logits.masked_fill(~primary_mask_sl, float("-inf"))
        temp = max(self.temperature, 1e-6)
        primary_log_probs = F.log_softmax(masked_primary / temp, dim=-1)
        p_local = primary_full - PLACE_OFFSET  # PLACE_OFFSET=0 so this == primary_full
        primary_log_prob = primary_log_probs[p_local] if 0 <= p_local < primary_logits.shape[0] else primary_log_probs.max()

        # ── 5. Log-prob of capture (when present). ────────────────────────────
        capture_log_prob: Optional[torch.Tensor] = None
        if capture_full is not None:
            cap_logits = logits[CAPTURE_OFFSET:ACTION_DIM]
            cap_mask_sl = mask[CAPTURE_OFFSET:ACTION_DIM]
            if cap_mask_sl.any():
                masked_cap = cap_logits.masked_fill(~cap_mask_sl, float("-inf"))
                cap_log_probs = F.log_softmax(masked_cap / temp, dim=-1)
                c_local = capture_full - CAPTURE_OFFSET
                if 0 <= c_local < cap_logits.shape[0]:
                    capture_log_prob = cap_log_probs[c_local]

        self.last_decision = LearnedDecision(
            state=state.detach().cpu(),
            phase_id=phase_id,
            legal_mask=mask.detach().cpu(),
            primary_index=primary_full,
            capture_index=capture_full,
            primary_log_prob=primary_log_prob.detach().cpu(),
            capture_log_prob=capture_log_prob.detach().cpu() if capture_log_prob is not None else None,
            value=value.detach().cpu(),
        )
        return move
