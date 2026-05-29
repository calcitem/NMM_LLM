"""LearnedAgent: PyTorch inference wrapper for the NMMNet policy/value model.

Drop-in replacement for `ai.game_ai.GameAI` at the `choose_move(board)` level.

Two sampling modes:
    * ``argmax`` — deterministic greedy pick from the masked logits
      (used for evaluation / serving once a checkpoint is trained).
    * ``sample`` — temperature-scaled categorical sample over the masked
      softmax, used for self-play exploration.

The agent owns its own `NMMNet` instance and (optionally) loads a checkpoint
on construction.  It records the most recent (state, primary_action,
capture_action, log_prob, value) tuple so trainers can pick the data up
without re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from game.board import BoardState
from learned_ai.models.action_encoder import (
    ACTION_DIM,
    CAPTURE_OFFSET,
    MOVE_OFFSET,
    PLACE_OFFSET,
    decode_action,
    get_legal_mask,
    move_requires_capture,
)
from learned_ai.models.backbone import NMMNet
from learned_ai.models.state_encoder import detect_phase, encode_state


@dataclass
class LearnedDecision:
    """Trace of the most recent move for trainers / loggers."""

    state: torch.Tensor
    phase_id: int
    legal_mask: torch.Tensor
    primary_index: int
    capture_index: Optional[int]
    primary_log_prob: torch.Tensor
    capture_log_prob: Optional[torch.Tensor]
    value: torch.Tensor


class LearnedAgent:
    def __init__(
        self,
        color: str = "B",
        model: Optional[NMMNet] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        mode: str = "sample",
        temperature: float = 1.0,
        seed: Optional[int] = None,
        backbone_hidden=(256, 256, 128),
        head_hidden=(64,),
        dropout: float = 0.0,
    ) -> None:
        self.color = color
        self.device = torch.device(device)
        self.mode = mode
        self.temperature = float(temperature)
        self.last_was_blunder = False
        self.last_thinking = "learned"

        ckpt = None
        if checkpoint_path:
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

        # When no explicit model is given, prefer the architecture embedded in
        # the checkpoint so a small (smoke) net loads into a small net and a
        # full net loads into a full net — avoids state_dict shape mismatches.
        if model is None:
            if isinstance(ckpt, dict) and ckpt.get("model_config"):
                mc = ckpt["model_config"]
                model = NMMNet(
                    backbone_hidden=tuple(mc.get("backbone_hidden", backbone_hidden)),
                    head_hidden=tuple(mc.get("head_hidden", head_hidden)),
                    dropout=float(mc.get("dropout", dropout)),
                )
            else:
                model = NMMNet(
                    backbone_hidden=backbone_hidden,
                    head_hidden=head_hidden,
                    dropout=dropout,
                )
        self.model = model.to(self.device)

        if ckpt is not None:
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(state_dict)

        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(seed)
        else:
            self._gen.seed()

        self.last_decision: Optional[LearnedDecision] = None

    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in {"argmax", "sample"}:
            raise ValueError(f"mode must be 'argmax' or 'sample'; got {mode!r}")
        self.mode = mode

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-6)

    # ------------------------------------------------------------------

    def choose_move(self, board: BoardState, **_: object) -> dict:
        if board.turn != self.color:
            # Allowed but unusual — the agent will still produce a legal move
            # for whoever is on move (used in self-play where both sides are
            # this agent).
            pass

        state = encode_state(board).to(self.device)
        phase_id = detect_phase(board)
        legal_mask = get_legal_mask(board).to(self.device)
        if not legal_mask.any():
            return {}

        out = self.model.forward(state, phase_id=phase_id, legal_mask=legal_mask)
        logits = out["logits"]
        value = out["value"]

        primary_logits = logits[PLACE_OFFSET:CAPTURE_OFFSET]
        primary_mask = legal_mask[PLACE_OFFSET:CAPTURE_OFFSET]
        if not primary_mask.any():
            return {}

        primary_index, primary_log_prob = self._select(
            primary_logits, primary_mask
        )
        primary_index_full = primary_index + PLACE_OFFSET

        capture_index_full: Optional[int] = None
        capture_log_prob: Optional[torch.Tensor] = None
        if move_requires_capture(board, primary_index_full):
            cap_logits = logits[CAPTURE_OFFSET:ACTION_DIM]
            cap_mask = legal_mask[CAPTURE_OFFSET:ACTION_DIM]
            if not cap_mask.any():
                # Should not happen if the engine produced any mill-forming
                # move; fall back to the first legal capture.
                from game.board import POSITIONS

                from learned_ai.models.action_encoder import POS_INDEX

                first = board.legal_captures(board.turn)[0]
                capture_index_full = CAPTURE_OFFSET + POS_INDEX[first]
            else:
                cap_index, cap_log_prob = self._select(cap_logits, cap_mask)
                capture_index_full = cap_index + CAPTURE_OFFSET
                capture_log_prob = cap_log_prob

        move = decode_action(
            primary_index_full, board, capture_index=capture_index_full
        )

        self.last_decision = LearnedDecision(
            state=state.detach().cpu(),
            phase_id=phase_id,
            legal_mask=legal_mask.detach().cpu(),
            primary_index=primary_index_full,
            capture_index=capture_index_full,
            primary_log_prob=primary_log_prob.detach().cpu(),
            capture_log_prob=(
                capture_log_prob.detach().cpu()
                if capture_log_prob is not None
                else None
            ),
            value=value.detach().cpu(),
        )
        return move

    # ------------------------------------------------------------------

    def _select(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        if self.mode == "argmax" or self.temperature <= 1e-6:
            idx = int(torch.argmax(masked_logits).item())
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            return idx, log_probs[idx]

        scaled = masked_logits / self.temperature
        log_probs = torch.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        # torch.multinomial requires probs to sum to 1; guarantee no NaNs.
        if not torch.isfinite(probs).all():
            probs = torch.where(
                torch.isfinite(probs), probs, torch.zeros_like(probs)
            )
            probs = probs / probs.sum().clamp(min=1e-9)
        idx = int(
            torch.multinomial(probs.cpu(), 1, generator=self._gen).item()
        )
        return idx, log_probs[idx]
