"""ai/neural_evaluator.py — Neural-net leaf evaluator for GameAI's negamax.

Wraps a trained NMMNet checkpoint and exposes evaluate(board) -> int from
board.turn's perspective — the same sign convention as ai.heuristics.evaluate.
The score is bounded to ±scale (default 1000) via tanh so it stays well below
the INF sentinel used for terminal positions.

Usage::

    from ai.neural_evaluator import NeuralEvaluator
    from ai.game_ai import GameAI

    ev = NeuralEvaluator.from_checkpoint("learned_ai/checkpoints/latest.pt")
    ai = GameAI(color="W", difficulty=2, neural_evaluator=ev)
    move = ai.choose_move(board)
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from game.board import BoardState
from learned_ai.models.action_encoder import get_legal_mask
from learned_ai.models.state_encoder import detect_phase, encode_state


class NeuralEvaluator:
    """Leaf evaluator that uses a NMMNet value head instead of the heuristic."""

    def __init__(
        self,
        model,
        device: str = "cpu",
        scale: int = 1000,
    ) -> None:
        """
        model  : NMMNet (any module whose .forward() returns {"value": tensor}).
        device : torch device string.  Default "cpu" — game server typically has no GPU.
        scale  : score magnitude at tanh saturation; keep well below INF = 10_000_000.
        """
        self.model = model
        self.device = torch.device(device)
        self.scale = scale
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, board: BoardState) -> int:
        """Return score from board.turn's perspective (positive = good for board.turn).

        The net's value head is trained with +1 for the winner and -1 for the
        loser from the side-to-move's perspective, so no sign flip is needed.
        """
        state = encode_state(board).to(self.device)
        phase_id = detect_phase(board)
        mask = get_legal_mask(board).to(self.device)
        out = self.model.forward(state, phase_id=phase_id, legal_mask=mask)
        raw = float(out["value"].item())
        return int(math.tanh(raw) * self.scale)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        scale: int = 1000,
    ) -> "NeuralEvaluator":
        """Load NMMNet weights from a training checkpoint (.pt file)."""
        from learned_ai.models.backbone import NMMNet

        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        mc: dict = {}
        if isinstance(ckpt, dict):
            mc = ckpt.get("model_config", {})
            state_dict = ckpt.get("model", ckpt)
        else:
            state_dict = ckpt

        model = NMMNet(
            backbone_hidden=tuple(mc.get("backbone_hidden", (256, 256, 128))),
            head_hidden=tuple(mc.get("head_hidden", (64,))),
            dropout=0.0,
        )
        model.load_state_dict(state_dict)
        return cls(model, device=device, scale=scale)
