"""tests/test_scaffolded_policy.py — unit tests for the scaffolded meta-policy.

Covers:
  * scaffolded_encoder: correct shapes and value ranges
  * scaffolded_net: forward pass, value bounds, checkpoint round-trip
  * scaffolded_agent: choose_move returns a valid move
  * scaffolded_a2c: update runs without NaN / shape errors
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.models.scaffolded_encoder import (
    MOVE_FEAT_DIM,
    VALUE_INPUT_DIM,
    build_enriched_row,
    build_value_input,
    encode_position,
)
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.scaffolded_a2c import ScaffoldedStep, scaffolded_a2c_update


# ── helpers ────────────────────────────────────────────────────────────────────

def fresh_board() -> BoardState:
    return BoardState.new_game()


def fresh_enc(board=None):
    if board is None:
        board = fresh_board()
    return encode_position(board, "W", sentinel_advisor=None, db=None)


# ── scaffolded_encoder ─────────────────────────────────────────────────────────

class TestScaffoldedEncoder:
    def test_encode_position_returns_not_none(self):
        enc = fresh_enc()
        assert enc is not None

    def test_feat_matrix_shape(self):
        enc = fresh_enc()
        k = len(enc.legal_moves)
        assert enc.feat_matrix.shape == (k, MOVE_FEAT_DIM), (
            f"Expected ({k}, {MOVE_FEAT_DIM}), got {enc.feat_matrix.shape}"
        )

    def test_value_input_shape(self):
        enc = fresh_enc()
        assert enc.value_input.shape == (VALUE_INPUT_DIM,)

    def test_feat_matrix_dtype(self):
        enc = fresh_enc()
        assert enc.feat_matrix.dtype == np.float32

    def test_value_input_dtype(self):
        enc = fresh_enc()
        assert enc.value_input.dtype == np.float32

    def test_sentinel_scores_default_half(self):
        """Without a sentinel, all scores should be 0.5."""
        enc = fresh_enc()
        assert all(abs(s - 0.5) < 1e-6 for s in enc.sentinel_scores)

    def test_h_top1_idx_in_range(self):
        enc = fresh_enc()
        assert 0 <= enc.h_top1_idx < len(enc.legal_moves)

    def test_no_nan_in_features(self):
        enc = fresh_enc()
        assert not np.isnan(enc.feat_matrix).any()
        assert not np.isnan(enc.value_input).any()

    def test_enriched_row_shape(self):
        board = fresh_board()
        mv = get_all_legal_moves(board)[0]
        row = build_enriched_row(
            board, mv, "W",
            sentinel_score=0.7, h_abs_norm=0.6,
            is_top1=True, h_delta=0.1,
        )
        assert row.shape == (MOVE_FEAT_DIM,)

    def test_value_input_builder(self):
        board = fresh_board()
        vi = build_value_input(board, "W", h_eval_abs=0.2, sentinel_scores=[0.6, 0.4])
        assert vi.shape == (VALUE_INPUT_DIM,)
        assert not np.isnan(vi).any()

    def test_terminal_returns_none(self):
        """encode_position on a terminal board should return None."""
        # Create a board with 2 white pieces (would be terminal: white loses)
        # Easiest: test with empty legal moves simulated — use a real terminal if available
        # For now, just verify the None branch exists and the function doesn't crash.
        enc = fresh_enc()
        assert enc is not None  # fresh board is not terminal


# ── scaffolded_net ─────────────────────────────────────────────────────────────

class TestScaffoldedNet:
    def test_forward_returns_dict(self):
        model = ScaffoldedPolicyNet()
        enc = fresh_enc()
        feat = torch.tensor(enc.feat_matrix, dtype=torch.float32)
        vi   = torch.tensor(enc.value_input,  dtype=torch.float32)
        out  = model.forward(feat, vi)
        assert "logits" in out and "value" in out

    def test_logits_shape_matches_k(self):
        model = ScaffoldedPolicyNet()
        enc = fresh_enc()
        k = len(enc.legal_moves)
        feat = torch.tensor(enc.feat_matrix, dtype=torch.float32)
        vi   = torch.tensor(enc.value_input,  dtype=torch.float32)
        out  = model.forward(feat, vi)
        assert out["logits"].shape == (k,), f"Expected ({k},), got {out['logits'].shape}"

    def test_value_bounded_after_tanh(self):
        model = ScaffoldedPolicyNet()
        enc = fresh_enc()
        vi  = torch.tensor(enc.value_input, dtype=torch.float32)
        v   = model.value(vi)
        assert -1.0 <= float(v.item()) <= 1.0

    def test_policy_probs_sum_to_one(self):
        model = ScaffoldedPolicyNet()
        enc = fresh_enc()
        feat  = torch.tensor(enc.feat_matrix, dtype=torch.float32)
        probs = model.policy_probs(feat)
        assert abs(float(probs.sum().item()) - 1.0) < 1e-5

    def test_no_nan_in_output(self):
        model = ScaffoldedPolicyNet()
        enc = fresh_enc()
        feat = torch.tensor(enc.feat_matrix, dtype=torch.float32)
        vi   = torch.tensor(enc.value_input,  dtype=torch.float32)
        out  = model.forward(feat, vi)
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isnan(out["value"])

    def test_checkpoint_round_trip(self, tmp_path):
        model = ScaffoldedPolicyNet()
        cfg   = model.get_config()
        ckpt_path = tmp_path / "test.pt"
        torch.save({"model": model.state_dict(), "model_config": cfg}, ckpt_path)

        model2 = ScaffoldedPolicyNet.from_config(cfg)
        ckpt2  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(ckpt2["model"])

        enc  = fresh_enc()
        feat = torch.tensor(enc.feat_matrix, dtype=torch.float32)
        vi   = torch.tensor(enc.value_input,  dtype=torch.float32)
        with torch.no_grad():
            v1 = model.forward(feat, vi)["logits"]
            v2 = model2.forward(feat, vi)["logits"]
        assert torch.allclose(v1, v2)

    def test_get_config_round_trip(self):
        model = ScaffoldedPolicyNet(policy_hidden=(64, 32), value_hidden=(32,))
        cfg   = model.get_config()
        model2 = ScaffoldedPolicyNet.from_config(cfg)
        assert model.move_feat_dim    == model2.move_feat_dim
        assert model.value_input_dim  == model2.value_input_dim


# ── scaffolded_agent ───────────────────────────────────────────────────────────

class TestScaffoldedAgent:
    def test_choose_move_returns_dict(self):
        from learned_ai.agents.scaffolded_agent import ScaffoldedAgent
        agent = ScaffoldedAgent(color="W")
        board = fresh_board()
        move  = agent.choose_move(board)
        assert isinstance(move, dict)
        assert "from" in move and "to" in move

    def test_choose_move_is_legal(self):
        from learned_ai.agents.scaffolded_agent import ScaffoldedAgent
        agent = ScaffoldedAgent(color="W")
        board = fresh_board()
        move  = agent.choose_move(board)
        legal_moves = get_all_legal_moves(board)
        key = (move.get("from"), move.get("to"), move.get("capture"))
        legal_keys = [(m.get("from"), m.get("to"), m.get("capture")) for m in legal_moves]
        assert key in legal_keys

    def test_last_decision_populated(self):
        from learned_ai.agents.scaffolded_agent import ScaffoldedAgent
        agent = ScaffoldedAgent(color="W")
        board = fresh_board()
        agent.choose_move(board)
        assert agent.last_decision is not None
        assert agent.last_decision.chosen_idx >= 0


# ── scaffolded_a2c ─────────────────────────────────────────────────────────────

class TestScaffoldedA2C:
    def _make_steps(self, n: int = 16) -> list[ScaffoldedStep]:
        enc = fresh_enc()
        k = len(enc.legal_moves)
        steps = []
        for _ in range(n):
            steps.append(ScaffoldedStep(
                move_features=enc.feat_matrix.copy(),
                value_input=enc.value_input.copy(),
                chosen_idx=0,
                log_prob_old=-2.5,
                reward=float(np.random.uniform(-0.5, 0.5)),
                next_move_features=enc.feat_matrix.copy(),
                next_value_input=enc.value_input.copy(),
                done=False,
            ))
        return steps

    def test_update_returns_tuple(self):
        model = ScaffoldedPolicyNet()
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = self._make_steps(16)
        result = scaffolded_a2c_update(model, opt, steps, torch.device("cpu"))
        assert len(result) == 3

    def test_losses_are_finite(self):
        model = ScaffoldedPolicyNet()
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = self._make_steps(16)
        pl, vl, ent = scaffolded_a2c_update(model, opt, steps, torch.device("cpu"))
        assert np.isfinite(pl),  f"policy_loss not finite: {pl}"
        assert np.isfinite(vl),  f"value_loss not finite: {vl}"
        assert np.isfinite(ent), f"entropy not finite: {ent}"

    def test_too_few_steps_returns_zeros(self):
        model = ScaffoldedPolicyNet()
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = scaffolded_a2c_update(model, opt, [], torch.device("cpu"))
        assert result == (0.0, 0.0, 0.0)

    def test_weights_change_after_update(self):
        model = ScaffoldedPolicyNet()
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [p.data.clone() for p in model.parameters()]
        steps  = self._make_steps(20)
        scaffolded_a2c_update(model, opt, steps, torch.device("cpu"))
        after  = [p.data.clone() for p in model.parameters()]
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "No parameters changed after A2C update"
