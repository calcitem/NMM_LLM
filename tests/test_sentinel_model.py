"""Tests for learned_ai/sentinel/model.py (SentinelNet + loss)."""

from __future__ import annotations

import torch

from learned_ai.sentinel.feature_builder import FEATURE_DIM
from learned_ai.sentinel.model import SentinelNet, SentinelOutput, sentinel_loss


def test_forward_shape_batch():
    net = SentinelNet(input_dim=FEATURE_DIM, hidden_dims=[64, 32], dropout=0.0)
    x = torch.randn(8, FEATURE_DIM)
    out = net(x)
    assert isinstance(out, SentinelOutput)
    for t in out.as_dict().values():
        assert t.shape == (8,)


def test_forward_single_vector():
    net = SentinelNet(input_dim=FEATURE_DIM, hidden_dims=[64, 32])
    x = torch.randn(FEATURE_DIM)
    out = net(x)
    # B==1 squeezes to shape (1,)
    for t in out.as_dict().values():
        assert t.shape == (1,)


def test_output_ranges():
    net = SentinelNet(input_dim=FEATURE_DIM, hidden_dims=[64, 32])
    x = torch.randn(32, FEATURE_DIM) * 5.0
    out = net(x)
    assert torch.all(out.mistake_risk >= 0.0) and torch.all(out.mistake_risk <= 1.0)
    assert torch.all(out.opportunity_score >= 0.0) and torch.all(out.opportunity_score <= 1.0)
    assert torch.all(out.turning_point_confidence >= 0.0) and torch.all(out.turning_point_confidence <= 1.0)
    assert torch.all(out.trajectory_value_delta >= -1.0) and torch.all(out.trajectory_value_delta <= 1.0)


def test_default_arch_builds():
    net = SentinelNet()  # default [256,128,64]
    out = net(torch.randn(4, FEATURE_DIM))
    assert out.mistake_risk.shape == (4,)


def test_backward_pass():
    net = SentinelNet(input_dim=FEATURE_DIM, hidden_dims=[64, 32])
    x = torch.randn(16, FEATURE_DIM)
    out = net(x)
    targets = {
        "mistake_risk": torch.rand(16),
        "opportunity_score": torch.rand(16),
        "trajectory_value_delta": torch.rand(16) * 2 - 1,
        "turning_point_confidence": torch.rand(16),
    }
    weights = torch.rand(16) + 0.1
    losses = sentinel_loss(out, targets, sample_weight=weights)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    # at least one trunk parameter received a gradient
    grads = [p.grad for p in net.trunk.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_loss_without_sample_weight():
    net = SentinelNet(input_dim=FEATURE_DIM, hidden_dims=[32])
    out = net(torch.randn(5, FEATURE_DIM))
    targets = {
        "mistake_risk": torch.rand(5),
        "opportunity_score": torch.rand(5),
        "trajectory_value_delta": torch.rand(5) * 2 - 1,
        "turning_point_confidence": torch.rand(5),
    }
    losses = sentinel_loss(out, targets)
    assert torch.isfinite(losses["total"])
